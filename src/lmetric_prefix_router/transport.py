# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM-HUST/vllm-hust-prefix-router at bcc9691.
"""Per-backend HTTP connection pools for long-lived generation streams."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from .timing import RequestTiming


@dataclass(frozen=True)
class BackendPoolConfig:
    """Connection and admission limits applied independently per backend."""

    max_connections: int = 512
    max_pending_requests: int = 512
    queue_timeout_s: float = 1.0
    connect_timeout_s: float = 10.0
    request_timeout_s: float = 6 * 60 * 60
    keepalive_timeout_s: float = 30.0
    dns_cache_ttl_s: int = 300

    def __post_init__(self) -> None:
        for name in ("max_connections", "max_pending_requests", "dns_cache_ttl_s"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_connections == 0:
            raise ValueError("max_connections must be greater than zero")
        for name in (
            "queue_timeout_s",
            "connect_timeout_s",
            "request_timeout_s",
            "keepalive_timeout_s",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")


@dataclass(frozen=True)
class BackendConfig:
    node_id: str
    url: str
    pool: BackendPoolConfig = field(default_factory=BackendPoolConfig)

    def __post_init__(self):
        value = urlsplit(self.url)
        if value.scheme != "http" or value.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError("qualification backends must be local HTTP services")
        if (
            value.username
            or value.password
            or value.query
            or value.fragment
            or value.path not in {"", "/"}
        ):
            raise ValueError("invalid backend authority or path")


class BackendPoolSaturated(RuntimeError):
    """Raised before dispatch when a backend cannot admit another request."""


@dataclass(frozen=True)
class BackendPoolStats:
    """Point-in-time admission and queue statistics."""

    node_id: str
    active_requests: int
    queued_requests: int
    max_connections: int
    max_pending_requests: int
    admitted_requests: int
    rejected_requests: int
    total_queue_wait_s: float
    max_queue_wait_s: float


class BackendConnectionPool:
    """A connection pool and bounded admission queue for exactly one backend.

    Generation responses hold an HTTP/1.1 connection for their full streaming
    lifetime. A single shared aiohttp session therefore imposes its connector
    limit across every node. This class gives each node an independent connector
    and rejects excessive queueing before it can inflate TTFT for tens of seconds.
    """

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        self._slots = asyncio.Semaphore(config.pool.max_connections)
        self._active_requests = 0
        self._queued_requests = 0
        self._admitted_requests = 0
        self._rejected_requests = 0
        self._total_queue_wait_s = 0.0
        self._max_queue_wait_s = 0.0

    @property
    def session(self) -> aiohttp.ClientSession:
        """Return the started client session."""
        if self._session is None or self._session.closed:
            raise RuntimeError(f"backend pool {self.config.node_id!r} is not started")
        return self._session

    async def start(self) -> None:
        """Create this backend's independent keep-alive connector."""
        if self._session is not None and not self._session.closed:
            return
        pool = self.config.pool
        connector = aiohttp.TCPConnector(
            limit=pool.max_connections,
            limit_per_host=pool.max_connections,
            keepalive_timeout=pool.keepalive_timeout_s,
            ttl_dns_cache=pool.dns_cache_ttl_s,
        )
        timeout = aiohttp.ClientTimeout(
            total=pool.request_timeout_s,
            connect=pool.connect_timeout_s,
            sock_connect=pool.connect_timeout_s,
            sock_read=pool.request_timeout_s,
        )
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            auto_decompress=False,
        )

    async def close(self) -> None:
        """Close idle and active connector resources."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    def stats(self) -> BackendPoolStats:
        """Return current admission metrics without exposing aiohttp internals."""
        return BackendPoolStats(
            node_id=self.config.node_id,
            active_requests=self._active_requests,
            queued_requests=self._queued_requests,
            max_connections=self.config.pool.max_connections,
            max_pending_requests=self.config.pool.max_pending_requests,
            admitted_requests=self._admitted_requests,
            rejected_requests=self._rejected_requests,
            total_queue_wait_s=self._total_queue_wait_s,
            max_queue_wait_s=self._max_queue_wait_s,
        )

    @asynccontextmanager
    async def _admit(self, timing: RequestTiming | None = None) -> AsyncIterator[None]:
        pool = self.config.pool
        if timing is not None:
            timing.mark("admission_start")
        if self._slots.locked() and self._queued_requests >= pool.max_pending_requests:
            self._rejected_requests += 1
            raise BackendPoolSaturated(
                f"backend {self.config.node_id!r} admission queue is full"
            )

        started_at = time.perf_counter()
        self._queued_requests += 1
        try:
            try:
                await asyncio.wait_for(
                    self._slots.acquire(), timeout=pool.queue_timeout_s
                )
            except asyncio.TimeoutError as exc:
                self._rejected_requests += 1
                raise BackendPoolSaturated(
                    f"backend {self.config.node_id!r} admission timed out"
                ) from exc
        finally:
            self._queued_requests -= 1

        queue_wait_s = time.perf_counter() - started_at
        self._total_queue_wait_s += queue_wait_s
        self._max_queue_wait_s = max(self._max_queue_wait_s, queue_wait_s)
        self._active_requests += 1
        self._admitted_requests += 1
        if timing is not None:
            timing.mark("pool_admitted")
        try:
            yield
        finally:
            self._active_requests -= 1
            self._slots.release()

    def _target_url(self, path: str) -> str:
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("upstream path must be an absolute path, not a URL")
        return f"{self.config.url.rstrip('/')}{path}"

    @asynccontextmanager
    async def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        timing: RequestTiming | None = None,
        before_dispatch: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[aiohttp.ClientResponse]:
        """Dispatch one request while holding an admission slot through streaming."""
        async with self._admit(timing):
            # Admission can yield while an incarnation or cache view changes.
            # The synchronous guard runs with the slot held and may refuse the
            # request before constructing its HTTP transport. It cannot retry.
            if before_dispatch is not None:
                before_dispatch()
            async with self.session.request(
                method,
                self._target_url(path),
                headers=headers,
                **kwargs,
            ) as response:
                if timing is not None:
                    timing.mark("upstream_headers")
                yield response
