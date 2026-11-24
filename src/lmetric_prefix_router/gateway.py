"""Loopback qualification gateway for two independently managed TP endpoints."""

import argparse
import asyncio
from collections.abc import Callable
from dataclasses import dataclass

import aiohttp
from aiohttp import web

from .policy import ExternalBalancer, Replica
from .transport import BackendConfig, BackendConnectionPool, BackendPoolSaturated


@dataclass(frozen=True)
class Snapshot:
    """A trusted reader supplies endpoint-ordered evidence and rechecks its epoch."""

    prompt_tokens: int
    replicas: tuple[Replica, ...]
    still_current: Callable[[], bool]


Reader = Callable[[str, bytes, tuple[int, ...]], Snapshot]


def application(urls: tuple[str, str], reader: Reader | None = None) -> web.Application:
    """Without a reader, run the common least-inflight, rotating-tie baseline."""
    if urls[0].rstrip("/") == urls[1].rstrip("/"):
        raise ValueError("two distinct endpoint addresses are required")
    pools = [
        BackendConnectionPool(BackendConfig(str(i), url)) for i, url in enumerate(urls)
    ]
    inflight = [0, 0]
    balancer = ExternalBalancer()
    app = web.Application(client_max_size=8 * 1024 * 1024)

    async def sessions(_app):
        try:
            for pool in pools:
                await pool.start()
            yield
        finally:
            for pool in pools:
                await pool.close()

    async def forward(request):
        body = await request.read()
        evidence = reader(request.path, body, tuple(inflight)) if reader else None
        if evidence and len(evidence.replicas) != len(pools):
            raise web.HTTPServiceUnavailable(text="incomplete endpoint membership")
        base = tuple(Replica(None, n, 0, None) for n in inflight)
        # Both arms use the same local in-flight ledger, including queued requests.
        replicas = (
            tuple(
                Replica(
                    r.matched_tokens, n, r.pending_prefill_tokens, r.gauge_age_seconds
                )
                for r, n in zip(evidence.replicas, inflight, strict=True)
            )
            if evidence
            else base
        )
        rank = balancer.choose(
            evidence.prompt_tokens if evidence else 0,
            replicas,
            use_prefix=evidence is not None,
        )
        inflight[rank] += 1

        def validate():
            if evidence and not evidence.still_current():
                raise web.HTTPServiceUnavailable(text="routing evidence expired")

        stream = None
        try:
            async with pools[rank].request(
                "POST",
                request.path,
                data=body,
                allow_redirects=False,
                headers={
                    "Content-Type": "application/json",
                    "Accept-Encoding": "identity",
                },
                before_dispatch=validate,
            ) as response:
                headers = {"X-Qualification-Endpoint": str(rank)}
                for key in ("Content-Type", "Content-Encoding", "Retry-After"):
                    if key in response.headers:
                        headers[key] = response.headers[key]
                stream = web.StreamResponse(status=response.status, headers=headers)
                await stream.prepare(request)
                async for chunk in response.content.iter_any():
                    await stream.write(chunk)
                await stream.write_eof()
                return stream
        except BackendPoolSaturated as exc:
            raise web.HTTPServiceUnavailable(
                text="backend admission saturated"
            ) from exc
        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as exc:
            if stream and stream.prepared:
                # Abort the downstream HTTP message without a successful terminal chunk.
                if request.transport:
                    request.transport.abort()
                raise
            raise web.HTTPBadGateway(text="backend unavailable") from exc
        finally:
            inflight[rank] -= 1

    app.cleanup_ctx.append(sessions)
    for path in ("/v1/completions", "/v1/chat/completions"):
        app.router.add_post(path, forward)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs=2, required=True)
    parser.add_argument("--port", type=int, default=18300)
    args = parser.parse_args()
    web.run_app(
        application(tuple(args.backends)),
        host="127.0.0.1",
        port=args.port,
        handler_cancellation=True,
    )


if __name__ == "__main__":
    main()
