"""Local text-serving development front door for the preinstalled 0.23 DP host."""

import argparse
import asyncio
import json
import time
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass

import aiohttp
from aiohttp import web

from .cache023 import CacheView
from .events023 import EventStreams
from .fingerprint023 import HostFingerprint
from .policy import Replica, choose_rank
from .rank_metrics import poll_gauges
from .receipts import UsageObserver
from .transport import BackendConfig, BackendConnectionPool, BackendPoolSaturated


@dataclass(eq=False)
class Reservation:
    rank: int | None
    prefill: int
    reason: str


class NativePool:
    def __init__(self, policy="combined", rank_count=2):
        if policy not in {"combined", "native", "prefix"}:
            raise ValueError("unknown shared-endpoint policy")
        self.policy = policy
        if rank_count not in (1, 2):
            raise ValueError("native023 supports one or two DP ranks")
        self.rank_count = rank_count
        self.expected_ranks = set(range(rank_count))
        self.observed_ranks = None
        self.next_tie = 0
        self.views = tuple(CacheView(rank) for rank in range(self.rank_count))
        self.gauges = {}
        self.active = set()

    def reserve(self, fingerprint):
        length, hashes = fingerprint or (0, ())
        now = time.monotonic()
        replicas = []
        for view in self.views:
            rank = view.rank
            count = pending = 0.0
            for row in self.active:
                weight = (
                    1 / self.rank_count if row.rank is None else int(row.rank == rank)
                )
                count += weight
                pending += weight * row.prefill
            gauge = self.gauges.get(rank)
            age = now - gauge[0] if gauge else None
            if gauge and 0 <= age < 1:
                count = max(count, gauge[1] + gauge[2])
            hit = view.match(hashes, length) if fingerprint is not None else None
            replicas.append(Replica(hit, count, pending, age))
        hits = [r.matched_tokens for r in replicas]
        selected = None
        predicted = None
        if self.policy == "native":
            reason = "native_policy"
        elif fingerprint is None:
            reason = "unsupported_input"
        elif self.observed_ranks != self.expected_ranks:
            reason = (
                "topology_unverified"
                if self.observed_ranks is None
                else "topology_mismatch"
            )
        elif None in hits:
            reason = "cache_view_unavailable"
        elif self.rank_count == 1:
            reason = "single_replica"
            predicted = hits[0]
        elif self.policy == "combined":
            selected = choose_rank(length, tuple(replicas))
            if max(hits) == 0:
                reason = "all_miss"
            elif selected is None:
                reason = "equal_hit_native"
            elif len(set(hits)) == 1:
                reason = "equal_hit_local_load"
            else:
                reason = "longest_prefix"
            predicted = hits[selected] if selected is not None else hits[0]
        else:
            owners = [r for r, hit in enumerate(hits) if hit == max(hits)]
            selected = owners[self.next_tie % len(owners)]
            if len(owners) > 1:
                self.next_tie += 1
            predicted = hits[selected]
            reason = "prefix_policy"
        entry = Reservation(selected, length - (predicted or 0), reason)
        self.active.add(entry)
        return entry, predicted


def application(
    backend, event_port, fingerprint, policy="combined", rank_count=2, replay_port=None
):
    state = NativePool(policy, rank_count)
    counters = Counter()
    reasons = Counter()
    pool = BackendConnectionPool(BackendConfig("shared-dp", backend))
    app = web.Application(client_max_size=8 * 1024 * 1024)

    async def resources(_app):
        async with EventStreams(state.views, event_port, replay_port):
            await pool.start()
            poll = asyncio.create_task(poll_gauges(state, backend))
            try:
                yield
            finally:
                poll.cancel()
                with suppress(asyncio.CancelledError):
                    await poll
                await pool.close()

    async def status(request):
        return web.json_response(
            dict(
                profile="native023",
                policy=policy,
                counters=dict(counters),
                route_reasons=dict(reasons),
                expected_ranks=sorted(state.expected_ranks),
                observed_metric_ranks=(
                    sorted(state.observed_ranks)
                    if state.observed_ranks is not None
                    else None
                ),
                active=len(state.active),
                ranks=[
                    dict(
                        rank=v.rank,
                        trusted=v.trusted,
                        sequence=v.sequence,
                        sync_state=v.sync_state,
                    )
                    for v in state.views
                ],
                gauges_fresh=[
                    r in state.gauges and time.monotonic() - state.gauges[r][0] <= 0.6
                    for r in range(state.rank_count)
                ],
            )
        )

    async def forward(request):
        try:
            body = await request.read()
            payload = json.loads(body)
        except (UnicodeDecodeError, ValueError) as exc:
            raise web.HTTPBadRequest(text="invalid JSON") from exc
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(text="expected an object")
        fp = None if policy == "native" else fingerprint(request.path, payload)
        headers = {"Content-Type": "application/json", "Accept-Encoding": "identity"}
        if "Authorization" in request.headers:
            headers["Authorization"] = request.headers["Authorization"]
        entry = predicted = None

        def select():
            nonlocal entry, predicted
            entry, predicted = state.reserve(fp)
            counters["delegated" if entry.rank is None else "pinned"] += 1
            reasons[entry.reason] += 1
            if entry.rank is not None:
                headers["X-Data-Parallel-Rank"] = str(entry.rank)

        observer = UsageObserver(
            streaming=payload.get("stream") is True, observe_content=True
        )
        stream = None
        try:
            async with pool.request(
                "POST",
                request.path,
                data=body,
                headers=headers,
                allow_redirects=False,
                before_dispatch=select,
            ) as response:
                info = {
                    "X-Lmetric-Reason": entry.reason,
                    "X-Lmetric-Route": "native"
                    if entry.rank is None
                    else str(entry.rank),
                    "X-Lmetric-Expected-Cached": "unknown"
                    if predicted is None
                    else str(predicted),
                }
                for key in ("Content-Type", "Content-Encoding", "Retry-After"):
                    if key in response.headers:
                        info[key] = response.headers[key]
                stream = web.StreamResponse(status=response.status, headers=info)
                await stream.prepare(request)
                async for chunk in response.content.iter_any():
                    observer.feed(chunk)
                    if observer.has_content:
                        entry.prefill = 0
                    await stream.write(chunk)
                await stream.write_eof()
                observer.finish()
                if 200 <= response.status < 300:
                    if predicted is None or observer.cached_tokens is None:
                        counters["unpaired"] += 1
                    elif predicted == observer.cached_tokens:
                        counters["matched"] += 1
                    else:
                        counters["mismatched"] += 1
                return stream
        except BackendPoolSaturated as exc:
            raise web.HTTPServiceUnavailable(
                text="backend admission saturated"
            ) from exc
        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as exc:
            if stream is not None and stream.prepared:
                if request.transport:
                    request.transport.abort()
                raise
            raise web.HTTPBadGateway(text="engine unavailable") from exc
        finally:
            if entry is not None:
                state.active.discard(entry)

    app.cleanup_ctx.append(resources)
    app.router.add_get("/status", status)
    for path in ("/v1/completions", "/v1/chat/completions"):
        app.router.add_post(path, forward)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--event-port", type=int, required=True)
    parser.add_argument("--dp-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--replay-port", type=int)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--policy", choices=["combined", "prefix", "native"], default="combined"
    )
    args = parser.parse_args()
    fp = HostFingerprint(args.model_path, args.served_model_name)
    web.run_app(
        application(
            args.backend,
            args.event_port,
            fp,
            args.policy,
            args.dp_size,
            args.replay_port,
        ),
        host="127.0.0.1",
        port=args.port,
        handler_cancellation=True,
    )


if __name__ == "__main__":
    main()
