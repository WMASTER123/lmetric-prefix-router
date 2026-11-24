"""Local text-serving development front door for the preinstalled 0.23 DP host."""

import argparse
import asyncio
import json
import time
from collections import Counter, deque
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import version

import aiohttp
from aiohttp import web

from .cache023 import BLOCK, GROUPS, CacheView
from .events023 import EventStreams
from .fingerprint023 import HostFingerprint
from .managed023 import LocalEngine, ManagedReplay
from .policy import Replica, choose_rank
from .provider import BUNDLE_ID, DIST
from .rank_metrics import poll_gauges
from .receipts import UsageObserver
from .timing import RequestTiming
from .transport import BackendConfig, BackendConnectionPool, BackendPoolSaturated


@dataclass(eq=False)
class Reservation:
    rank: int | None
    prefill: int
    reason: str


class NativePool:
    def __init__(self, policy="combined", rank_count=2, block_size=BLOCK, specs=GROUPS):
        if policy not in {"combined", "native", "prefix"}:
            raise ValueError("unknown shared-endpoint policy")
        self.policy = policy
        if rank_count not in (1, 2):
            raise ValueError("native023 supports one or two DP ranks")
        self.rank_count = rank_count
        self.expected_ranks = set(range(rank_count))
        self.observed_ranks = None
        self.next_tie = 0
        self.views = tuple(
            CacheView(rank, block_size, specs) for rank in range(self.rank_count)
        )
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
    backend,
    event_port,
    fingerprint,
    policy="combined",
    rank_count=2,
    replay_port=None,
    engine=None,
):
    state = NativePool(
        policy,
        rank_count,
        engine.block_size if engine else BLOCK,
        engine.specs if engine else GROUPS,
    )
    counters = Counter()
    reasons = Counter()
    timings = deque(maxlen=64)
    installed = version(DIST)
    pool = BackendConnectionPool(BackendConfig("shared-dp", backend))
    app = web.Application(client_max_size=8 * 1024 * 1024)

    async def resources(_app):
        streams = (
            ManagedReplay(state.views, engine)
            if engine
            else EventStreams(state.views, event_port, replay_port)
        )
        async with streams:
            if engine and policy != "native":
                with suppress(aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                    await fingerprint.verify_chat(backend)
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
                extension_id=BUNDLE_ID,
                version=installed,
                backend=backend.rstrip("/"),
                ready=bool(
                    engine
                    and state.expected_ranks == state.observed_ranks
                    and all(v.ready for v in state.views)
                    and all(
                        r in state.gauges
                        and 0 <= time.monotonic() - state.gauges[r][0] < 1
                        for r in state.expected_ranks
                    )
                ),
                counters=dict(counters),
                route_reasons=dict(reasons),
                expected_ranks=sorted(state.expected_ranks),
                observed_metric_ranks=(
                    sorted(state.observed_ranks)
                    if state.observed_ranks is not None
                    else None
                ),
                active=len(state.active),
                topology={
                    "dp": rank_count,
                    "tp": engine.tp if engine else None,
                    "source": "engine_process" if engine else "configuration",
                },
                chat_verified=getattr(fingerprint, "chat_verified", False),
                engine_generation=engine.digest if engine else None,
                replay_endpoints=(
                    {
                        str(r): f"tcp://127.0.0.1:{engine.replay_port + r}"
                        for r in range(rank_count)
                    }
                    if engine
                    else {}
                ),
                recent_timings=list(timings),
                ranks=[
                    dict(
                        rank=v.rank,
                        trusted=v.ready,
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
        timing = RequestTiming()
        try:
            body = await request.read()
            payload = json.loads(body)
        except (UnicodeDecodeError, ValueError) as exc:
            raise web.HTTPBadRequest(text="invalid JSON") from exc
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(text="expected an object")
        timing.mark("parsed")
        fp = None if policy == "native" else fingerprint(request.path, payload)
        timing.mark("fingerprinted")
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
            timing.mark("routing_complete")

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
                timing=timing,
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
                    timing.mark("first_body")
                    observer.feed(chunk)
                    if observer.has_content:
                        entry.prefill = 0
                        timing.mark("first_visible_content")
                    await stream.write(chunk)
                await stream.write_eof()
                observer.finish()
                timing.http_status = response.status
                timing.terminal = "completed"
                if 200 <= response.status < 300:
                    if (
                        predicted is None
                        or fp is None
                        or observer.prompt_tokens is None
                    ):
                        counters["unpaired"] += 1
                    elif observer.prompt_tokens != fp[0]:
                        counters["mismatched"] += 1
                        counters["prompt_length_mismatch"] += 1
                    elif observer.cached_tokens is None:
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
                timing.node_id = "native" if entry.rank is None else str(entry.rank)
            timing.mark("finished")
            timings.append(timing.snapshot())

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
    parser.add_argument("--dp-size", type=int, choices=(1, 2))
    parser.add_argument("--engine-pid", type=int)
    parser.add_argument("--replay-port", type=int)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--policy", choices=["combined", "prefix", "native"], default="combined"
    )
    args = parser.parse_args()
    engine = None
    if args.engine_pid is not None:
        if args.replay_port is None:
            parser.error("--engine-pid requires --replay-port")
        engine = LocalEngine(
            args.engine_pid,
            args.backend,
            args.model_path,
            args.served_model_name,
            args.event_port,
            args.replay_port,
        )
        if args.dp_size is not None and args.dp_size != engine.dp:
            parser.error("--dp-size disagrees with engine launch")
    fp = HostFingerprint(
        args.model_path, args.served_model_name, engine.block_size if engine else BLOCK
    )
    web.run_app(
        application(
            args.backend,
            args.event_port,
            fp,
            args.policy,
            engine.dp if engine else args.dp_size or 2,
            args.replay_port,
            engine,
        ),
        host="127.0.0.1",
        port=args.port,
        handler_cancellation=True,
    )


if __name__ == "__main__":
    main()
