import time

import pytest

from lmetric_prefix_router.cache023 import BLOCK, GROUPS, CacheView, block_key
from lmetric_prefix_router.native023 import NativePool


def store(group, keys):
    return [
        "BlockStored",
        keys,
        None,
        [],
        BLOCK,
        None,
        "GPU",
        None,
        [],
        group,
        GROUPS[group],
        None,
    ]


def test_reset_alignment_and_last_token_boundary():
    v = CacheView(0)
    keys = [block_key(i) for i in (3, 4)]
    v.apply(0, [0, [store(0, keys)], 0])
    assert v.match(keys, 4097) is None
    v.apply(
        1,
        [
            0,
            [
                ["AllBlocksCleared"],
                store(0, keys),
                *[store(g, keys[1:]) for g in (1, 2, 3)],
            ],
            0,
        ],
    )
    assert v.match(keys, 4096) == 0
    assert v.match(keys, 4097) == 4096
    v.apply(2, [0, [["BlockRemoved", keys[:1], "GPU", 0]], 0])
    assert v.match(keys, 4097) == 0
    v.apply(4, [0, [store(0, keys)], 0])
    assert v.match(keys, 4097) is None


def test_bad_removal_invalidates_view():
    v = CacheView(1)
    with pytest.raises(ValueError, match="corresponding"):
        v.apply(1, [0, [["AllBlocksCleared"], ["BlockRemoved", [1], "GPU", 3]], 1])
    assert not v.trusted


def test_extra_key_stores_and_removals_preserve_other_prefixes():
    view = CacheView(0)
    plain, salted = block_key(1), block_key(2)
    events = [store(group, [plain, salted]) for group in range(len(GROUPS))]
    for event in events:
        event[8] = [None, ["probe-salt"]]
    view.apply(0, [0, [["AllBlocksCleared"], *events], 0])
    assert view.match([plain], 2600) == BLOCK
    assert view.match([salted], 2600) == 0
    view.apply(1, [0, [["BlockRemoved", [salted], "GPU", 0]], 0])
    assert view.trusted and view.match([plain], 2600) == BLOCK
    clash = store(1, [salted])
    with pytest.raises(ValueError, match="ambiguous"):
        view.apply(2, [0, [clash], 0])
    assert not view.trusted


@pytest.mark.parametrize(
    "policy,route", [("combined", 1), ("prefix", 1), ("native", None)]
)
def test_shared_host_decision(policy, route):
    pool = NativePool(policy)
    pool.observed_ranks = {0, 1}
    keys = [block_key(1)]
    for v in pool.views:
        events = [["AllBlocksCleared"]]
        if v.rank:
            events.extend(store(g, keys) for g in range(4))
        v.apply(0, [0, events, v.rank])
    reservation, _ = pool.reserve((2200, keys))
    assert reservation.rank == route
    assert reservation in pool.active
    if policy == "combined":
        pool.views[0].apply(1, [0, [store(g, keys) for g in range(4)], 0])
        pool.gauges = {r: (time.monotonic(), 0, 0) for r in (0, 1)}
        assert pool.reserve((2200, keys))[0].rank is None
        pool.views[1].invalidate()
        assert pool.reserve((2200, keys))[1] is None


def test_shared_http_entry_owns_rank_and_releases_stream_reservations(monkeypatch):
    import asyncio
    from contextlib import AsyncExitStack, asynccontextmanager

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from lmetric_prefix_router import native023

    keys = [block_key(7)]
    feeds = []
    observed = []

    @asynccontextmanager
    async def feed(views, port, replay_port):
        feeds.extend(views)
        for v in views:
            events = [["AllBlocksCleared"]]
            if v.rank == 1:
                events.extend(store(g, keys) for g in range(4))
            v.apply(1, [0, events, v.rank])
        yield

    monkeypatch.setattr(native023, "EventStreams", feed)

    async def generate(request):
        payload = await request.json()
        observed.append(request.headers.get("X-Data-Parallel-Rank"))
        if payload["stream"]:
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b'data: {"choices":[{"text":"a"}]}\n\n')
            await response.write(
                b'data: {"usage":{"prompt_tokens":2200,'
                b'"prompt_tokens_details":{"cached_tokens":2048}}}\n\n'
            )
            await response.write(b"data: [DONE]\n\n")
            return response
        return web.json_response({"usage": {"prompt_tokens": 2200}})

    async def run():
        async with AsyncExitStack() as stack:
            upstream = web.Application()

            async def metrics(request):
                text = "\n".join(
                    f'vllm:num_requests_{kind}{{engine="{rank}"}} 0'
                    for rank in (0, 1)
                    for kind in ("running", "waiting")
                )
                return web.Response(text=text)

            upstream.router.add_get("/metrics", metrics)
            upstream.router.add_post("/v1/completions", generate)
            server = await stack.enter_async_context(TestServer(upstream))
            app = native023.application(
                str(server.make_url("/")), 18300, lambda *a: (2200, keys)
            )
            client = await stack.enter_async_context(TestClient(TestServer(app)))
            url = "/v1/completions"
            headers = {"X-Data-Parallel-Rank": "0"}
            for _ in range(100):
                async with client.get("/status") as status:
                    if (await status.json())["observed_metric_ranks"] == [0, 1]:
                        break
                await asyncio.sleep(0.01)
            async with client.post(
                url, json={"stream": True}, headers=headers
            ) as response:
                assert response.headers["X-Lmetric-Route"] == "1"
                assert response.headers["X-Lmetric-Reason"] == "longest_prefix"
                assert response.headers["X-Lmetric-Expected-Cached"] == "2048"
                assert b"[DONE]" in await response.read()
            feeds[0].invalidate()
            async with client.post(
                url, json={"stream": False}, headers=headers
            ) as response:
                assert response.headers["X-Lmetric-Route"] == "native"
                assert response.headers["X-Lmetric-Reason"] == "cache_view_unavailable"
                await response.read()
            async with client.get("/status") as response:
                state = await response.json()
            assert observed == ["1", None]
            assert state["active"] == 0
            assert state["route_reasons"] == {
                "longest_prefix": 1,
                "cache_view_unavailable": 1,
            }
            assert state["counters"] == {
                "pinned": 1,
                "matched": 1,
                "delegated": 1,
                "unpaired": 1,
            }

    asyncio.run(run())


def test_real_zmq_frames_and_disconnect_invalidation():
    import asyncio

    import msgspec
    import zmq
    import zmq.asyncio

    from lmetric_prefix_router.events023 import consume

    async def run():
        ctx = zmq.asyncio.Context()
        sub, pub = ctx.socket(zmq.SUB), ctx.socket(zmq.XPUB)
        monitor = sub.get_monitor_socket(events=zmq.EVENT_DISCONNECTED)
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        port = sub.bind_to_random_port("tcp://127.0.0.1")
        view = CacheView(0)
        reader = asyncio.create_task(consume(view, sub, monitor))
        try:
            pub.connect(f"tcp://127.0.0.1:{port}")
            assert await asyncio.wait_for(pub.recv(), 5) == b"\x01"
            events = [["AllBlocksCleared"], *[store(g, [7]) for g in range(4)]]
            await pub.send_multipart(
                [b"", bytes(8), msgspec.msgpack.encode([0, events, 0])]
            )
            for _ in range(250):
                if view.trusted:
                    break
                await asyncio.sleep(0.01)
            assert view.match([block_key(7)], 2200) == BLOCK
            pub.close(0)
            await asyncio.wait_for(reader, 5)
            assert view.match([block_key(7)], 2200) is None
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            for socket in (monitor, sub, pub):
                socket.close(0)
            ctx.destroy(linger=0)

    # pyzmq uses add_reader, as on the Linux host.
    from contextlib import closing

    with closing(asyncio.SelectorEventLoop()) as loop:
        loop.run_until_complete(run())
