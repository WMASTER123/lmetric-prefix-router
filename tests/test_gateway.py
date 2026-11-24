import asyncio
from contextlib import AsyncExitStack

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from lmetric_prefix_router.gateway import Snapshot, application
from lmetric_prefix_router.policy import Replica


def test_stream_dispatch_and_epoch_check():
    async def run():
        seen = []
        started = asyncio.Event()
        finish = asyncio.Event()

        async def generate(request):
            assert "X-Data-Parallel-Rank" not in request.headers
            seen.append(await request.text())
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b"data: first\n\n")
            if seen[-1] == "hold":
                started.set()
                await asyncio.wait_for(finish.wait(), timeout=5)
            await response.write(b"data: [DONE]\n\n")
            return response

        async with AsyncExitStack() as stack:
            urls = []
            for _ in range(2):
                backend = web.Application()
                backend.router.add_post("/v1/completions", generate)
                server = await stack.enter_async_context(TestServer(backend))
                urls.append(str(server.make_url("/")))
            replicas = (Replica(0, 0, 0, 0), Replica(2, 0, 0, 0))

            def reader(path, body, counts):
                if body in {b"warm", b"stale"}:
                    return Snapshot(2, replicas, lambda: body != b"stale")
                return None

            client = await stack.enter_async_context(
                TestClient(TestServer(application(tuple(urls), reader)))
            )
            path = "/v1/completions"
            hold = await client.post(path, data="hold")
            await asyncio.wait_for(started.wait(), timeout=5)
            assert hold.headers["X-Qualification-Endpoint"] == "0"
            for payload in ("cold", "warm"):
                async with client.post(
                    path, data=payload, headers={"X-Data-Parallel-Rank": "0"}
                ) as response:
                    assert response.headers["X-Qualification-Endpoint"] == "1"
                    assert await response.read() == b"data: first\n\ndata: [DONE]\n\n"
            finish.set()
            await hold.read()
            async with client.post(path, data="stale") as response:
                assert response.status == 503
            assert seen == ["hold", "cold", "warm"]

    asyncio.run(run())
