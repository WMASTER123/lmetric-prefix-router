import asyncio
import time
from contextlib import suppress
from types import SimpleNamespace

import pytest

from lmetric_prefix_router import managed023
from lmetric_prefix_router.cache023 import BLOCK, GROUPS, CacheView, block_key
from lmetric_prefix_router.managed023 import ManagedReplay, launch_options


def test_managed_replay_cold_start_timeout_recovery_silent_remove_and_restart(
    monkeypatch,
):
    async def check():
        view = CacheView(0)
        engine = SimpleNamespace(replay_port=18400, current=lambda _: True)
        reader = ManagedReplay([view], engine)
        reader.ctx = None
        calls = asyncio.Queue()
        replies = asyncio.Queue()

        async def read(ctx, endpoint, start):
            await calls.put(start)
            rows = await replies.get()
            if isinstance(rows, Exception):
                raise rows
            return rows

        monkeypatch.setattr(managed023, "read_delta", read)
        task = asyncio.create_task(reader.follow(view))

        async def cursor(n):
            assert await asyncio.wait_for(calls.get(), 2) == n

        try:
            await cursor(0)
            key = block_key(5)
            stored = [
                [
                    "BlockStored",
                    [key],
                    None,
                    [],
                    BLOCK,
                    None,
                    "GPU",
                    None,
                    [],
                    g,
                    s,
                    None,
                ]
                for g, s in enumerate(GROUPS)
            ]
            await replies.put([[0, stored, 0]])
            await cursor(1)
            assert view.match([key], 2600) == BLOCK
            await replies.put(asyncio.TimeoutError())
            await cursor(1)
            assert view.match([key], 2600) is None
            await replies.put([])
            await cursor(1)
            assert view.match([key], 2600) == BLOCK
            await replies.put([[1, [["BlockRemoved", [key], "GPU", 0]], 0]])
            await cursor(2)
            assert view.match([key], 2600) == 0
            view.valid_until = time.monotonic() - 1
            assert view.match([key], 2600) is None
            engine.current = lambda _: False
            await replies.put([])
            await asyncio.wait_for(task, 2)
            assert not view.ready and view.sync_state == "generation_changed"
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    asyncio.run(check())


def test_launch_options_reject_duplicate_or_unqualified_switch():
    argv = ["python", "vllm", "serve", "/model", "--tensor-parallel-size", "2"]
    assert launch_options(argv) == ("/model", {"--tensor-parallel-size": "2"})
    for extra in (["--tensor-parallel-size", "1"], ["--chat-template", "custom.jinja"]):
        with pytest.raises(ValueError):
            launch_options(argv + extra)


@pytest.mark.parametrize("sequence", [0, 1])
def test_managed_reader_rejects_missing_origin(sequence):
    from contextlib import closing

    import msgspec
    import zmq
    import zmq.asyncio

    async def check():
        ctx = zmq.asyncio.Context()
        server = ctx.socket(zmq.ROUTER)
        port = server.bind_to_random_port("tcp://127.0.0.1")
        task = asyncio.create_task(
            managed023.read_delta(ctx, f"tcp://127.0.0.1:{port}", 0)
        )
        try:
            peer, delim, start = await asyncio.wait_for(server.recv_multipart(), 2)
            assert delim == b"" and start == bytes(8)
            await server.send_multipart(
                [
                    peer,
                    b"",
                    sequence.to_bytes(8, "big"),
                    msgspec.msgpack.encode([0, [], 0]),
                ]
            )
            await server.send_multipart([peer, b"", b"\xff" * 8, b""])
            if sequence:
                with pytest.raises(ValueError, match="incomplete"):
                    await asyncio.wait_for(task, 2)
            else:
                assert await asyncio.wait_for(task, 2) == [[0, [], 0]]
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            server.close(linger=0)
            ctx.destroy(linger=0)

    with closing(asyncio.SelectorEventLoop()) as loop:
        loop.run_until_complete(check())
