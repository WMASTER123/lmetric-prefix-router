import asyncio
import time
from contextlib import closing

import msgspec
import pytest
import zmq
import zmq.asyncio

from lmetric_prefix_router.cache023 import BLOCK, GROUPS, CacheView, block_key
from lmetric_prefix_router.events023 import EventStreams, replay_to_live
from lmetric_prefix_router.native023 import NativePool


def test_pool_distinguishes_topology_trust_and_zero():
    pool = NativePool(rank_count=1)
    fp = (2300, [block_key(11)])
    for membership, expected in [
        (None, "topology_unverified"),
        ({0, 1}, "topology_mismatch"),
        ({0}, "cache_view_unavailable"),
    ]:
        pool.observed_ranks = membership
        row, predicted = pool.reserve(fp)
        assert (row.rank, row.reason, predicted) == (None, expected, None)
    pool.views[0].apply(0, [0, [["AllBlocksCleared"]], 0])
    row, predicted = pool.reserve(fp)
    assert (row.rank, row.reason, predicted) == (None, "single_replica", 0)
    assert pool.reserve(None)[0].reason == "unsupported_input"


def test_dual_replica_reasons_and_both_owner_directions():
    pool = NativePool()
    pool.observed_ranks = {0, 1}
    key = block_key(7)
    fp = (2300, [key])

    def route():
        row, predicted = pool.reserve(fp)
        pool.active.remove(row)
        return row.rank, row.reason, predicted

    for v in pool.views:
        v.apply(0, [0, [["AllBlocksCleared"]], v.rank])
    assert route() == (None, "all_miss", 0)
    stores = [
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
            group,
            spec,
            None,
        ]
        for group, spec in enumerate(GROUPS)
    ]
    for rank in (1, 0):
        pool.views[rank].apply(1, [0, stores, rank])
        if rank == 1:
            assert route() == (1, "longest_prefix", BLOCK)
    pool.gauges = {rank: (time.monotonic(), 0, 0) for rank in range(2)}
    assert route() == (None, "equal_hit_native", BLOCK)
    pool.gauges = {}
    assert route() == (0, "equal_hit_local_load", BLOCK)
    pool.views[1].apply(2, [0, [["AllBlocksCleared"]], 1])
    assert route() == (0, "longest_prefix", BLOCK)


@pytest.mark.parametrize("fault", ["none", "gap", "old_publisher", "no_clear", "short"])
def test_replay_anchored_to_live(fault):
    async def run():
        ctx = zmq.asyncio.Context()
        server = ctx.socket(zmq.ROUTER)
        port = server.bind_to_random_port("tcp://127.0.0.1")
        endpoint = f"tcp://127.0.0.1:{port}"
        anchor = msgspec.msgpack.encode([2.23, [], 0])

        async def respond():
            identity, delim, start = await server.recv_multipart()
            assert not delim and start == bytes(8)
            batches = [
                (
                    0,
                    msgspec.msgpack.encode(
                        [0, [] if fault == "no_clear" else [["AllBlocksCleared"]], 0]
                    ),
                )
            ]
            if fault != "gap":
                batches.append((1, msgspec.msgpack.encode([1, [], 0])))
            if fault != "short":
                batches.append(
                    (
                        2,
                        anchor
                        if fault != "old_publisher"
                        else msgspec.msgpack.encode([99, [], 0]),
                    )
                )
            for seq, payload in batches:
                await server.send_multipart(
                    [identity, b"", seq.to_bytes(8, "big"), payload]
                )
            await server.send_multipart([identity, b"", b"\xff" * 8, b""])

        try:
            _, result = await asyncio.wait_for(
                asyncio.gather(respond(), replay_to_live(ctx, endpoint, 0, 2, anchor)),
                3,
            )
            assert (result is not None and result.trusted) == (fault == "none")
        finally:
            server.close(0)
            ctx.destroy(linger=0)

    with closing(asyncio.SelectorEventLoop()) as loop:
        loop.run_until_complete(run())


def test_subscriber_recovers_on_replacement_publisher():
    async def run():
        ctx = zmq.asyncio.Context()
        from socket import socket

        with socket() as unused:
            unused.bind(("127.0.0.1", 0))
            port = unused.getsockname()[1]
        view = CacheView(0)

        async def wait_state(expected):
            for _ in range(300):
                if view.sync_state == expected:
                    return
                await asyncio.sleep(0.01)
            pytest.fail(f"expected {expected}, got {view.sync_state}")

        try:
            async with EventStreams([view], port):
                for _ in range(2):
                    pub = ctx.socket(zmq.XPUB)
                    try:
                        pub.connect(f"tcp://127.0.0.1:{port}")
                        assert await asyncio.wait_for(pub.recv(), 5) == b"\x01"
                        await pub.send_multipart(
                            [
                                b"",
                                bytes(8),
                                msgspec.msgpack.encode([0, [["AllBlocksCleared"]], 0]),
                            ]
                        )
                        await wait_state("observed_contiguous")
                    finally:
                        pub.close(0)
                    await wait_state("disconnected")
                    assert not view.trusted
        finally:
            ctx.destroy(linger=0)

    with closing(asyncio.SelectorEventLoop()) as loop:
        loop.run_until_complete(run())
