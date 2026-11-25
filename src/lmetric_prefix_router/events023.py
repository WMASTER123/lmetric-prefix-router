"""Rank-local KV subscriptions and bounded recovery anchored to a live batch."""

import asyncio
import logging

import msgspec
import zmq
import zmq.asyncio

from .cache023 import CacheView

log = logging.getLogger(__name__)
MAX_BATCHES = 1024
MAX_BYTES = 8 * 1024 * 1024
FEED_ERRORS = (ValueError, TypeError, IndexError, KeyError, msgspec.DecodeError)


async def replay_to_live(ctx, endpoint, rank, target, anchor):
    """Rebuild from a retained clear through the exact live (seq, payload)."""
    first = max(0, target + 1 - MAX_BATCHES)
    recovered = CacheView(rank)
    sock = ctx.socket(zmq.DEALER)
    try:
        sock.setsockopt(zmq.MAXMSGSIZE, MAX_BYTES)
        sock.setsockopt(zmq.RCVHWM, 8)
        sock.connect(endpoint)
        await sock.send_multipart([b"", first.to_bytes(8, "big")])
        total = 0
        for _ in range(MAX_BATCHES):
            delim, seq, body = await sock.recv_multipart()
            if delim or len(seq) != 8 or seq == b"\xff" * 8:
                break
            number = int.from_bytes(seq, "big")
            total += len(body)
            if number < first or number > target or total > MAX_BYTES:
                break
            if recovered.sequence is not None and number != recovered.sequence + 1:
                break
            recovered.apply(number, msgspec.msgpack.decode(body))
            if number == target:
                if body == anchor and recovered.trusted:
                    return recovered
                break
    finally:
        sock.close(linger=0)
    return None


async def consume(view: CacheView, subscriber, monitor, replay_endpoint=None):
    poller = zmq.asyncio.Poller()
    poller.register(subscriber, zmq.POLLIN)
    poller.register(monitor, zmq.POLLIN)
    retry_after = 0.0
    loop = asyncio.get_running_loop()
    try:
        while True:
            ready = dict(await poller.poll())
            if monitor in ready:
                await monitor.recv_multipart()
                view.invalidate("disconnected")
                return
            if subscriber not in ready:
                continue
            try:
                topic, number, body = await subscriber.recv_multipart()
                if topic or len(number) != 8:
                    raise ValueError("invalid KV event framing")
                sequence = int.from_bytes(number, "big")
                view.apply(sequence, msgspec.msgpack.decode(body))
                if view.trusted or not replay_endpoint or loop.time() < retry_after:
                    continue
                rebuilt = await asyncio.wait_for(
                    replay_to_live(
                        subscriber.context, replay_endpoint, view.rank, sequence, body
                    ),
                    timeout=1.0,
                )
                if rebuilt is not None and not await monitor.poll(timeout=0):
                    view.groups = rebuilt.groups
                    view.excluded = rebuilt.excluded
                    view.sync_state = "replayed_to_live"
                    view.trusted = True
                retry_after = loop.time() + 2
            except asyncio.TimeoutError:
                retry_after = loop.time() + 2
                view.invalidate("replay_timeout")
            except FEED_ERRORS:
                view.invalidate("invalid_event")
                log.warning("rank %s invalid event or replay", view.rank)
    finally:
        if view.trusted:
            view.invalidate("disconnected")


class EventStreams:
    def __init__(self, views, first_port, replay_port=None):
        self.views, self.first_port, self.replay_port = views, first_port, replay_port

    def open_feed(self, rank):
        sub = self.ctx.socket(zmq.SUB)
        try:
            sub.setsockopt(zmq.SUBSCRIBE, b"")
            sub.setsockopt(zmq.RCVHWM, 100000)
            sub.bind(f"tcp://127.0.0.1:{self.first_port + rank}")
            monitor = sub.get_monitor_socket(events=zmq.EVENT_DISCONNECTED)
        except BaseException:
            sub.close(linger=0)
            raise
        return sub, monitor

    async def supervise(self, view, sub, monitor):
        endpoint = (
            f"tcp://127.0.0.1:{self.replay_port + view.rank}"
            if self.replay_port is not None
            else None
        )
        while True:
            try:
                await consume(view, sub, monitor, endpoint)
            except zmq.ZMQError:
                log.exception("rank %s subscription failed", view.rank)
            finally:
                view.invalidate("disconnected")
                view.sequence = None
                monitor.close(linger=0)
                sub.close(linger=0)
            while True:
                await asyncio.sleep(0.25)
                try:
                    sub, monitor = self.open_feed(view.rank)
                    break
                except zmq.ZMQError:
                    log.warning("rank %s cannot reopen event port", view.rank)

    async def __aenter__(self):
        self.ctx = zmq.asyncio.Context()
        self.tasks = []
        try:
            for view in self.views:
                sockets = self.open_feed(view.rank)
                self.tasks.append(asyncio.create_task(self.supervise(view, *sockets)))
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *_):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.ctx.destroy(linger=0)
