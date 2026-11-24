"""A single local subscriber per DP rank, with fail-closed cache evidence."""

import asyncio
import logging

import msgspec
import zmq
import zmq.asyncio

from .cache023 import CacheView

log = logging.getLogger(__name__)


async def consume(view: CacheView, subscriber, monitor):
    poller = zmq.asyncio.Poller()
    poller.register(subscriber, zmq.POLLIN)
    poller.register(monitor, zmq.POLLIN)
    try:
        while True:
            ready = dict(await poller.poll())
            if monitor in ready:
                await monitor.recv_multipart()
                log.warning(
                    "rank %s publisher disconnected; cache view disabled", view.rank
                )
                return
            if subscriber in ready:
                _, number, body = await subscriber.recv_multipart()
                if len(number) != 8:
                    raise ValueError("invalid KV sequence frame")
                view.apply(int.from_bytes(number, "big"), msgspec.msgpack.decode(body))
    except (
        ValueError,
        TypeError,
        IndexError,
        KeyError,
        zmq.ZMQError,
        msgspec.DecodeError,
    ):
        log.exception("rank %s cache feed disabled", view.rank)
    finally:
        view.invalidate()


class EventStreams:
    def __init__(self, views, first_port):
        self.views, self.first_port = views, first_port

    async def __aenter__(self):
        self.ctx = zmq.asyncio.Context()
        self.tasks = []
        self.sockets = []
        try:
            for view in self.views:
                sub = self.ctx.socket(zmq.SUB)
                self.sockets.append(sub)
                sub.setsockopt(zmq.SUBSCRIBE, b"")
                sub.setsockopt(zmq.RCVHWM, 100000)
                monitor = sub.get_monitor_socket(events=zmq.EVENT_DISCONNECTED)
                self.sockets.append(monitor)
                sub.bind(f"tcp://127.0.0.1:{self.first_port + view.rank}")
                self.tasks.append(asyncio.create_task(consume(view, sub, monitor)))
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *_):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for sock in reversed(self.sockets):
            sock.close(linger=0)
        self.ctx.destroy(linger=0)
