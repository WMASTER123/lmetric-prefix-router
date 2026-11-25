"""Process-bound replay for local vLLM schedulers with an initially empty block pool."""

import asyncio
import hashlib
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlsplit

import msgspec
import zmq
import zmq.asyncio

from .cache023 import BLOCK, GROUPS, CacheView
from .events023 import FEED_ERRORS, MAX_BATCHES, MAX_BYTES


def launch_options(argv):
    """Accept the qualified local serving profile; conflicting switches fail closed."""
    pos = argv.index("serve")
    if Path(argv[pos - 1]).name != "vllm":
        raise ValueError("expected a local vllm serve process")
    flags = {
        "--language-model-only",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--enable-prompt-tokens-details",
        "--enforce-eager",
    }
    values = {
        "--host",
        "--port",
        "--served-model-name",
        "--tensor-parallel-size",
        "--data-parallel-size",
        "--dtype",
        "--max-model-len",
        "--max-num-seqs",
        "--max-num-batched-tokens",
        "--gpu-memory-utilization",
        "--prefix-caching-hash-algo",
        "--kv-events-config",
        "--block-size",
        "--api-server-count",
    }
    args = iter(argv[pos + 2 :])
    options = {}
    for key in args:
        if key in options or key not in flags | values:
            raise ValueError(f"unqualified or duplicate engine option {key}")
        options[key] = True if key in flags else next(args, None)
    return argv[pos + 1], options


class LocalEngine:
    """Hold the API and each replay listener's process/FD identity until shutdown."""

    def __init__(self, pid, backend, model_path, served_name, event_port, replay_port):
        import psutil

        api = psutil.Process(pid)
        path, opts = launch_options(api.cmdline())
        config = json.loads(opts["--kv-events-config"])
        expected = {
            "--host": "127.0.0.1",
            "--served-model-name": served_name,
            "--prefix-caching-hash-algo": "sha256",
            "--enable-prefix-caching": True,
            "--max-model-len": "4096",
        }
        address = urlsplit(backend)
        if (
            Path(path).resolve() != Path(model_path).resolve()
            or address.hostname != "127.0.0.1"
            or address.port != int(opts["--port"])
            or any(opts.get(k) != v for k, v in expected.items())
            or config.get("enable_kv_cache_events") is not True
            or config.get("publisher") != "zmq"
            or config.get("topic", "")
            or config.get("endpoint") != f"tcp://127.0.0.1:{event_port}"
            or config.get("replay_endpoint") != f"tcp://127.0.0.1:{replay_port}"
        ):
            raise ValueError("engine and router cache profiles differ")
        for key in ("PYTHONHASHSEED", "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES"):
            if api.environ().get(key) != os.environ.get(key):
                raise ValueError(f"engine and router disagree on {key}")
        self.dp = int(opts.get("--data-parallel-size", 1))
        self.tp = int(opts.get("--tensor-parallel-size", 1))
        self.block_size = int(opts.get("--block-size", 2048))
        model = json.loads((Path(path) / "config.json").read_text())
        if (
            model.get("architectures") == ["Qwen3_5MoeForConditionalGeneration"]
            and self.block_size == BLOCK
        ):
            self.specs = GROUPS
        elif (
            model.get("architectures") == ["Qwen2ForCausalLM"]
            and model.get("use_sliding_window") is False
            and opts.get("--block-size") == "128"
        ):
            self.specs = ("full_attention",)
        else:
            raise ValueError("unsupported cache layout in managed profile")
        self.replay_port = replay_port
        ports = [address.port, *range(replay_port, replay_port + self.dp)]
        owners = {}
        processes = [api, *api.children(recursive=True)]
        for proc in processes:
            try:
                for conn in proc.net_connections(kind="tcp4"):
                    if (
                        conn.status == psutil.CONN_LISTEN
                        and conn.laddr.ip == "127.0.0.1"
                        and conn.laddr.port in ports
                    ):
                        owners.setdefault(conn.laddr.port, []).append((proc, conn.fd))
            except psutil.NoSuchProcess:
                continue
        if (
            set(owners) != set(ports)
            or any(len(items) != 1 for items in owners.values())
            or owners[address.port][0][0] != api
            or len({owners[p][0][0].pid for p in ports}) != len(ports)
            or any(
                not owners[p][0][0].name().startswith("VLLM::EngineCor")
                for p in ports[1:]
            )
        ):
            raise ValueError("expected one API and a distinct replay owner per DP rank")
        self.sockets = []
        self.generations = []
        for port in ports:
            proc, fd = owners[port][0]
            link = Path(f"/proc/{proc.pid}/fd/{fd}")
            self.sockets.append((proc, link, os.readlink(link)))
            self.generations.append((proc.pid, proc.create_time()))
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        identity = [
            boot,
            os.readlink(f"/proc/{pid}/ns/pid"),
            path,
            opts,
            self.generations,
            os.environ["PYTHONHASHSEED"],
        ]
        self.digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()

    def current(self, rank):
        try:
            return all(
                proc.is_running() and os.readlink(link) == inode
                for proc, link, inode in (self.sockets[0], self.sockets[rank + 1])
            )
        except OSError:
            return False


async def read_delta(ctx, endpoint, start, *, snapshot=None):
    """Validate through END_SEQ. A private snapshot folds history as it arrives."""
    sock = ctx.socket(zmq.DEALER)
    try:
        sock.setsockopt(zmq.MAXMSGSIZE, MAX_BYTES)
        sock.setsockopt(zmq.RCVHWM, 8)
        sock.connect(endpoint)
        await sock.send_multipart([b"", start.to_bytes(8, "big")])
        rows = []
        size = 0
        sequence = start
        while True:
            delim, seq, body = await sock.recv_multipart()
            if delim or len(seq) != 8:
                raise ValueError("invalid replay envelope")
            if seq == b"\xff" * 8 and not body:
                return rows
            if int.from_bytes(seq, "big") != sequence or not body:
                raise ValueError("incomplete replay history")
            if snapshot is None:
                size += len(body)
                if len(rows) >= MAX_BATCHES or size > MAX_BYTES:
                    raise ValueError("oversized incremental replay")
                rows.append(msgspec.msgpack.decode(body))
            else:
                snapshot.apply(sequence, msgspec.msgpack.decode(body))
                if (sequence + 1) % 32 == 0:
                    await asyncio.sleep(0)
            sequence += 1
    finally:
        sock.close(linger=0)


class ManagedReplay:
    def __init__(self, views, engine):
        self.views, self.engine = views, engine

    async def follow(self, view):
        start = 0
        endpoint = f"tcp://127.0.0.1:{self.engine.replay_port + view.rank}"
        while True:
            try:
                if not self.engine.current(view.rank):
                    view.invalidate("generation_changed")
                    return
                began = time.monotonic()
                snapshot = None
                if start == 0:
                    snapshot = CacheView(view.rank, view.block_size, view.specs)
                    snapshot.trusted = True  # Bound scheduler starts with empty pools.
                rows = await asyncio.wait_for(
                    read_delta(self.ctx, endpoint, start, snapshot=snapshot),
                    30 if snapshot is not None else 1,
                )
                if not self.engine.current(view.rank):
                    view.invalidate("generation_changed")
                    return
                if snapshot is not None:
                    view.groups, view.excluded = snapshot.groups, snapshot.excluded
                    view.sequence, view.trusted = snapshot.sequence, snapshot.trusted
                    start = 0 if snapshot.sequence is None else snapshot.sequence + 1
                else:
                    for i, batch in enumerate(rows, start):
                        view.apply(i, batch)
                    start += len(rows)
                if time.monotonic() - began > 1:
                    view.valid_until = 0
                    view.sync_state = "awaiting_fresh_tail"
                else:
                    view.sync_state = "managed_contiguous"
                    view.valid_until = time.monotonic() + 1.25
            except (asyncio.TimeoutError, zmq.ZMQError):
                # No batch is applied until its entire reply is checked. Preserve the
                # committed cursor while withholding predictions until a fresh poll.
                view.valid_until = 0
                view.sync_state = "replay_unavailable"
            except FEED_ERRORS:
                view.invalidate("replay_incomplete")
                start = 0
            await asyncio.sleep(0.25)

    async def __aenter__(self):
        import select

        self.ctx = zmq.asyncio.Context()
        self.fds = ExitStack()
        self.tasks = []
        try:
            for view in self.views:
                monitor = select.poll()
                for i in (0, view.rank + 1):
                    proc = self.engine.sockets[i][0]
                    fd = os.pidfd_open(proc.pid)
                    self.fds.callback(os.close, fd)
                    monitor.register(fd, select.POLLIN)
                view.generation_alive = lambda monitor=monitor: not monitor.poll(0)
                view.valid_until = 0
                self.tasks.append(asyncio.create_task(self.follow(view)))
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *_):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for view in self.views:
            view.invalidate()
            view.generation_alive = None
        self.fds.close()
        self.ctx.destroy(linger=0)
