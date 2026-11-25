"""Serial, no-reset qualification for a fresh dedicated native023 deployment."""

import argparse
import json
import re
import time
from urllib.request import Request, urlopen


def fetch(url, payload=None, headers=None):
    req = Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urlopen(req, timeout=90) as res:
        return res.headers, res.read()


def wait_for(read, predicate):
    end = time.monotonic() + 25
    while True:
        value = read()
        if predicate(value):
            return value
        if time.monotonic() >= end:
            raise RuntimeError(f"functional check timed out: {value}")
        time.sleep(0.1)


def run(engine, router, model, dp, tp, block, rejoin=False):
    def state():
        return json.loads(fetch(router + "/status")[1])

    def metrics():
        text = fetch(engine + "/metrics")[1].decode()
        result = {}
        for kind in ("hits", "queries"):
            pattern = rf"^vllm:prefix_cache_{kind}_total" + (
                r'\{[^}]*\bengine="(\d+)"[^}]*\} (\S+)$'
            )
            result[kind] = {
                int(rank): int(float(v)) for rank, v in re.findall(pattern, text, re.M)
            }
            assert set(result[kind]) == set(range(dp))
        return result

    def request(
        label,
        endpoint,
        prompt,
        route_reason=None,
        pin=None,
        expected=0,
        chat_tokens=None,
    ):
        chat = chat_tokens is not None
        n = len(chat_tokens if chat else prompt)
        path = "/v1/chat/completions" if chat else "/v1/completions"
        key = "messages" if chat else "prompt"
        payload = {"model": model, key: prompt, "temperature": 0, "max_tokens": 2}
        before = metrics()
        headers, raw = fetch(
            endpoint + path,
            payload,
            {"X-Data-Parallel-Rank": str(pin)} if pin is not None else {},
        )
        usage = json.loads(raw)["usage"]
        assert usage["prompt_tokens"] == n, (label, n, usage["prompt_tokens"])
        after = wait_for(
            metrics,
            lambda s: sum(s["queries"].values()) > sum(before["queries"].values()),
        )
        q = [after["queries"][i] - before["queries"][i] for i in range(dp)]
        h = [after["hits"][i] - before["hits"][i] for i in range(dp)]
        owners = [rank for rank in range(dp) if q[rank] == n]
        assert len(owners) == 1, (label, n, q)
        actual = owners[0]
        assert q == [n if rank == actual else 0 for rank in range(dp)], (label, q)
        assert h == [expected if rank == actual else 0 for rank in range(dp)], (
            label,
            h,
        )
        receipt = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        assert receipt == expected or (expected == 0 and receipt is None)
        if route_reason:
            route, reason = route_reason
            assert (headers["X-Lmetric-Route"], headers["X-Lmetric-Reason"]) == (
                route,
                reason,
            )
            assert headers["X-Lmetric-Expected-Cached"] == str(expected)
            if route != "native":
                assert actual == int(route)
        elif pin is not None:
            assert actual == pin
        print(
            json.dumps(
                dict(
                    case=label,
                    actual_rank=actual,
                    reason=headers.get("X-Lmetric-Reason"),
                    predicted=headers.get("X-Lmetric-Expected-Cached"),
                    prompt_tokens=n,
                    usage_cached=receipt,
                    queries_delta=q,
                    hits_delta=h,
                )
            ),
            flush=True,
        )

    status = wait_for(state, lambda s: s["ready"])
    assert status["topology"] == dict(dp=dp, tp=tp, source="engine_process")
    assert status["chat_verified"]
    if not rejoin:
        assert sum(metrics()["queries"].values()) == 0
    print(
        json.dumps(
            dict(topology=status["topology"], generation=status["engine_generation"])
        ),
        flush=True,
    )

    def seq(rank):
        return state()["ranks"][rank]["sequence"]

    def advanced(rank, old):
        wait_for(
            state,
            lambda s: s["ready"] and s["ranks"][rank]["sequence"] not in (None, old),
        )

    length = 2601
    hit = (length - 1) // block * block
    offset = 7000 if rejoin else 0
    if rejoin:
        route = ("native", "equal_hit_native" if dp == 2 else "single_replica")
        request("recovered-old-prefix", router, [250] * length, route, expected=hit)
    reason = "single_replica" if dp == 1 else "all_miss"
    request("cold", router, [101 + offset] * length, ("native", reason))
    for rank in range(dp):
        ids = [250 + offset + rank] * length
        old = seq(rank)
        request(f"seed-{rank}", engine, ids, pin=rank)
        advanced(rank, old)
        route = (
            ("native", "single_replica") if dp == 1 else (str(rank), "longest_prefix")
        )
        request(f"owner-{rank}", router, ids, route, pin=(1 - rank), expected=hit)
        if dp == 2:
            peer = 1 - rank
            old = seq(peer)
            request(f"seed-peer-{peer}", engine, ids, pin=peer)
            advanced(peer, old)
            request(
                f"both-hot-{rank}",
                router,
                ids,
                ("native", "equal_hit_native"),
                expected=hit,
            )
    messages = [
        {
            "role": "user",
            "content": ("New topic. " if rejoin else "")
            + "The next word is green. " * 430
            + "Write a short reply.",
        }
    ]
    tokens = json.loads(
        fetch(engine + "/tokenize", {"model": model, "messages": messages})[1]
    )["tokens"]
    old = seq(0)
    request("chat-seed", engine, messages, pin=0, chat_tokens=tokens)
    advanced(0, old)
    request(
        "chat-hot",
        router,
        messages,
        ("native", "single_replica") if dp == 1 else ("0", "longest_prefix"),
        expected=(len(tokens) - 1) // block * block,
        chat_tokens=tokens,
    )
    print(json.dumps({"final": state()}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", required=True)
    p.add_argument("--router", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dp", type=int, required=True, choices=(1, 2))
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--block", type=int, required=True)
    p.add_argument(
        "--rejoin",
        action="store_true",
        help="Repeat on the first run's warm dedicated engine",
    )
    run(**vars(p.parse_args()))
