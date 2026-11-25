"""Functional probe for an isolated native023 engine and its local MOD.

This explicitly resets the supplied development engine's prefix cache.
"""

import argparse
import json
import time
from urllib.request import Request, urlopen


def call(url, data=None, rank=None):
    headers = {"Content-Type": "application/json"}
    if rank is not None:
        headers["X-Data-Parallel-Rank"] = str(rank)
    req = Request(url, data=data, headers=headers)
    with urlopen(req, timeout=90) as response:
        return response.headers, response.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--router", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reset-isolated-engine", action="store_true", required=True)
    args = ap.parse_args()
    engine, router = args.engine.rstrip("/"), args.router.rstrip("/")

    def status():
        return json.loads(call(router + "/status")[1])

    def generate(endpoint, prompt, rank=None, **extra):
        payload = dict(
            model=args.model, prompt=prompt, max_tokens=1, temperature=0, **extra
        )
        headers, raw = call(
            endpoint + "/v1/completions", json.dumps(payload).encode(), rank
        )
        usage = json.loads(raw)["usage"]
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        return (
            headers.get("X-Lmetric-Route"),
            headers.get("X-Lmetric-Expected-Cached"),
            cached,
        )

    def wait_for(condition):
        end = time.monotonic() + 20
        while not condition():
            if time.monotonic() > end:
                raise RuntimeError(
                    f"event or gauge synchronization timed out: {status()}"
                )
            time.sleep(0.05)

    def check(label, result, rank, expected):
        route, prediction, actual = result
        print(
            json.dumps(
                dict(case=label, route=route, predicted=prediction, actual=actual)
            ),
            flush=True,
        )
        assert route == rank and prediction == str(expected)
        if expected not in (0, "unknown"):
            assert actual == expected
        if expected == 0:
            assert actual in (0, None)

    state = status()
    if state["policy"] != "combined":
        raise ValueError("probe requires combined")
    for rank in (0, 1):
        generate(engine, [60 + rank] * 2200, rank)
    call(engine + "/reset_prefix_cache", b"")
    for rank in (0, 1):
        generate(engine, [76 + rank] * 20, rank)
    wait_for(lambda: all(v["trusted"] for v in status()["ranks"]))
    check("cold", generate(router, [52] * 2200, 0), "native", 0)
    for owner in (0, 1):
        prompt = [301 + owner] * 2600
        before = status()["ranks"][owner]["sequence"]
        generate(engine, prompt, owner)
        wait_for(
            lambda owner=owner, before=before: (
                status()["ranks"][owner]["sequence"] != before
            )
        )
        check(
            f"single-owner-{owner}",
            generate(router, prompt, 1 - owner),
            str(owner),
            2048,
        )
        peer = 1 - owner
        before = status()["ranks"][peer]["sequence"]
        generate(engine, prompt, peer)
        wait_for(
            lambda peer=peer, before=before: (
                status()["ranks"][peer]["sequence"] != before
                and all(status()["gauges_fresh"])
            )
        )
        check(f"both-hot-{owner}", generate(router, prompt), "native", 2048)
    check(
        "salt-native",
        generate(router, [301] * 2600, cache_salt="separate-probe-identity"),
        "native",
        "unknown",
    )
    print(json.dumps(status()))


if __name__ == "__main__":
    main()
