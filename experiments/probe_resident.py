"""Probe cache receipts without trusting a routing prediction or changing the engine."""

import argparse
import json
import uuid
from time import monotonic
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--peer-url")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    run = uuid.uuid4().hex

    def complete(endpoint, case, prompt, salt):
        data = json.dumps(
            dict(
                model=args.model,
                prompt=prompt,
                cache_salt=salt,
                temperature=0,
                max_tokens=8,
                return_token_ids=True,
                stream=False,
            )
        ).encode()
        req = Request(
            endpoint.rstrip("/") + "/v1/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        started = monotonic()
        with urlopen(req, timeout=120) as response:
            result = json.load(response)
        choice, usage = result["choices"][0], result["usage"]
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        print(
            json.dumps(
                dict(
                    run=run,
                    case=case,
                    prompt_tokens=usage["prompt_tokens"],
                    output_tokens=usage["completion_tokens"],
                    cached=cached,
                    elapsed_ms=round(1000 * (monotonic() - started), 2),
                )
            ),
            flush=True,
        )
        return choice

    seed = "Explain this routing test. " + ("A replica spans two devices. " * 100)
    first = complete(args.url, "cold", seed, run)
    if not first.get("prompt_token_ids") or not first.get("token_ids"):
        raise RuntimeError("No exact prompt/output token IDs for continuation probe")
    history = first["prompt_token_ids"] + first["token_ids"]
    continuation = history + history[:12]
    if args.peer_url:
        complete(args.peer_url, "untouched-peer", continuation, run)
    complete(args.url, "same-replica-continuation", continuation, run)
    complete(args.url, "different-salt", continuation, uuid.uuid4().hex)


if __name__ == "__main__":
    main()
