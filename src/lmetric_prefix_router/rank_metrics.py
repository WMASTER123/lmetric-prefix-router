# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-hust-prefix-router at b63ffe2.
"""Out-of-band DP engine gauges for the shared vLLM metrics endpoint."""

import asyncio
import math
import re
import time

from aiohttp import ClientError, ClientSession, ClientTimeout

_GAUGE = re.compile(r"^vllm:num_requests_(running|waiting)\{([^}]*)\}\s+([^\s]+)$")
_ENGINE = re.compile(r'(?:^|,)engine="(\d+)"(?:,|$)')


def parse_gauges(body, stamp):
    values = {}
    for line in body.splitlines():
        row = _GAUGE.fullmatch(line)
        if row and (label := _ENGINE.search(row[2])):
            rank = int(label[1])
            value = float(row[3])
            if not math.isfinite(value) or value < 0 or not value.is_integer():
                raise ValueError("invalid vLLM DP gauge")
            item = values.setdefault(rank, {})
            if row[1] in item:
                raise ValueError("ambiguous vLLM DP gauge")
            item[row[1]] = int(value)
    if any(len(pair) != 2 for pair in values.values()):
        raise ValueError("incomplete vLLM DP gauge pair")
    return {i: (stamp, v["running"], v["waiting"]) for i, v in values.items()}


async def poll_gauges(policy, backend_url):
    async with ClientSession(timeout=ClientTimeout(total=1)) as session:
        while True:
            try:
                stamp = time.monotonic()
                async with session.get(backend_url.rstrip("/") + "/metrics") as res:
                    res.raise_for_status()
                    policy.gauges = parse_gauges(await res.text(), stamp)
                    policy.observed_ranks = set(policy.gauges)
            except (ClientError, asyncio.TimeoutError, ValueError):
                policy.gauges = {}
                # Keep a known topology mismatch latched across scrape failures.
                if policy.observed_ranks == policy.expected_ranks:
                    policy.observed_ranks = None
            await asyncio.sleep(0.25)
