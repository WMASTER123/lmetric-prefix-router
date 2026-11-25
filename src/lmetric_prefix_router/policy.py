"""Select a whole replica from a complete snapshot of a homogeneous DP pool."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Replica:
    """Position in the input tuple is the DP rank; missing hits are unknown."""

    matched_tokens: int | None
    inflight: float
    pending_prefill_tokens: float
    gauge_age_seconds: float | None


def choose_rank(prompt_tokens: int, replicas: tuple[Replica, ...]) -> int | None:
    """Return the DP rank, or None to delegate before dispatch.

    Callers apply health, tenant and generation checks first. A partial view
    cannot certify the longest match. Fractional load is valid for delegated
    requests whose destination has not been observed.
    """
    if prompt_tokens < 0 or not replicas:
        raise ValueError("expected a nonnegative prompt length and at least one rank")
    for replica in replicas:
        if replica.matched_tokens is not None and not (
            0 <= replica.matched_tokens <= prompt_tokens
        ):
            raise ValueError("hit length exceeds the request bounds")
        if not all(
            isfinite(n) and n >= 0
            for n in (replica.inflight, replica.pending_prefill_tokens)
        ):
            raise ValueError("load must be finite and nonnegative")

    hits = [r.matched_tokens for r in replicas if r.matched_tokens is not None]
    if len(hits) != len(replicas) or max(hits) == 0:
        return None
    longest = max(hits)
    if all(h == longest for h in hits) and all(
        r.gauge_age_seconds is not None and 0 <= r.gauge_age_seconds <= 0.6
        for r in replicas
    ):
        return None

    def score(rank: int) -> tuple[float, float, int]:
        r = replicas[rank]
        work = prompt_tokens - longest + r.pending_prefill_tokens
        return work * (r.inflight + 1), r.inflight, rank

    return min((i for i, h in enumerate(hits) if h == longest), key=score)


class ExternalBalancer:
    """Route between eligible independent TP2 endpoints in a stable order.

    One instance belongs to one router event loop. Call selection and admission
    without awaiting between them. Both comparison arms use the same fallback.
    """

    def __init__(self) -> None:
        self._start = 0

    def choose(
        self, prompt_tokens: int, replicas: tuple[Replica, ...], *, use_prefix: bool
    ) -> int:
        if not replicas:
            raise ValueError("no eligible endpoints")
        pinned = choose_rank(prompt_tokens, replicas) if use_prefix else None
        if pinned is not None:
            return pinned
        for replica in replicas:
            if not isfinite(replica.inflight) or replica.inflight < 0:
                raise ValueError("invalid inflight estimate")
        count = len(replicas)
        order = ((self._start + offset) % count for offset in range(count))
        selected = min(order, key=lambda i: replicas[i].inflight)
        self._start = (selected + 1) % count
        return selected
