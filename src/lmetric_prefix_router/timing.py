# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM-HUST/vllm-hust-prefix-router at bcc9691.
from __future__ import annotations

import time
from collections.abc import Callable


class RequestTiming:
    """Relative timestamps on one process clock, not engine/token timestamps."""

    def __init__(self, *, clock: Callable[[], int] = time.perf_counter_ns) -> None:
        self._clock = clock
        self._start = clock()
        self.events: dict[str, int] = {"entry": 0}
        self.request_id: str | None = None
        self.node_id: str | None = None
        self.terminal = "router_error"
        self.http_status: int | None = None

    def mark(self, name: str) -> None:
        if name not in self.events:
            self.events[name] = self._clock() - self._start

    def snapshot(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "node_id": self.node_id,
            "terminal": self.terminal,
            "http_status": self.http_status,
            "elapsed_ns": dict(self.events),
            "clock": "process_relative_perf_counter_ns",
            "first_body_is_first_token": False,
        }
