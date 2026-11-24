# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-hust-prefix-router at bcc9691.
import json


class UsageObserver:
    """Extract usage from JSON or SSE while bounding response buffering."""

    def __init__(
        self,
        *,
        streaming: bool,
        observe_content: bool = False,
        max_bytes: int = 256 * 1024,
    ) -> None:
        self.streaming, self.max_bytes = streaming, max_bytes
        self.observe_content = observe_content
        self.has_content = False
        self.cached_tokens: int | None = None
        self.prompt_tokens: int | None = None
        self.backend_response_id: str | None = None
        self._buffer = b""
        self._disabled = False

    def feed(self, chunk: bytes) -> None:
        if self._disabled:
            return
        if len(self._buffer) + len(chunk) > self.max_bytes:
            self._buffer = b""
            self._disabled = True
            return
        self._buffer += chunk
        if self.streaming:
            self._buffer = self._buffer.replace(b"\r\n", b"\n")
            while b"\n\n" in self._buffer:
                frame, self._buffer = self._buffer.split(b"\n\n", 1)
                data = b"\n".join(
                    line[5:].lstrip()
                    for line in frame.split(b"\n")
                    if line.startswith(b"data:")
                )
                self._observe(data)

    def finish(self) -> None:
        if not self.streaming and not self._disabled:
            self._observe(self._buffer)
        self._buffer = b""

    def _observe(self, data: bytes) -> None:
        try:
            payload = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(payload, dict):
            return
        if self.observe_content and not self.has_content:
            for choice in payload.get("choices") or ():
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                if isinstance(delta, dict) and (
                    choice.get("text")
                    or any(
                        delta.get(k)
                        for k in ("content", "reasoning_content", "reasoning")
                    )
                    or any(
                        call["function"].get("arguments")
                        for call in delta.get("tool_calls") or ()
                        if isinstance(call, dict)
                        and isinstance(call.get("function"), dict)
                    )
                ):
                    self.has_content = True
                    break
        # A stream can be interrupted before its final usage event. Response
        # identity and actual cache usage are independent observations.
        response_id = payload.get("id")
        if isinstance(response_id, str) and 0 < len(response_id) <= 256:
            if self.backend_response_id not in (None, response_id):
                # Conflicting response identities cannot support one receipt.
                self.backend_response_id = None
                self.prompt_tokens = self.cached_tokens = None
                self._buffer = b""
                self._disabled = True
                return
            self.backend_response_id = response_id
        if not isinstance(payload.get("usage"), dict):
            return
        usage = payload["usage"]
        prompt = usage.get("prompt_tokens")
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        if type(prompt) is int and prompt >= 0:
            self.prompt_tokens = prompt
        if type(cached) is int and 0 <= cached <= (self.prompt_tokens or 0):
            self.cached_tokens = cached
