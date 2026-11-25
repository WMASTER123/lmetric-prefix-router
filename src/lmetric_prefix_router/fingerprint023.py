"""Read one pinned vLLM installation's tokenizer and native block hash functions."""

import os
from importlib.metadata import version

import aiohttp

from .cache023 import BLOCK, block_key


class HostFingerprint:
    def __init__(self, model_path, served_model, block_size=BLOCK):
        if (version("vllm"), version("vllm-ascend")) != ("0.23.0+empty", "0.23.0"):
            raise RuntimeError(
                "native023 requires the qualified preinstalled host versions"
            )
        if not os.environ.get("PYTHONHASHSEED", "").isdigit():
            raise RuntimeError("set the same fixed PYTHONHASHSEED on engine and router")
        from vllm.tokenizers.registry import get_tokenizer
        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core.kv_cache_utils import (
            hash_block_tokens,
            init_none_hash,
            maybe_convert_block_hash,
        )

        self.model = served_model
        self.block_size = block_size
        self.chat_verified = False
        self.tokenizer = get_tokenizer(model_path)
        self.function = get_hash_fn_by_name("sha256")
        init_none_hash(self.function)
        self.hash_block = hash_block_tokens
        self.external = maybe_convert_block_hash

    def chat_ids(self, messages):
        from jinja2 import TemplateError

        if (
            not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(m, dict)
                or set(m) != {"role", "content"}
                or m["role"] not in ("system", "user", "assistant")
                or not isinstance(m["content"], str)
                for m in messages
            )
        ):
            return None
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return self.tokenizer.encode(rendered, add_special_tokens=False)
        except (ValueError, TypeError, TemplateError):
            return None

    async def verify_chat(self, backend):
        samples = [
            [{"role": "user", "content": "早上好，hello!"}],
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ],
        ]
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as client:
            for messages in samples:
                async with client.post(
                    backend.rstrip("/") + "/tokenize",
                    json={"model": self.model, "messages": messages},
                ) as response:
                    response.raise_for_status()
                    tokens = self.chat_ids(messages)
                    if not tokens or tokens != (await response.json()).get("tokens"):
                        return False
        self.chat_verified = True
        return True

    def __call__(self, path, payload):
        # Only verified, identity-preserving completion formats get a cache prediction.
        fields = {
            "model",
            "prompt",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "max_tokens",
            "ignore_eos",
            "seed",
            "stop",
            "return_token_ids",
            "add_special_tokens",
        }
        if payload.get("model") != self.model:
            return None
        if path == "/v1/chat/completions" and self.chat_verified:
            if payload.keys() - (
                (fields - {"prompt", "add_special_tokens"})
                | {"messages", "max_completion_tokens"}
            ):
                return None
            tokens = self.chat_ids(payload.get("messages"))
        elif path == "/v1/completions" and not payload.keys() - fields:
            tokens = payload.get("prompt")
        else:
            return None
        if isinstance(tokens, str):
            special = payload.get("add_special_tokens", True)
            if not isinstance(special, bool):
                return None
            tokens = self.tokenizer.encode(tokens, add_special_tokens=special)
        if (
            not isinstance(tokens, list)
            or not 0 < len(tokens) <= 4096
            or not all(type(t) is int and t >= 0 for t in tokens)
        ):
            return None
        hashes = []
        parent = None
        # vLLM recomputes the last token, so the last full block at an exact
        # boundary is ineligible even when its hash exists in the cache.
        for start in range(0, len(tokens) - self.block_size, self.block_size):
            parent = self.hash_block(
                self.function,
                parent,
                tuple(tokens[start : start + self.block_size]),
                extra_keys=None,
            )
            hashes.append(block_key(self.external(parent)))
        return len(tokens), hashes
