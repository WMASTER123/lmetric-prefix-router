"""Read one pinned vLLM installation's tokenizer and native block hash functions."""

import os
from importlib.metadata import version

from .cache023 import BLOCK, block_key


class HostFingerprint:
    def __init__(self, model_path, served_model):
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
        self.tokenizer = get_tokenizer(model_path)
        self.function = get_hash_fn_by_name("sha256")
        init_none_hash(self.function)
        self.hash_block = hash_block_tokens
        self.external = maybe_convert_block_hash

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
        if (
            path != "/v1/completions"
            or payload.get("model") != self.model
            or payload.keys() - fields
        ):
            return None
        tokens = payload.get("prompt")
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
        for start in range(0, len(tokens) - BLOCK + 1, BLOCK):
            parent = self.hash_block(
                self.function,
                parent,
                tuple(tokens[start : start + BLOCK]),
                extra_keys=None,
            )
            hashes.append(block_key(self.external(parent)))
        return len(tokens), hashes
