"""Rank-local, reset-synchronized Qwen3.5 aligned cache evidence on vLLM 0.23."""

from collections import Counter

BLOCK = 2048
GROUPS = ("full_attention", "mamba", "mamba", "mamba")


def block_key(value):
    if type(value) is int and 0 <= value < 1 << 64:
        return value.to_bytes(8, "big")
    if isinstance(value, bytes) and value:
        return value
    raise ValueError("invalid vLLM block hash")


class CacheView:
    """One publisher incarnation; stale/unknown evidence cannot select a rank."""

    def __init__(self, rank):
        self.rank = rank
        self.sequence = None
        self.trusted = False
        self.groups = [Counter() for _ in GROUPS]

    def invalidate(self):
        self.trusted = False
        for group in self.groups:
            group.clear()

    def match(self, hashes, length):
        if not self.trusted:
            return None
        result = 0
        for i, key in enumerate(hashes[: max(0, length - 1) // BLOCK], start=1):
            if key not in self.groups[0]:
                break
            if all(key in g for g in self.groups[1:]):
                result = i * BLOCK
        return result

    def apply(self, sequence, batch):
        try:
            if len(batch) < 3 or batch[2] != self.rank:
                raise ValueError("unexpected publisher rank")
            if self.sequence is not None and sequence != self.sequence + 1:
                self.invalidate()
            self.sequence = sequence
            for event in batch[1]:
                if event[0] == "AllBlocksCleared":
                    self.invalidate()
                    self.trusted = True
                elif self.trusted:
                    self._change(event)
        except (ValueError, TypeError, IndexError, KeyError):
            self.invalidate()
            raise

    def _change(self, event):
        kind = event[0]
        if kind == "BlockStored":
            _, hashes, _, _, size, lora, medium, name, extras, group, spec, window = (
                event[:12]
            )
            if medium not in (None, "GPU", "NPU"):
                return
            if (
                type(group) is not int
                or not 0 <= group < len(GROUPS)
                or size != BLOCK
                or spec != GROUPS[group]
                or lora is not None
                or name
                or (extras and any(extras))
                or window
            ):
                raise ValueError("unqualified cache layout or identity")
            self.groups[group].update(block_key(k) for k in hashes)
        elif kind == "BlockRemoved":
            _, hashes, medium, group = event[:4]
            if medium not in (None, "GPU", "NPU"):
                return
            if type(group) is not int or not 0 <= group < len(GROUPS):
                raise ValueError("unknown removed group")
            for raw in hashes:
                key = block_key(raw)
                counts = self.groups[group]
                if not counts[key]:
                    raise ValueError("removal has no corresponding stored block")
                counts[key] -= 1
                if not counts[key]:
                    del counts[key]
        else:
            raise ValueError("unexpected cache event")
