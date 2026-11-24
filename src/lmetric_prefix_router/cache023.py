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
        self.sync_state = "awaiting_baseline"
        self.groups = [Counter() for _ in GROUPS]
        # Extra-key blocks still emit ordinary removals. Keep their references
        # outside the index used for plain completion lookups.
        self.excluded = [Counter() for _ in GROUPS]

    def invalidate(self, reason="awaiting_baseline"):
        self.trusted = False
        self.sync_state = reason
        for group in (*self.groups, *self.excluded):
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
            if len(batch) < 3 or type(batch[2]) is not int or batch[2] != self.rank:
                raise ValueError("unexpected publisher rank")
            if self.sequence is not None and sequence != self.sequence + 1:
                self.invalidate("sequence_gap")
            self.sequence = sequence
            for event in batch[1]:
                if event[0] == "AllBlocksCleared":
                    self.invalidate()
                    self.trusted = True
                    self.sync_state = "observed_contiguous"
                elif self.trusted:
                    self._change(event)
        except (ValueError, TypeError, IndexError, KeyError):
            self.invalidate("invalid_event")
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
                or window
            ):
                raise ValueError("unqualified cache layout or identity")
            if extras and len(extras) != len(hashes):
                raise ValueError("incomplete block identities")
            plain, excluded = self.groups[group], self.excluded[group]
            for i, raw in enumerate(hashes):
                key = block_key(raw)
                qualified = not (extras and extras[i])
                dest, other = (plain, excluded) if qualified else (excluded, plain)
                if key in other:
                    raise ValueError("ambiguous external hash across identities")
                dest[key] += 1
        elif kind == "BlockRemoved":
            _, hashes, medium, group = event[:4]
            if medium not in (None, "GPU", "NPU"):
                return
            if type(group) is not int or not 0 <= group < len(GROUPS):
                raise ValueError("unknown removed group")
            for raw in hashes:
                key = block_key(raw)
                counts = (
                    self.excluded[group]
                    if key in self.excluded[group]
                    else self.groups[group]
                )
                if not counts[key]:
                    raise ValueError("removal has no corresponding stored block")
                counts[key] -= 1
                if not counts[key]:
                    del counts[key]
        else:
            raise ValueError("unexpected cache event")
