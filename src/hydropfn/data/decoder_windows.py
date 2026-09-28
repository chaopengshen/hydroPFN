"""Window schedules with explicit real history and exactly-once scoring."""
from __future__ import annotations


def training_starts(start, stop, length, stride):
    if length < 1 or stride < 1 or stop - start < length:
        raise ValueError("invalid training window extent")
    starts = list(range(start, stop - length + 1, stride))
    if starts[-1] != stop - length:
        starts.append(stop - length)
    return starts


def evaluation_windows(start, stop, length, warmup):
    """Return (input_start, score_start, score_stop); never use right padding."""
    if not 0 <= warmup < length or start < warmup or stop <= start:
        raise ValueError("invalid scoring extent or warmup")
    if stop < length:
        raise ValueError("not enough historical record")
    out = []
    for lo in range(start, stop, length - warmup):
        hi = min(lo + length - warmup, stop)
        input_start = min(lo - warmup, stop - length)
        if input_start < 0 or lo - input_start < warmup:
            raise ValueError("insufficient real warmup")
        out.append((input_start, lo, hi))
    return out
