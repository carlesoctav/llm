from __future__ import annotations


def round_up_to_power_of_two(value: int, *, max_value: int) -> int:
    if value <= 0:
        raise ValueError(f"`value` must be positive, got {value}.")
    if max_value <= 0:
        raise ValueError(f"`max_value` must be positive, got {max_value}.")
    if value > max_value:
        raise ValueError(f"`value` ({value}) exceeds `max_value` ({max_value}).")
    # Round up to the next power of two.
    rounded = 1 << (value - 1).bit_length()
    return min(rounded, max_value)


def power_of_two_buckets(max_value: int) -> list[int]:
    if max_value <= 0:
        raise ValueError(f"`max_value` must be positive, got {max_value}.")
    buckets: list[int] = []
    v = 1
    while v < max_value:
        buckets.append(v)
        v *= 2
    buckets.append(max_value)
    # De-dupe if max_value is already a power of 2.
    out: list[int] = []
    for x in buckets:
        if not out or out[-1] != x:
            out.append(x)
    return out
