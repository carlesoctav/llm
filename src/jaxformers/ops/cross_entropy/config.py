import jax.numpy as jnp
from typing import NamedTuple


class BlockSizes(NamedTuple):
    v: int | None
    h: int | None
    b: int | None


def infer_block_sizes(impl, b, h, v, *, dtype=None, device_kind=None):
    if impl == "xla_chunked":
        return infer_xla_chunked_block_size(
            b, h, v, dtype=dtype, device_kind=device_kind
        )
    elif impl == "pallas_tpu":
        return infer_pallas_tpu_block_size(
            b, h, v, dtype=dtype, device_kind=device_kind
        )
    elif impl == "reference":
        return None
    else:
        raise NotImplementedError


def _largest_divisor_leq(n: int, limit: int) -> int:
    limit = min(n, limit)
    for d in range(limit, 0, -1):
        if n % d == 0:
            return d
    return 1


def _largest_divisor_multiple_of_128(n: int, limit: int) -> int:
    upper = min(n, limit)
    upper -= upper % 128
    for d in range(upper, 127, -128):
        if n % d == 0:
            return d
    if n % 128 == 0:
        return 128
    return 1


def infer_xla_chunked_block_size(
    b: int,
    h: int,
    v: int,
    *,
    dtype: jnp.dtype | None = None,
    device_kind: str | None = None,
) -> BlockSizes:
    del dtype, device_kind  # not used for now
    if b <= 0 or h <= 0 or v <= 0:
        raise ValueError(f"b, h, v must all be > 0, got b={b}, h={h}, v={v}.")

    # Avoid pathological defaults like b=b and v=32768, which materializes a massive
    # [b_block, v_block] logits tile (and corresponding backward buffers).
    # b_block = _largest_divisor_leq(b, 1024)
    # h_block = _largest_divisor_leq(h, 512)
    # h_block = h
    # b_block = b
    max_v = min(v, 8192)

    b_block = 1024
    h_block = 512
    v_block = max(128, 128 * (max_v // 128))

    return BlockSizes(v=v_block, h=h_block, b=b_block)


def infer_pallas_tpu_block_size(
    b: int,
    h: int,
    v: int,
    *,
    dtype: jnp.dtype | None = None,
    device_kind: str | None = None,
) -> BlockSizes:
    del dtype, device_kind  # not used for now
    if b <= 0 or h <= 0 or v <= 0:
        raise ValueError(f"b, h, v must all be > 0, got b={b}, h={h}, v={v}.")

    # TPU-friendly defaults. b/h must divide local shape for the Pallas kernel.
    b_block = _largest_divisor_multiple_of_128(b, 1024)
    h_block = _largest_divisor_multiple_of_128(h, 256)
    if v >= 128:
        v_block = min(256, 128 * (v // 128))
    else:
        v_block = v

    return BlockSizes(v=v_block, h=h_block, b=b_block)
