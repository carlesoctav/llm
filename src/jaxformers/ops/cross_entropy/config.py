import jax.numpy as jnp
from typing import NamedTuple

class BlockSizes(NamedTuple):
    v: int | None
    h: int | None
    b: int | None


def infer_block_sizes(impl, b, h, v, *, dtype = None, device_kind = None):
    if impl == "xla_chunked":
        return infer_xla_chunked_block_size(b, h, v, dtype = dtype, device_kind = device_kind)
    elif impl == "reference":
        return None
    else:
        raise NotImplementedError


def infer_xla_chunked_block_size(
    b: int,
    h: int,
    v: int,
    *,
    dtype: jnp.dtype | None = None,
    device_kind: str | None = None,
) -> int:
    del dtype, device_kind # not used for now
    target = min(v, 32768)
    if target <= 0:
        return 1
    if target == v:
        return target

    return BlockSizes(b = b, v=max(128, 128 * (target // 128)), h = h)
