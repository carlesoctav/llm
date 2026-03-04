import math
from typing import cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

_FLASH_IMPORT_ERROR: Exception | None
try:
    from jax.experimental.pallas.ops.tpu.flash_attention import (  # type: ignore[attr-defined]
        BlockSizes as _FlashBlockSizes,
        flash_attention as _tpu_flash_attention,
    )
except Exception as exc:  # pragma: no cover
    _tpu_flash_attention = None
    _FlashBlockSizes = None
    _FLASH_IMPORT_ERROR = exc
else:
    _FLASH_IMPORT_ERROR = None


def _canonicalize_bnts(
    x: Array,
    *,
    B: int,
    N: int,
    T: int,
    S: int,
    name: str,
) -> jax.Array:
    """Normalize an additive attention bias to `[B, N, T, S]`."""
    x_arr = jnp.asarray(x)
    if x_arr.ndim == 2:  # [T, S]
        if x_arr.shape != (T, S):
            raise ValueError(f"{name} expected shape (T,S)=({T},{S}), got {x_arr.shape}")
        x_arr = x_arr[None, None, :, :]
    elif x_arr.ndim == 3:
        if x_arr.shape == (B, T, S):  # [B, T, S] -> [B, 1, T, S]
            x_arr = x_arr[:, None, :, :]
        elif x_arr.shape == (N, T, S):  # [N, T, S] -> [1, N, T, S]
            x_arr = x_arr[None, :, :, :]
        else:
            raise ValueError(
                f"{name} expected shape (B,T,S)=({B},{T},{S}) or (N,T,S)=({N},{T},{S}); got {x_arr.shape}"
            )
    elif x_arr.ndim == 4:
        if x_arr.shape[2:] != (T, S):
            raise ValueError(f"{name} expected trailing dims (T,S)=({T},{S}), got {x_arr.shape}")
    else:
        raise ValueError(f"{name} must be rank-2/3/4, got shape {x_arr.shape}")

    Bx, Hx, _, _ = x_arr.shape
    if Bx not in (1, B):
        raise ValueError(f"{name} batch dim must be 1 or B={B}, got {Bx}")
    if Hx not in (1, N):
        raise ValueError(f"{name} head dim must be 1 or N={N}, got {Hx}")

    if Bx == 1 and B != 1:
        x_arr = jnp.broadcast_to(x_arr, (B, Hx, T, S))
    if Hx == 1 and N != 1:
        x_arr = jnp.broadcast_to(x_arr, (B, N, T, S))
    return cast(jax.Array, x_arr)


def flash_attention_dot_product_attention(
    query: Float[Array, "B T N H"],
    key: Float[Array, "B S K H"],
    value: Float[Array, "B S K H"],
    bias: Array | None = None,
    mask: Bool[Array, " B #N T S"] | None = None,
    q_sharding: jax.sharding.NamedSharding | None = None,
    **kwargs,
) -> Float[Array, "B T N H"]:
    """TPU flash-attention wrapper using JAX Pallas.

    Shapes / notation:
      - query:  [B, T, N, H]
      - key:    [B, S, K, H]
      - value:  [B, S, K, H]
      - logits: [B, N, T, S] (conceptual)
      - out:    [B, T, N, H]

    Notes:
      - Uses `jax.experimental.pallas.ops.tpu.flash_attention.flash_attention`.
      - Only supports masking via `is_causal=True` (arbitrary boolean masks are
        not supported without materializing a full `[B,N,T,S]` bias tensor).
      - For MQA/GQA (`K != N`), K/V are broadcast to N heads.
    """
    if _tpu_flash_attention is None:  # pragma: no cover
        raise ImportError("TPU flash_attention is not available in this JAX build") from _FLASH_IMPORT_ERROR

    if jax.default_backend() != "tpu":
        raise NotImplementedError("flash_attention_dot_product_attention currently only supports TPU backends")

    precision = kwargs.pop("precision", None)
    if precision is not None:
        # The pallas kernel controls matmul precision internally.
        pass

    dropout_rate = float(kwargs.pop("dropout_rate", 0.0))
    dropout_rng = kwargs.pop("dropout_rng", None)
    if dropout_rate != 0.0 or dropout_rng is not None:
        raise NotImplementedError("TPU flash_attention wrapper does not support dropout yet")

    is_causal = bool(kwargs.pop("is_causal", False) or kwargs.pop("causal", False))
    debug = bool(kwargs.pop("debug", False))
    segment_ids = kwargs.pop("segment_ids", None)
    block_sizes = kwargs.pop("block_sizes", None)

    if mask is not None:
        raise NotImplementedError("TPU flash_attention wrapper does not support arbitrary boolean masks; use is_causal")

    query = cast(jax.Array, query)
    key = cast(jax.Array, key)
    value = cast(jax.Array, value)

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Expected query/key/value rank-4: [B,T,N,H], [B,S,K,H], [B,S,K,H]")

    B, T, N, H = query.shape
    Bk, S, K, Hk = key.shape
    Bv, Sv, Kv, Hv = value.shape
    if (Bk, Bv) != (B, B):
        raise ValueError("Batch size mismatch between query/key/value")
    if (Sv, Kv, Hv) != (S, K, Hk):
        raise ValueError("Key/value shape mismatch")
    if Hk != H:
        raise ValueError("Query/key/value head-dim mismatch")

    if K <= 0 or N % K != 0:
        raise ValueError("Number of query heads must be a positive multiple of key/value heads")

    sm_scale = float(kwargs.pop("sm_scale", 1.0 / math.sqrt(H)))

    if block_sizes is None:
        if _FlashBlockSizes is None:  # pragma: no cover
            raise ImportError("TPU flash_attention BlockSizes is not available") from _FLASH_IMPORT_ERROR
        block_q = min(128, T)
        block_k = 128
        block_t = math.gcd(int(T), 128)
        if S % block_k != 0:
            raise ValueError(f"flash_attention requires kv_seq_len divisible by 128; got S={S}")
        block_sizes = _FlashBlockSizes(
            block_q=block_q,
            block_k_major=block_k,
            block_k=block_k,
            block_b=1,
            block_q_major_dkv=block_t,
            block_k_major_dkv=block_k,
            block_k_dkv=block_k,
            block_q_dkv=block_t,
            block_k_major_dq=block_k,
            block_k_dq=block_k,
            block_q_dq=block_t,
        )

    def _flash_unsharded(q_in: jax.Array, k_in: jax.Array, v_in: jax.Array) -> jax.Array:
        b, t, n, h = q_in.shape
        _, s, k_heads, _ = k_in.shape

        if k_heads != n:
            group = n // k_heads
            k_in = jnp.broadcast_to(k_in[:, :, :, None, :], (b, s, k_heads, group, h)).reshape((b, s, n, h))
            v_in = jnp.broadcast_to(v_in[:, :, :, None, :], (b, s, k_heads, group, h)).reshape((b, s, n, h))

        q_t = jnp.transpose(q_in, (0, 2, 1, 3))
        k_t = jnp.transpose(k_in, (0, 2, 1, 3))
        v_t = jnp.transpose(v_in, (0, 2, 1, 3))

        ab = None
        if bias is not None:
            ab = _canonicalize_bnts(bias, B=b, N=n, T=t, S=s, name="bias").astype(jnp.float32)

        out_t = _tpu_flash_attention(
            q_t,
            k_t,
            v_t,
            ab=ab,
            segment_ids=segment_ids,
            causal=is_causal,
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            debug=debug,
        )
        return jnp.transpose(out_t, (0, 2, 1, 3))

    sharding = q_sharding or getattr(query, "sharding", None)
    if isinstance(sharding, jax.sharding.NamedSharding) and sharding.mesh.size > 1:
        if bias is not None or segment_ids is not None:
            raise NotImplementedError("Sharded TPU flash_attention wrapper does not support bias/segment_ids yet")
        q_spec = sharding.spec
        b_axis = q_spec[0] if len(q_spec) > 0 else None
        n_axis = q_spec[2] if len(q_spec) > 2 else None
        kv_head_axis = n_axis if (n_axis is not None and K == N) else None
        k_spec = jax.sharding.PartitionSpec(b_axis, None, kv_head_axis, None)
        v_spec = jax.sharding.PartitionSpec(b_axis, None, kv_head_axis, None)

        out = jax.shard_map(
            _flash_unsharded,
            mesh=sharding.mesh,
            in_specs=(q_spec, k_spec, v_spec),
            out_specs=q_spec,
            check_vma=False,
        )(query, key, value)
    else:
        out = _flash_unsharded(query, key, value)

    if q_sharding is not None:
        out = jax.lax.with_sharding_constraint(out, q_sharding)

    return cast(Float[Array, "B T N H"], out)
