import jax
import jax.numpy as jnp
from functools import partial
from jaxtyping import Array, Float, Int

from .config import BlockSizes

def _apply_logit_soft_cap(logits: Float[Array, "B V"], logit_soft_cap: float | None = None) -> Float[Array, "B V"]:
    if logit_soft_cap is None:
        return logits
    return jnp.tanh(logits / logit_soft_cap) * logit_soft_cap


def _infer_named_sharding(x):
    # Works for both eager jax.Arrays (x.sharding) and some tracers (x.aval.sharding).
    sharding = getattr(x, "sharding", None)
    if sharding is None:
        aval = getattr(x, "aval", None)
        sharding = getattr(aval, "sharding", None) if aval is not None else None
    return (
        sharding if isinstance(sharding, jax.sharding.NamedSharding) else None
    )


def _named_sharding_is_nontrivial(named_sharding: jax.sharding.NamedSharding, shape) -> bool:
    # Consider it non-trivial if any non-broadcasted dim is partitioned across an axis
    # with size > 1.
    mesh = named_sharding.mesh
    spec = tuple(named_sharding.spec)
    spec = spec + (None,) * (len(shape) - len(spec))
    for axis_spec, dim in zip(spec, shape):
        if dim == 1 or axis_spec is None:
            continue
        if isinstance(axis_spec, tuple):
            if any(mesh.shape.get(a, 1) > 1 for a in axis_spec):
                return True
        else:
            if mesh.shape.get(axis_spec, 1) > 1:
                return True
    return False


def _fused_cross_entropy_chunked_xla_body(
    x: Float[Array, "B H"],
    labels: Int[Array, " B"],
    w: Float[Array, "V H"],
    *,
    block_sizes: BlockSizes,
    dtype: jnp.dtype = jnp.float32,
    logit_soft_cap: float | None = None,
    precision: jax.lax.PrecisionLike = None,
):
    B, H = x.shape
    V, _ = w.shape
    b_block = block_sizes.b
    v_block = block_sizes.v
    h_block = block_sizes.h

    if B % b_block != 0:
        raise ValueError(f"B={B} must be divisible by b_block={b_block}")
    if H % h_block != 0:
        raise ValueError(f"H={H} must be divisible by h_block={h_block}")

    Vpad = (-V) % v_block
    w_pad = jnp.pad(w, ((0, Vpad), (0, 0)))

    num_b = B // b_block
    num_h = H // h_block
    num_v = (V + Vpad) // v_block

    def b_body(bi, val):
        lse, loss = val
        b0 = b_block * bi
        yb = jax.lax.dynamic_slice(labels, (b0,), (b_block,))  # [B_chunk]

        lse_b = jnp.full((b_block,), -jnp.inf, dtype=dtype)
        label_logits_b = jnp.full((b_block,), -jnp.inf, dtype=dtype)

        @jax.checkpoint
        def v_body(vi, val):
            lse_b, label_logits_b = val
            v0 = vi * v_block


            def h_body(hi, acc):
                h0 = h_block * hi
                x_bh = jax.lax.dynamic_slice(x, (b0, h0), (b_block, h_block))
                w_hv = jax.lax.dynamic_slice(w_pad, (v0, h0), (v_block, h_block))
                return acc + jax.lax.dot_general(x_bh, w_hv, (((1,), (1,)), ((), ())), precision, preferred_element_type = dtype)

            logits = jax.lax.fori_loop(
                0, num_h, h_body, jnp.zeros((b_block, v_block), dtype = dtype)
            )  # [B_block, V_block]
            logits = _apply_logit_soft_cap(logits, logit_soft_cap)
            valid = v0 + jnp.arange(v_block) < V
            logits = jnp.where(valid, logits, -jnp.inf)

            in_block = (yb >= v0) & (yb < v0 + v_block)
            idx = jnp.where(in_block, yb - v0, 0)
            block_label_logits = logits[jnp.arange(b_block), idx]
            label_logits_b = jnp.where(in_block, block_label_logits, label_logits_b)

            block_lse = jax.nn.logsumexp(logits, axis=-1)  # (B_block, )
            lse_b = jnp.logaddexp(lse_b, block_lse)  # (B_block,)

            return lse_b, label_logits_b

        lse_b, label_logits_b = jax.lax.fori_loop(
            0, num_v, v_body, (lse_b, label_logits_b)
        )  # (B_block)
        loss_b = lse_b - label_logits_b
        lse = jax.lax.dynamic_update_slice(lse, lse_b, (b0,))
        loss = jax.lax.dynamic_update_slice(loss, loss_b, (b0,))
        return lse, loss

    lse0, loss0 = jnp.zeros((B,), dtype), jnp.zeros((B,), dtype)
    (lse, loss) = jax.lax.fori_loop(0, num_b, b_body, (lse0, loss0))

    return loss, lse


_fused_cross_entropy_chunked_xla_direct = partial(
    jax.jit,
    static_argnames=["block_sizes", "dtype", "precision"],
)(_fused_cross_entropy_chunked_xla_body)


def fused_cross_entropy_chunked_xla(
    x: Float[Array, "B H"],
    labels: Int[Array, " B"],
    w: Float[Array, "V H"],
    *,
    block_sizes: BlockSizes,
    dtype: jnp.dtype = jnp.float32,
    logit_soft_cap: float | None = None,
    precision: jax.lax.PrecisionLike = None,
):
    """Chunked CE kernel.

    If inputs are sharded on the batch axis, run the kernel under `shard_map` so
    internal loop carries remain type-stable (JAX includes sharding in carry types).
    """
    # Prefer label sharding (what the caller likely intends). If labels are replicated
    # but activations are sharded, fall back to x so we still avoid carry type changes.
    labels_sharding = _infer_named_sharding(labels)
    x_sharding = _infer_named_sharding(x)
    batch_sharding = None
    if labels_sharding is not None and _named_sharding_is_nontrivial(
        labels_sharding, labels.shape
    ):
        batch_sharding = labels_sharding
    elif x_sharding is not None and _named_sharding_is_nontrivial(x_sharding, x.shape):
        batch_sharding = x_sharding

    if batch_sharding is None:
        return _fused_cross_entropy_chunked_xla_direct(
            x,
            labels,
            w,
            block_sizes=block_sizes,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
        )

    from jax.experimental import shard_map
    from jax.sharding import PartitionSpec as P

    mesh = batch_sharding.mesh
    spec = tuple(batch_sharding.spec) + (None,) * (labels.ndim - len(batch_sharding.spec))
    batch_axis = spec[0]  # labels is [B]
    if batch_axis is None:
        return _fused_cross_entropy_chunked_xla_direct(
            x,
            labels,
            w,
            block_sizes=block_sizes,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
        )

    def per_shard(x_local, labels_local, w_rep):
        return _fused_cross_entropy_chunked_xla_body(
            x_local,
            labels_local,
            w_rep,
            block_sizes=block_sizes,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
        )

    per_shard_mapped = shard_map.shard_map(
        per_shard,
        mesh,
        (P(batch_axis, None), P(batch_axis), P(None, None)),
        (P(batch_axis), P(batch_axis)),
        check_rep=False,
    )

    return per_shard_mapped(x, labels, w)
