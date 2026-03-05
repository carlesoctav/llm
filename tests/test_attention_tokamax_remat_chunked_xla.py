import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxformers.ops.attention import tokamax_remat_chunked_xla_dot_product_attention


def _causal_mask(B: int, T: int) -> jax.Array:
    base = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    return jnp.broadcast_to(base, (B, 1, T, T))


def _sdpa_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    mask: jax.Array | None,
    is_causal: bool,
) -> jax.Array:
    # `jax.nn.dot_product_attention` expects query/key/value to share head count.
    B, T, N, H = q.shape
    _, S, K, _ = k.shape
    if K != N:
        if K <= 0 or N % K != 0:
            raise ValueError("N must be a positive multiple of K for MQA/GQA reference")
        repeat = N // K
        k = jnp.repeat(k, repeat, axis=-2)
        v = jnp.repeat(v, repeat, axis=-2)

    # Match tokamax/jaxformers mask convention: [B, 1, T, S] or [B, N, T, S].
    if mask is not None and mask.shape == (B, 1, T, S):
        mask = jnp.broadcast_to(mask, (B, N, T, S))

    return jax.nn.dot_product_attention(
        q,
        k,
        v,
        mask=mask,
        is_causal=is_causal,
        implementation="xla",
    )


@pytest.mark.parametrize("K", [1, 4])
@pytest.mark.parametrize("mode", ["mask", "is_causal", "none"])
def test_tokamax_remat_xla_chunked_matches_sdpa_small(K: int, mode: str):
    B, T, N, H = 2, 8, 4, 16
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    q = jax.random.normal(kq, (B, T, N, H), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (B, T, K, H), dtype=jnp.bfloat16)
    v = jax.random.normal(kv, (B, T, K, H), dtype=jnp.bfloat16)

    if mode == "mask":
        mask = _causal_mask(B, T)
        is_causal = False
    elif mode == "is_causal":
        mask = None
        is_causal = True
    else:
        mask = None
        is_causal = False

    out = tokamax_remat_chunked_xla_dot_product_attention(
        q,
        k,
        v,
        mask=mask,
        is_causal=is_causal,
        precision=jax.lax.Precision.HIGHEST,
        query_chunk_size=4,
        key_chunk_size=4,
    )
    ref = _sdpa_reference(q, k, v, mask=mask, is_causal=is_causal)

    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32),
        np.asarray(ref, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )

    def ours(q_in, k_in, v_in, *, mask, is_causal):
        return tokamax_remat_chunked_xla_dot_product_attention(
            q_in,
            k_in,
            v_in,
            mask=mask,
            is_causal=is_causal,
            precision=jax.lax.Precision.HIGHEST,
            query_chunk_size=4,
            key_chunk_size=4,
        )

    def sdpa(q_in, k_in, v_in, *, mask, is_causal):
        return _sdpa_reference(q_in, k_in, v_in, mask=mask, is_causal=is_causal)

    def loss(attn_fn, q_in, k_in, v_in):
        out = attn_fn(q_in, k_in, v_in, mask=mask, is_causal=is_causal)
        return jnp.sum(out, dtype=jnp.float32)

    dq_o, dk_o, dv_o = jax.grad(lambda q_in, k_in, v_in: loss(ours, q_in, k_in, v_in), argnums=(0, 1, 2))(
        q, k, v
    )
    dq_r, dk_r, dv_r = jax.grad(lambda q_in, k_in, v_in: loss(sdpa, q_in, k_in, v_in), argnums=(0, 1, 2))(
        q, k, v
    )

    np.testing.assert_allclose(np.asarray(dq_o, np.float32), np.asarray(dq_r, np.float32), rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(np.asarray(dk_o, np.float32), np.asarray(dk_r, np.float32), rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(np.asarray(dv_o, np.float32), np.asarray(dv_r, np.float32), rtol=5e-2, atol=5e-2)
