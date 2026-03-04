import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxformers.attention_utils import eager_dot_product_attention
from jaxformers.ops.attention import xla_chunked_dot_product_attention


def _causal_mask(B: int, T: int) -> jax.Array:
    base = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    return jnp.broadcast_to(base, (B, 1, T, T))


@pytest.mark.parametrize("K", [1, 4])
@pytest.mark.parametrize("mode", ["mask", "is_causal", "none"])
def test_xla_chunked_matches_eager_small(K: int, mode: str):
    B, T, N, H = 2, 8, 4, 16
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    q = jax.random.normal(kq, (B, T, N, H), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (B, T, K, H), dtype=jnp.bfloat16)
    v = jax.random.normal(kv, (B, T, K, H), dtype=jnp.bfloat16)

    if mode == "mask":
        mask = _causal_mask(B, T)
        is_causal = False
        ref_mask = mask
    elif mode == "is_causal":
        mask = None
        is_causal = True
        ref_mask = _causal_mask(B, T)
    else:
        mask = None
        is_causal = False
        ref_mask = None

    out = xla_chunked_dot_product_attention(
        q,
        k,
        v,
        mask=mask,
        is_causal=is_causal,
        query_chunk_size=4,
        key_chunk_size=4,
        precision=jax.lax.Precision.HIGHEST,
    )
    ref = eager_dot_product_attention(q, k, v, mask=ref_mask)

    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32),
        np.asarray(ref, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )

