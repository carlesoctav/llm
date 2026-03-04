import pytest

import jax
import jax.numpy as jnp

from jaxformers.attention_utils import eager_dot_product_attention
from jaxformers.ops.attention import flash_attention_dot_product_attention


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU flash_attention requires TPU backend")
def test_flash_attention_matches_eager_forward_and_grad() -> None:
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    B, T, N, K, H = 1, 128, 4, 1, 128
    q = jax.random.normal(kq, (B, T, N, H), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (B, T, K, H), dtype=jnp.bfloat16)
    v = jax.random.normal(kv, (B, T, K, H), dtype=jnp.bfloat16)
    base = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    mask = jnp.broadcast_to(base, (B, 1, T, T))

    def loss_flash(q_in, k_in, v_in):
        out = flash_attention_dot_product_attention(q_in, k_in, v_in, is_causal=True)
        return jnp.sum(out, dtype=jnp.float32)

    def loss_eager(q_in, k_in, v_in):
        out = eager_dot_product_attention(q_in, k_in, v_in, mask=mask)
        return jnp.sum(out, dtype=jnp.float32)

    out_flash = flash_attention_dot_product_attention(q, k, v, is_causal=True)
    out_eager = eager_dot_product_attention(q, k, v, mask=mask)

    assert out_flash.shape == out_eager.shape
    assert jnp.allclose(out_flash.astype(jnp.float32), out_eager.astype(jnp.float32), rtol=2e-2, atol=2e-2)

    grads_flash = jax.grad(loss_flash, argnums=(0, 1, 2))(q, k, v)
    grads_eager = jax.grad(loss_eager, argnums=(0, 1, 2))(q, k, v)
    for g_flash, g_eager in zip(grads_flash, grads_eager, strict=True):
        assert g_flash.shape == g_eager.shape
        assert jnp.allclose(g_flash.astype(jnp.float32), g_eager.astype(jnp.float32), rtol=5e-2, atol=5e-2)
