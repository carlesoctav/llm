import jax
import jax.numpy as jnp

from jaxformers.ops.cross_entropy.config import BlockSizes
from jaxformers.ops.cross_entropy.reference import cross_entropy_reference
from jaxformers.ops.cross_entropy.xla_chunked import fused_cross_entropy_chunked_xla


def test_xla_chunked_custom_vjp_forward_backward_matches_reference():
    b, h, v = 256, 128, 512
    block_sizes = BlockSizes(v=256, h=128, b=128)

    key = jax.random.PRNGKey(0)
    key_x, key_w, key_y = jax.random.split(key, 3)
    x = jax.random.normal(key_x, (b, h), dtype=jnp.bfloat16)
    w = jax.random.normal(key_w, (v, h), dtype=jnp.bfloat16)
    labels = jax.random.randint(key_y, (b,), 0, v, dtype=jnp.int32)

    loss_ref, lse_ref = cross_entropy_reference(
        x,
        labels,
        w,
        dtype=jnp.float32,
        logit_soft_cap=30.0,
    )
    loss_xla, lse_xla = fused_cross_entropy_chunked_xla(
        x,
        labels,
        w,
        block_sizes=block_sizes,
        dtype=jnp.float32,
        logit_soft_cap=30.0,
    )

    assert jnp.allclose(loss_xla, loss_ref, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(lse_xla, lse_ref, atol=1e-4, rtol=1e-4)

    def obj_ref(x_, w_):
        loss, lse = cross_entropy_reference(
            x_,
            labels,
            w_,
            dtype=jnp.float32,
            logit_soft_cap=30.0,
        )
        return jnp.sum(loss) + 1e-4 * jnp.sum(lse**2)

    def obj_xla(x_, w_):
        loss, lse = fused_cross_entropy_chunked_xla(
            x_,
            labels,
            w_,
            block_sizes=block_sizes,
            dtype=jnp.float32,
            logit_soft_cap=30.0,
        )
        return jnp.sum(loss) + 1e-4 * jnp.sum(lse**2)

    dx_ref, dw_ref = jax.grad(obj_ref, argnums=(0, 1))(x, w)
    dx_xla, dw_xla = jax.grad(obj_xla, argnums=(0, 1))(x, w)

    assert jnp.max(jnp.abs(dx_ref.astype(jnp.float32) - dx_xla.astype(jnp.float32))) < 2e-2
    assert jnp.max(jnp.abs(dw_ref.astype(jnp.float32) - dw_xla.astype(jnp.float32))) < 2e-2
