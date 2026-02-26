import jax
import jax.numpy as jnp
import pytest

from jaxformers.ops.cross_entropy.config import infer_block_sizes
from jaxformers.ops.cross_entropy.pallas_tpu import linear_softmax_cross_entropy_loss_pallas
from jaxformers.ops.cross_entropy.reference import cross_entropy_reference


@pytest.mark.require_tpu
def test_pallas_cross_entropy_forward_backward_matches_reference():
    if jax.default_backend() != "tpu":
        pytest.skip("TPU backend required for pallas_tpu kernel test")

    b, h, v = 1024, 512, 4096
    key = jax.random.PRNGKey(0)
    key_x, key_w, key_y = jax.random.split(key, 3)
    x = jax.random.normal(key_x, (b, h), dtype=jnp.bfloat16)
    w = jax.random.normal(key_w, (v, h), dtype=jnp.bfloat16)
    labels = jax.random.randint(key_y, (b,), 0, v, dtype=jnp.int32)

    block_sizes = infer_block_sizes("pallas_tpu", b, h, v, dtype=jnp.float32)

    loss_ref, lse_ref = cross_entropy_reference(
        x,
        labels,
        w,
        dtype=jnp.float32,
        logit_soft_cap=30.0,
    )
    loss_pal, lse_pal = linear_softmax_cross_entropy_loss_pallas(
        x,
        labels,
        w,
        block_sizes=block_sizes,
        dtype=jnp.float32,
        logit_soft_cap=30.0,
    )

    assert jnp.allclose(loss_pal, loss_ref, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(lse_pal, lse_ref, atol=1e-4, rtol=1e-4)

    def obj_ref(x_, w_):
        loss, lse = cross_entropy_reference(
            x_,
            labels,
            w_,
            dtype=jnp.float32,
            logit_soft_cap=30.0,
        )
        return jnp.sum(loss) + 1e-4 * jnp.sum(lse**2)

    def obj_pal(x_, w_):
        loss, lse = linear_softmax_cross_entropy_loss_pallas(
            x_,
            labels,
            w_,
            block_sizes=block_sizes,
            dtype=jnp.float32,
            logit_soft_cap=30.0,
        )
        return jnp.sum(loss) + 1e-4 * jnp.sum(lse**2)

    dx_ref, dw_ref = jax.grad(obj_ref, argnums=(0, 1))(x, w)
    dx_pal, dw_pal = jax.grad(obj_pal, argnums=(0, 1))(x, w)

    assert jnp.max(jnp.abs(dx_ref.astype(jnp.float32) - dx_pal.astype(jnp.float32))) < 1e-2
    assert jnp.max(jnp.abs(dw_ref.astype(jnp.float32) - dw_pal.astype(jnp.float32))) < 1e-2
