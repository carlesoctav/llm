import jax.numpy as jnp
import optax
from jaxformers.optimizers.divide import divide_every
from jaxformers.optimizers.log_grad_norm import get_logged_grad_norm, log_grad_norm


def test_log_grad_norm_updates_only_on_emit_and_is_pre_clip():
    tx = optax.chain(
        optax.apply_every(2),
        divide_every(2),
        log_grad_norm(),
        optax.clip_by_global_norm(1.0),
    )

    params = {"w": jnp.zeros((2,), dtype=jnp.float32)}
    state = tx.init(params)

    g1 = {"w": jnp.array([3.0, 4.0], dtype=jnp.float32)}
    g2 = {"w": jnp.array([0.0, 4.0], dtype=jnp.float32)}
    t1 = jnp.asarray(3, dtype=jnp.int32)
    t2 = jnp.asarray(1, dtype=jnp.int32)

    u1, state = tx.update(g1, state, params, token_count=t1)
    assert float(optax.global_norm(u1)) == 0.0
    assert float(get_logged_grad_norm(state)) == 0.0

    u2, state = tx.update(g2, state, params, token_count=t2)

    mean_grad = {"w": (g1["w"] + g2["w"]) / jnp.asarray(4.0, dtype=jnp.float32)}
    expected_logged = optax.global_norm(mean_grad)
    logged = get_logged_grad_norm(state)
    assert logged is not None
    assert jnp.allclose(logged, expected_logged)

    # Clipping happens after logging; ensure the emitted update is clipped.
    assert float(optax.global_norm(u2)) == 1.0
