import jax.numpy as jnp
import optax

from jaxformers.optimizers.lr import (
    custom_scale_by_learning_rate,
    get_logged_learning_rate,
    get_logged_step_size,
)


def test_scale_by_learning_rate_logs_lr_and_steps_only_on_emit():
    def sched(step):
        return step.astype(jnp.float32) + 1.0

    tx = optax.chain(
        optax.apply_every(2),
        custom_scale_by_learning_rate(sched, apply_every_k=2),
    )

    params = {"w": jnp.zeros((1,), dtype=jnp.float32)}
    state = tx.init(params)

    lr0 = get_logged_learning_rate(state)
    step0 = get_logged_step_size(state)
    assert lr0 is not None and float(lr0) == 1.0
    assert step0 is not None and float(step0) == -1.0

    g1 = {"w": jnp.asarray([1.0], dtype=jnp.float32)}
    u1, state = tx.update(g1, state, params)
    assert float(optax.global_norm(u1)) == 0.0
    assert float(get_logged_learning_rate(state)) == 1.0

    g2 = {"w": jnp.asarray([2.0], dtype=jnp.float32)}
    u2, state = tx.update(g2, state, params)
    assert float(u2["w"][0]) == -3.0
    assert float(get_logged_learning_rate(state)) == 1.0
    assert float(get_logged_step_size(state)) == -1.0

    g3 = {"w": jnp.asarray([1.0], dtype=jnp.float32)}
    u3, state = tx.update(g3, state, params)
    assert float(optax.global_norm(u3)) == 0.0
    assert float(get_logged_learning_rate(state)) == 1.0

    g4 = {"w": jnp.asarray([1.0], dtype=jnp.float32)}
    u4, state = tx.update(g4, state, params)
    assert float(u4["w"][0]) == -4.0
    assert float(get_logged_learning_rate(state)) == 2.0
    assert float(get_logged_step_size(state)) == -2.0
