import jax
import jax.numpy as jnp

from jaxformers.optimizers import adam


def _tree_nbytes(pytree) -> int:
    total = 0
    for leaf in jax.tree.leaves(pytree):
        if hasattr(leaf, "dtype") and hasattr(leaf, "size"):
            total += int(leaf.size) * int(leaf.dtype.itemsize)
    return total


def test_adam_freeze_mask_does_not_allocate_state_for_frozen_params():
    params = {
        "frozen": jnp.zeros((10_000,), dtype=jnp.float32),
        "train": jnp.zeros((10,), dtype=jnp.float32),
    }
    freeze_mask = {"frozen": True, "train": False}

    tx = adam.make(
        learning_rate=1e-3,
        grad_accum=1,
        max_grad_norm=None,
        freeze_mask=freeze_mask,
    )
    state = tx.init(params)

    # If we accidentally allocate Adam/apply_every state for the frozen param,
    # this will be ~O(10_000) floats, i.e. tens of KB+, instead of a few hundred bytes.
    assert _tree_nbytes(state) < 1_000

