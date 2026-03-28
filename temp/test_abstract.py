from jaxformers.print_utils import tree_pprint
from contextlib import contextmanager, nullcontext
import dataclasses

import jax
from transformers import Gemma3TextConfig

from jaxformers.modeling_utils import StoreWeights
from jaxformers.models.huggingface import gemma3
from jaxformers.optimizers import make_optimizer
from jaxformers.optimizers.scheduler import make_scheduler


@contextmanager
def disable_inner_set_mesh():
    # Temporary workaround for the current gemma3.init implementation, which
    # calls jax.set_mesh inside the traced function.
    orig_set_mesh = jax.set_mesh
    jax.set_mesh = lambda mesh: nullcontext()
    try:
        yield orig_set_mesh
    finally:
        jax.set_mesh = orig_set_mesh


def main():
    config = Gemma3TextConfig()
    devices = list(jax.devices())
    parallel_dims = {
        "dp_replicate": 1,
        "dp_shard": len(devices),
        "cp": 1,
        "tp": 1,
    }

    axis_shapes = tuple(parallel_dims.values())
    axis_names = tuple(parallel_dims.keys())
    axis_types = tuple(jax.sharding.AxisType.Explicit for _ in axis_names)
    mesh = jax.make_mesh(
        axis_shapes,
        axis_names,
        axis_types=axis_types,
        devices=devices,
    )

    rng = jax.random.key(0)
    scheduler = make_scheduler(None, 1e-3)

    with disable_inner_set_mesh() as outer_set_mesh:
        with outer_set_mesh(mesh):
            model = jax.eval_shape(
                lambda: gemma3.init(
                    config=config,
                    parallel_dims=parallel_dims,
                    devices=devices,
                    rngs=rng,
                    tokenizer=None,
                )
            )

            model = jax.eval_shape(
                lambda m: dataclasses.replace(
                    m,
                    weights=m.prepare_weights(m.weights, StoreWeights.FREE),
                ),
                model,
            )

            model = jax.eval_shape(
                lambda m: make_optimizer(
                    "adam",
                    m,
                    scheduler,
                    {},
                ),
                model,
            )

    tree_pprint(model.weights)
    weight_leaf = jax.tree.leaves(model.weights)[0]
    opt_leaf = jax.tree.leaves(model.opt_state)[0]
    tree_pprint(opt_leaf)

    print("backend:", jax.default_backend())
    print("devices:", devices)
    print("weights leaf type:", type(weight_leaf))
    print("weights leaf:", weight_leaf)
    print("opt_state leaf type:", type(opt_leaf))
    print("opt_state leaf:", opt_leaf)


if __name__ == "__main__":
    main()
