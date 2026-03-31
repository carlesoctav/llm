import jax
import jax.numpy as jnp

from jaxformers.distributed.parallel import DEFAULT_PARALLEL_DIMS
from jaxformers.models import qwen3


model = qwen3.load("Qwen/Qwen3-0.6B", DEFAULT_PARALLEL_DIMS)


@jax.jit
def train_step(model):
    print(model.config)
    print(type(model.config))
    return model.forward(
        model.weights,
        jnp.ones(
            (
                1,
                20,
            ),
            dtype=jnp.int32,
        ),
    )


a = train_step(model)
