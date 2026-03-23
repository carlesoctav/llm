from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp

from jaxformers import metric_utils
from jaxformers.data import make_eval_data


def _check_shape(shape, tree):
    def _f(path, leaf):
        assert leaf.shape == shape, f"metrics {path} must have shape {shape}"

    jax.tree.map(_f, tree)


@partial(jax.jit, static_argnums=0)
def _eval_step(fn, model, batch):
    metrics = fn(model, batch)
    mask = batch.get("_mask", None)
    if mask is not None:
        _check_shape(mask.shape, metrics)
        count = jnp.sum(mask)
        return jax.tree.map(lambda m: (jnp.sum(mask * m), count), metrics)
    else:
        return jax.tree.map(lambda m: (jnp.sum(m), jnp.prod(m.shape), metrics))

def make_eval(
    data_config: dict[str, Any],
    fn: Callable,
    *,
    mesh=None,
):

    def evaluator(model):
        #make this infinte stream or cacheable so we dont recreate laoder/stream again and again
        data = make_eval_data(data_config, mesh=mesh)
        list_aux = []
        for batch in data:
            batch_aux = _eval_step(fn, model, batch)
            list_aux.append(batch_aux)

        final_metrics = metric_utils.to_host(list_aux, flatten=False)
        final_metrics = metric_utils.add_aux(*list_aux)
        final_metrics = metric_utils.process_aux(final_metrics)
        return final_metrics

    return evaluator


def make(
    data_config: dict[str, Any],
    fn: Callable,
    *,
    mesh=None,
):
    return make_eval(data_config, fn, mesh=mesh)
