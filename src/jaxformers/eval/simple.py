from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from jaxformers import tree_util, metric_utils
from jaxformers.data.source.huggingface import make_huggingface_datasets


def _check_shape(shape, tree):
    def _f(path, leaf):
        assert leaf.shape == shape, f"metrics {path} must have shape {shape}"

    jax.tree.map(_f, tree)


@partial(jax.jit, static_argnums=0)
def _eval_step(fn, model, batch):
    metrics = fn(model, batch)
    mask = batch.get("_mask", None)
    if mask:
        _check_shape(mask.shape, metrics)
        count = jnp.sum(mask)
        return jax.tree.map(lambda m: (jnp.sum(mask * m), count), metrics)
    else:
        return jax.tree.map(lambda m: (jnp.sum(m), jnp.prod(m.shape), metrics))

def make_eval(
    load_data: dict[str, Any],
    transforms: dict[str, Any] | Callable,
    fn: Callable,
    *,
    streaming: bool = False,
):

    data = make_huggingface_datasets(load_data, streaming=streaming)

    def evaluator(model, loader):
        list_aux = []
        for batch in loader:
            batch_aux = _eval_step(fn, model, batch)
            list_aux.append(batch_aux)

        final_metrics = metric_utils.to_host(list_aux, flatten= False)
        final_metrics = metric_utils.add_aux(*list_aux)
        final_metrics = metric_utils.process_aux(final_metrics)
        return final_metrics
