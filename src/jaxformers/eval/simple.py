from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from tqdm import tqdm

from jaxformers import metric_utils
from jaxformers.data import make_eval_data
from jaxformers.sharding_utils import with_logical_axis


def _check_shape(shape, tree):
    def _f(path, leaf):
        assert leaf.shape == shape, f"metrics {path} must have shape {shape}"

    jax.tree.map_with_path(_f, tree)


@partial(jax.jit, static_argnums=0)
def _eval_step(fn, model, batch):
    mask = None
    metrics = fn(model, batch)
    _mask = batch.get("_mask", None)
    loss_mask = batch.get("loss_mask")
    mask = _mask[:, None] & loss_mask  # (B, ) #(B, T )
    if mask is not None:
        _check_shape(mask.shape, metrics)
        count = jnp.sum(mask)
        return jax.tree.map(lambda m: (jnp.sum(mask * m), count), metrics)
    else:
        return jax.tree.map(lambda m: (jnp.sum(m), jnp.prod(m.shape), metrics))


def make(
    name: str,
    data_config: dict[str, Any],
    fn: Callable,
    *,
    mesh=None,
):
    def evaluator(model):
        # make this infinte stream or cacheable so we dont recreate laoder/stream again and again
        list_aux = []
        with jax.set_mesh(model.mesh), with_logical_axis(model.rule):
            data = make_eval_data(data_config, mesh=model.mesh)
            for batch in tqdm(data, desc=f"{name}_eval"):
                batch_aux = _eval_step(fn, model, batch)
                list_aux.append(batch_aux)
            print(batch["_mask"])

        final_metrics = metric_utils.to_host(list_aux, flatten=False)
        final_metrics = metric_utils.host_add_aux(*list_aux)
        final_metrics = metric_utils.process_aux(final_metrics)
        return final_metrics

    return evaluator
