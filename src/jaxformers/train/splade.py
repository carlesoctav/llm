from __future__ import annotations

import dataclasses
import sys
import time
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import sws
from jax.experimental.rnn import PRNGKeyArray
from optax import microbatch
from tqdm.auto import tqdm

from jaxformers import metric_utils, tree_util
from jaxformers.benchmark_utils import (
    print_compiled_memory_stats,
    print_flops,
    print_timing,
    print_train_state_size,
)
from jaxformers.callbacks import make_callbacks
from jaxformers.checkpoint_utils import (
    CheckpointerWithInfo,
    is_used_checkpoint,
    load_checkpoint_from_path,
    make_checkpointer,
)
from jaxformers.data import make_data
from jaxformers.dispatch.lora import make_lora
from jaxformers.logger import make_logger
from jaxformers.modeling_utils import TrainState
from jaxformers.models import make_model
from jaxformers.optimizers import make_optimizer
from jaxformers.scheduler import make_scheduler
from jaxformers.sharding_utils import (
    make_logical_axis_rules,
    make_mesh,
    with_logical_axis,
)
from jaxformers.sws_utils import run as sws_run


DEFAULT_REDUCED = {"loss": "mean", "nll": "mean", "flops": "mean", "batch": "sum"}
DEFAULT_AUX = {"loss": (0.0, 0), "nll": (0.0, 0), "flops": (0.0, 0), "batch": 0}


def _preparse_absl_flags() -> None:
    try:
        from absl import flags
    except Exception:
        return

    if flags.FLAGS.is_parsed():
        return

    try:
        import tokamax._src.config as _tokamax_config  # noqa: F401
    except Exception:
        pass

    flags.FLAGS(sys.argv, known_only=True)


@contextmanager
def train_state_context(train_state: TrainState):
    with jax.set_mesh(train_state.mesh), with_logical_axis(train_state.rule):
        yield


def train_step(config: sws.FinalConfig, train_state: TrainState, batch, *, rngs):
    def loss_fn(train_model, freeze_model, batch, rngs):
        model = tree_util.combine(train_model, freeze_model)
        queries = model(
            **batch["query"],
            rngs=rngs,
            dtype=config.forward_dtype,
        )
        num_docs = batch["docs"]["input_ids"].shape[1]
        flat_docs = jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), batch["docs"])
        docs = model(
            **flat_docs,
            rngs=rngs,
            dtype=config.forward_dtype,
        )
        scores = model.score(queries, docs).astype(jnp.float32)
        scores = scores / config.temperature
        batch_size = queries.shape[0]
        labels = jnp.arange(batch_size) * num_docs
        nll = -jax.nn.log_softmax(scores)[jnp.arange(batch_size), labels].mean()
        flops = (
            config.flops_query_weight
            * jnp.square(queries.mean(axis=0).astype(jnp.float32)).sum()
            + config.flops_doc_weight
            * jnp.square(docs.mean(axis=0).astype(jnp.float32)).sum()
        )
        loss = nll + flops
        aux = {
            "loss": (loss, 1),
            "nll": (nll, 1),
            "flops": (flops, 1),
            "batch": batch_size,
        }
        return loss, aux

    if config.grad_accum > 1:
        microbatch_size = batch["query"]["input_ids"].shape[0] // config.grad_accum
        grad_fn = microbatch(
            jax.value_and_grad(loss_fn, has_aux=True),
            argnums=2,
            microbatch_size=microbatch_size,
        )
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    (loss, aux), grad = grad_fn(*train_state.trainable_params, batch, rngs)

    updates, nst = train_state.tx.update(grad, train_state.opt_state, train_state.model)
    nmodel = tree_util.apply_updates(train_state.model, updates)
    ntrain_state = dataclasses.replace(train_state, model=nmodel, opt_state=nst)
    callback_state = train_state.callback_state
    if train_state.callback_state is not None:
        ntrain_state, callback_state = train_state.callbacks.update(
            ntrain_state,
            train_state.callback_state,
            grad,
            updates,
            aux,
        )

    return dataclasses.replace(ntrain_state, callback_state=callback_state), aux


def pbar_display(metrics: dict[str, Any]) -> dict[str, Any]:
    def to_py(value: Any) -> Any:
        value = jax.device_get(value)
        if isinstance(value, np.ndarray) and value.shape == ():
            return value.item()
        if isinstance(value, np.generic):
            return value.item()
        return value

    return {k: to_py(v) for k, v in metrics.items()}


def train(
    config,
    train_state: TrainState,
    train_iterator: Iterator,
    evaluators: list[Callable[[TrainState]]] | None = None,
    logger: None = None,
    ckptr: CheckpointerWithInfo | None = None,
    *,
    rngs: PRNGKeyArray | None = None,
):
    step = train_state.step or 0
    global_aux = dict(DEFAULT_AUX)
    skip_eval = config.eval_every is None or evaluators is None
    first_step = True

    to_log_later = {}
    program_wall_t0 = None

    pbar = None
    if jax.process_index() == 0:
        pbar = tqdm(
            total=config.max_train_step,
            initial=int(step),
            desc="train",
            unit="step",
            dynamic_ncols=True,
        )

    try:
        while step < config.max_train_step:
            if ckptr is not None:
                ckptr.save_checkpoint(step, train_state, train_iterator)
            if not skip_eval and (step % config.eval_every) == 0:
                eval_process = {}
                for name, evaluator in evaluators.items():
                    eval_process[name] = evaluator(train_state)

            step_rngs = jax.random.fold_in(rngs, step) if rngs is not None else None
            with (
                jax.named_scope("train_step"),
                jax.profiler.StepTraceAnnotation("train_step", step_num=step),
                train_state_context(train_state),
            ):
                batch = next(train_iterator)
                if first_step:

                    @print_timing
                    def compile_train_step():
                        train_step_jit = jax.jit(
                            partial(train_step, config),
                            donate_argnums=(0,),
                        )
                        lower = train_step_jit.lower(train_state, batch, rngs=step_rngs)
                        return lower.compile()

                    train_step_fn = compile_train_step()
                    compiled_memory_stats = train_step_fn.memory_analysis()
                    memory_stats = print_compiled_memory_stats(compiled_memory_stats)
                    cost = print_flops(train_step_fn.cost_analysis())

                    to_log_later.update(memory_stats)
                    to_log_later.update(cost)

                    program_wall_t0 = time.monotonic()
                    train_state, aux = train_step_fn(train_state, batch, rngs=step_rngs)
                    first_step = False
                else:
                    train_state, aux = train_step_fn(train_state, batch, rngs=step_rngs)

            host_aux = metric_utils.to_host(aux, flatten=True)
            global_aux = metric_utils.host_add_aux(
                global_aux, host_aux, reduce_method=DEFAULT_REDUCED
            )
            processed_aux = metric_utils.process_aux(host_aux, "step")
            cum_processed_aux = metric_utils.process_aux(global_aux, "cum")
            callback_output = {}
            if train_state.callback_state is not None:
                callback_aux = {**processed_aux, **cum_processed_aux}
                callback_output, train_state, callback_state = (
                    train_state.callbacks.process(
                        {},
                        train_state,
                        train_state.callback_state,
                        callback_aux,
                    )
                )
                train_state = dataclasses.replace(
                    train_state, callback_state=callback_state
                )
            if jax.process_index() == 0:
                logger.log(processed_aux, step=step)
                logger.log(cum_processed_aux, step=step)
                logger.log(callback_output, step=step)
                if pbar is not None:
                    pbar.set_postfix(
                        pbar_display(
                            {**cum_processed_aux, **processed_aux, **callback_output}
                        )
                    )
                    pbar.update(1)
            step += 1

    finally:
        if program_wall_t0 is not None:
            to_log_later["program_time"] = time.monotonic() - program_wall_t0
        to_log_later.update(metric_utils.process_aux(global_aux, "cum"))
        query_count = to_log_later.get("cum/batch")
        program_time = to_log_later.get("program_time")

        if query_count is not None and program_time not in (None, 0):
            to_log_later["systems/query_s"] = query_count / program_time

        if jax.process_index() == 0:
            logger.config.update(to_log_later)
            if "program_time" in to_log_later:
                print(f"program_time: {to_log_later['program_time']:.3f}s")
            if "systems/query_s" in to_log_later:
                print(f"query/s: {to_log_later['systems/query_s']:.2f}")
        if pbar is not None:
            pbar.close()
        if ckptr is not None:
            ckptr.close()

    return train_state, to_log_later


def main(config: sws.FinalConfig):
    jax.config.update("jax_softmax_custom_jvp", True)
    _preparse_absl_flags()
    do_callback = getattr(config, "callback", None)
    do_load_state = getattr(config, "load_state", None)
    do_checkpoint = getattr(config, "checkpoint", None)
    do_lora = getattr(config, "lora", None)
    train_iterator = None

    logger = None
    if jax.process_index() == 0:
        logger = make_logger(config.logger_name, config.logger.to_dict())
        logger.config.update(config.to_dict())
    try:
        rngs = jax.random.key(config.seed)
        model_rngs, lora_rngs, train_rngs = jax.random.split(rngs, 3)
        rule = make_logical_axis_rules(**config.parallel.to_dict())
        mesh = make_mesh(**config.parallel.to_dict())
        with jax.set_mesh(mesh), with_logical_axis(rule):
            model = make_model(
                config.model_name,
                **config.model.to_dict(),
                rngs=model_rngs,
            )
        train_state = TrainState(model, mesh, rule=rule)

        with train_state_context(train_state):
            scheduler_config = (
                config.lr_scheduler.to_dict() if "lr_scheduler" in config else {}
            )
            scheduler = make_scheduler(
                config.lr_scheduler_name,
                config.learning_rate,
                config.max_train_step,
                scheduler_config=scheduler_config,
            )
            if do_lora:
                train_state = make_lora(
                    train_state,
                    config.init_lora,
                    config.lora.to_dict(),
                    rngs=lora_rngs,
                )
            train_state = dataclasses.replace(
                train_state,
                model=(
                    train_state.model.stack()
                    if config.weights_impl == "stack"
                    else train_state.model
                ),
            )
            train_state = dataclasses.replace(
                train_state,
                train_mask=jax.tree.map(
                    lambda leaf: jnp.issubdtype(leaf.dtype, jnp.floating),
                    train_state.model,
                ),
            )

            train_state = make_optimizer(
                config.optimizer_name,
                train_state,
                scheduler,
                config.optimizer.to_dict(),
            )
            print_train_state_size(train_state)

            if do_callback:
                callbacks = make_callbacks(config.callback.to_dict())
                train_state = dataclasses.replace(
                    train_state,
                    callback_state=callbacks.init(train_state),
                    callbacks=callbacks,
                )
            if do_load_state:
                if do_checkpoint and is_used_checkpoint(config.checkpoint.path):
                    raise ValueError(
                        f"Both 'load_state' and 'checkpoint' are active, but {config.checkpoint.path} is a non-empty checkpoint. "
                        f"To resume training from {config.checkpoint.path}, set load_state=None. "
                        f"To start a new run while loading the weights and opt_state from the old checkpoint, make sure the new checkpoint.path points to an empty (fresh) checkpoint folder."
                    )
                train_state = load_checkpoint_from_path(
                    train_state,
                    **config.load_state.to_dict(),
                )

            ckptr = None
            if do_checkpoint:
                ckptr = make_checkpointer(train_state, **config.checkpoint.to_dict())
                train_state = ckptr.load_checkpoint(train_state)

            train_ds = make_data(config.data.to_dict(), mesh=train_state.mesh)

        train_iterator = iter(train_ds)
        _, metrics = train(
            config,
            train_state,
            train_iterator,
            None,
            logger,
            ckptr,
            rngs=train_rngs,
        )
        return metrics
    finally:
        if train_iterator is not None:
            train_iterator.close()
        if logger is not None:
            logger.finish()


if __name__ == "__main__":
    sws_run(main)
