from __future__ import annotations

import dataclasses
import sys
import time
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Iterable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
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
from jaxformers.checkpointing import (
    CheckpointerWithInfo,
    is_used_checkpoint,
    load_checkpoint_from_path,
    make_checkpointer,
)
from jaxformers.data import make_data
from jaxformers.dispatch.lora import make_lora
from jaxformers.distributed import set_rule
from jaxformers.eval import make_eval
from jaxformers.logger import make_logger
from jaxformers.modeling_utils import Model
from jaxformers.models import make_model
from jaxformers.optimizers import make_optimizer
from jaxformers.print_utils import tree_pprint
from jaxformers.scheduler import make_scheduler
from jaxformers.sws_utils import run as sws_run


DEFAULT_REDUCED = {"loss": "mean", "token": "sum", "batch": "sum"}
DEFAULT_AUX = {"loss": (0.0, 0), "token": 0, "batch": 0}


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
def model_context(model: Model):
    with jax.set_mesh(model.mesh):
        if hasattr(model.weights, "rule"):
            with set_rule(model.weights.rule):
                yield
        else:
            yield


def forward(weights, *args, **kwargs):
    return weights(*args, **kwargs)


def embed(weights, *args, **kwargs):
    return weights.embed(*args, **kwargs)


def unembed(weights, *args, **kwargs):
    return weights.unembed(*args, **kwargs)


def prepare_weights(weights):
    return weights


def module_to_model(name: str, weights) -> Model:
    config = weights.config if hasattr(weights, "config") else weights.model.config
    lm_head_key = (
        "model.embed_tokens.weight" if config.tie_word_embeddings else "lm_head.weight"
    )
    return Model(
        name=name,
        config=config,
        weights=weights,
        forward=forward,
        embed=embed,
        unembed=unembed,
        prepare_weights=prepare_weights,
        tokenizer=None,
        lm_head_key=lm_head_key,
        mesh=weights.mesh,
    )


def predict_fn(model: Model, batch):
    with model_context(model):
        hidden = model.forward(model.weights, **batch["inputs"])
        return model.unembed(model.weights, hidden)


def loss_fn(model, batch, rngs=None):
    del rngs
    logits = predict_fn(model, batch)
    return {
        "loss": optax.softmax_cross_entropy_with_integer_labels(logits, batch["labels"])
    }


def train_step(config: sws.FinalConfig, model: Model, batch, *, rngs):
    def loss_fn(train_weights, freeze_weights, batch, rngs):
        weights = tree_util.combine(train_weights, freeze_weights)
        logits = model.unembed(
            weights,
            model.forward(
                weights,
                **batch["inputs"],
                rngs=rngs,
                dtype=config.forward_dtype,
            ),
            dtype=config.forward_dtype,
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["labels"])
        loss = loss * batch["loss_mask"]
        count = jnp.sum(batch["loss_mask"])
        loss = jnp.sum(loss)

        batch_size = batch["labels"].shape[0]
        aux = {"loss": (loss, count), "token": count, "batch": batch_size}
        return loss, aux

    if config.grad_accum > 1:
        microbatch_size = batch["labels"].shape[0] // config.grad_accum
        grad_fn = microbatch(
            jax.value_and_grad(loss_fn, has_aux=True),
            argnums=2,
            microbatch_size=microbatch_size,
        )
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    (loss, aux), grad = grad_fn(*model.trainable_params, batch, rngs)

    token_count = aux["token"]
    inv_token_count = (1 / token_count).astype(config.forward_dtype)
    grad = jtu.tree_map(lambda g: g * inv_token_count, grad)

    updates, nst = model.tx.update(grad, model.opt_state, model.weights)
    nweights = tree_util.apply_updates(model.weights, updates, config.forward_dtype)
    next_model = dataclasses.replace(model, weights=nweights, opt_state=nst)
    callback_state = model.callback_state
    if model.callback_state is not None:
        next_model, callback_state = model.callbacks.update(
            next_model,
            model.callback_state,
            grad,
            updates,
            aux,
        )

    return dataclasses.replace(next_model, callback_state=callback_state), aux


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
    model: Model,
    train_ds: Iterable,
    evaluators: list[Callable[[Model]]] | None = None,
    logger: None = None,
    ckptr: CheckpointerWithInfo | None = None,
    *,
    rngs: PRNGKeyArray | None = None,
):
    train_iterator = iter(train_ds)
    step = model.step or 0
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
                ckptr.save_checkpoint(step, model, train_iterator)
            if not skip_eval and (step % config.eval_every) == 0:
                eval_process = {}
                for name, evaluator in evaluators.items():
                    eval_process[name] = evaluator(model)

            batch = next(train_iterator)
            step_rngs = jax.random.fold_in(rngs, step) if rngs is not None else None
            if first_step:
                with jax.named_scope("compile train step"), model_context(model):

                    @print_timing
                    def compile_train_step():
                        train_step_jit = jax.jit(
                            partial(train_step, config),
                            donate_argnums=(0,),
                        )
                        lower = train_step_jit.lower(model, batch, rngs=step_rngs)
                        return lower.compile()

                    train_step_fn = compile_train_step()
                    memory_stats = print_compiled_memory_stats(
                        train_step_fn.memory_analysis()
                    )
                    cost = print_flops(train_step_fn.cost_analysis())

                    to_log_later.update(memory_stats)
                    to_log_later.update(cost)

                    program_wall_t0 = time.monotonic()
                    model, aux = train_step_fn(model, batch, rngs=step_rngs)
                    first_step = False
            else:
                with (
                    jax.named_scope("train_step"),
                    jax.profiler.StepTraceAnnotation(f"train_step_{step}"),
                    model_context(model),
                ):
                    model, aux = train_step_fn(model, batch, rngs=step_rngs)

            host_aux = metric_utils.to_host(aux, flatten=True)
            global_aux = metric_utils.host_add_aux(
                global_aux, host_aux, reduce_method=DEFAULT_REDUCED
            )
            processed_aux = metric_utils.process_aux(host_aux, "step")
            cum_processed_aux = metric_utils.process_aux(global_aux, "cum")
            callback_output = {}
            if model.callback_state is not None:
                callback_aux = {**processed_aux, **cum_processed_aux}
                callback_output, model, callback_state = model.callbacks.process(
                    {},
                    model,
                    model.callback_state,
                    callback_aux,
                )
                model = dataclasses.replace(model, callback_state=callback_state)
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
        to_log_later["program_time"] = time.monotonic() - program_wall_t0
        to_log_later.update(metric_utils.process_aux(global_aux, "cum"))
        token_count = to_log_later.get("cum/token")
        program_time = to_log_later.get("program_time")

        if token_count is not None and program_time not in (None, 0):
            to_log_later["systems/tok_s"] = token_count / program_time

        if jax.process_index() == 0:
            logger.config.update(to_log_later)
            if "program_time" in to_log_later:
                print(f"program_time: {to_log_later['program_time']:.3f}s")
            if "systems/tok_s" in to_log_later:
                print(f"tok/s: {to_log_later['systems/tok_s']:.2f}")
        if pbar is not None:
            pbar.close()
        if ckptr is not None:
            ckptr.close()

    return model, to_log_later


def main(config: sws.FinalConfig):
    _preparse_absl_flags()
    do_callback = getattr(config, "callback_name", None)
    do_load_model = getattr(config, "load_model", None)
    do_checkpoint = getattr(config, "checkpoint", None)
    do_lora = getattr(config, "lora", None)
    do_eval = getattr(config, "eval", None)

    logger = None
    if jax.process_index() == 0:
        logger = make_logger(config.logger_name, config.logger.to_dict())
        logger.config.update(config.to_dict())
    try:
        rngs = jax.random.key(config.seed) if config.seed else None
        model_rngs, lora_rngs, train_rngs = (
            jax.random.split(rngs, 3) if rngs is not None else (None, None, None)
        )
        model = make_model(
            config.model_name,
            config.init_model,
            config.model.to_dict(),
            rngs=model_rngs,
        )
        if not isinstance(model, Model):
            model = module_to_model(config.model_name, model)

        with model_context(model):
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
                model = make_lora(
                    model, config.init_lora, config.lora.to_dict(), rngs=lora_rngs
                )
            model = dataclasses.replace(
                model,
                weights=model.weights.stack() if config.model.additional_config.weights_impl == "stack" else model.weights
            )

            tree_pprint(model.weights)

            model = make_optimizer(
                config.optimizer_name,
                model,
                scheduler,
                config.optimizer.to_dict(),
            )
            print_train_state_size(model)

            if do_callback:
                callbacks = make_callbacks(config.callback_name, config.callback.to_dict())
                model = dataclasses.replace(
                    model,
                    callback_state=callbacks.init(model),
                    callbacks=callbacks,
                )
            if do_load_model:
                if do_checkpoint and is_used_checkpoint(config.checkpoint.path):
                    raise ValueError(
                        f"Both 'load_model' and 'checkpoint' are active, but {config.checkpoint.path} is a non-empty checkpoint. "
                        f"To resume training from {config.checkpoint.path}, set load_model=None. "
                        f"To start a new run while loading the weights and opt_state from the old checkpoint, make sure the new checkpoint.path points to an empty (fresh) checkpoint folder."
                    )
                model = load_checkpoint_from_path(model, **config.load_model.to_dict())

            ckptr = None
            if do_checkpoint:
                ckptr = make_checkpointer(model, **config.checkpoint.to_dict())
                model = ckptr.load_checkpoint(model)

            train_ds = make_data(config.data.to_dict(), mesh=model.mesh)

            evaluators = None
            if do_eval:
                evaluators = make_eval(
                    config.eval.to_dict(), {"predict": predict_fn, "loss": loss_fn}
                )

        _, metrics = train(
            config,
            model,
            train_ds,
            evaluators,
            logger,
            ckptr,
            rngs=train_rngs,
        )
        return metrics
    finally:
        if logger is not None:
            logger.finish()


if __name__ == "__main__":
    sws_run(main)
