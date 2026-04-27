from __future__ import annotations

import asyncio
import dataclasses
import sys
import time
from contextlib import contextmanager
from functools import partial
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
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
from jaxformers.inference import make as make_llm_client
from jaxformers.logger import make_logger
from jaxformers.modeling_utils import TrainState
from jaxformers.models import make_model
from jaxformers.optimizers import make_optimizer
from jaxformers.rl import make_rl_data
from jaxformers.scheduler import make_scheduler
from jaxformers.sharding_utils import (
    make_logical_axis_rules,
    make_mesh,
    with_logical_axis,
)
from jaxformers.sws_utils import run as sws_run


DEFAULT_REDUCED = {
    "loss": "mean",
    "token": "sum",
    "batch": "sum",
    "reward": "mean",
    "clipfrac": "mean",
    "approx_kl": "mean",
}
DEFAULT_AUX = {
    "loss": (0.0, 0),
    "token": 0,
    "batch": 0,
    "reward": (0.0, 0),
    "clipfrac": (0.0, 0),
    "approx_kl": (0.0, 0),
}


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


def pbar_display(metrics: dict[str, Any]) -> dict[str, Any]:
    def to_py(value: Any) -> Any:
        value = jax.device_get(value)
        if isinstance(value, np.ndarray) and value.shape == ():
            return value.item()
        if isinstance(value, np.generic):
            return value.item()
        return value

    return {k: to_py(v) for k, v in metrics.items()}


def train_step(config: sws.FinalConfig, train_state: TrainState, batch, *, rngs):
    def loss_fn(train_model, freeze_model, batch, rngs):
        model = tree_util.combine(train_model, freeze_model)
        hidden_states, _ = model(
            **batch["inputs"],
            rngs=rngs,
            dtype=config.forward_dtype,
            return_hidden_states=True,
        )
        logits = model.unembed(hidden_states)
        logprobs = jax.nn.log_softmax(logits, axis=-1)
        selected_logprobs = jnp.take_along_axis(
            logprobs,
            batch["labels"][..., None],
            axis=-1,
        ).squeeze(-1)

        loss_mask = batch["loss_mask"]
        behavior_logprobs = batch["behavior_logprobs"]
        advantages = batch["advantages"]

        log_ratio = selected_logprobs - behavior_logprobs
        ratio = jnp.exp(log_ratio)
        clipped_ratio = jnp.clip(
            ratio,
            1.0 - config.clip_epsilon,
            1.0 + config.clip_epsilon,
        )
        objective = jnp.minimum(ratio * advantages, clipped_ratio * advantages)
        loss = -(objective * loss_mask).sum()

        count = jnp.sum(loss_mask)
        batch_size = batch["labels"].shape[0]
        approx_kl = (jnp.square(log_ratio) * loss_mask).sum()
        clipfrac = ((ratio != clipped_ratio).astype(jnp.float32) * loss_mask).sum()
        aux = {
            "loss": (loss, count),
            "token": count,
            "batch": batch_size,
            "reward": (batch["reward"].sum(), batch_size),
            "clipfrac": (clipfrac, count),
            "approx_kl": (approx_kl, count),
        }
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

    (loss, aux), grad = grad_fn(*train_state.trainable_params, batch, rngs)
    del loss

    token_count = aux["token"]
    inv_token_count = (1 / token_count).astype(config.forward_dtype)
    grad = jtu.tree_map(lambda g: g * inv_token_count, grad)

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


def sync_inference_weights(train_state: TrainState, llm_client) -> None:
    llm_client.update_weights(train_state.model)


def train(
    config,
    train_state: TrainState,
    train_ds: Iterable,
    logger,
    llm_client,
    *,
    rngs: PRNGKeyArray | None = None,
):
    train_iterator = iter(train_ds)
    step = train_state.step or 0
    global_aux = dict(DEFAULT_AUX)
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
            batch = next(train_iterator)
            step_rngs = jax.random.fold_in(rngs, step) if rngs is not None else None
            if first_step:
                with (
                    jax.named_scope("compile train step"),
                    train_state_context(train_state),
                    llm_client.runtime_lock(),
                ):

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
                    # sync_inference_weights(train_state, llm_client)
                    first_step = False
            else:
                with (
                    jax.named_scope("train_step"),
                    jax.profiler.StepTraceAnnotation(f"train_step_{step}"),
                    train_state_context(train_state),
                    llm_client.runtime_lock(),
                ):
                    train_state, aux = train_step_fn(train_state, batch, rngs=step_rngs)
                    # sync_inference_weights(train_state, llm_client)

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
        token_count = to_log_later.get("cum/token")
        program_time = to_log_later.get("program_time")
        if token_count is not None and program_time not in (None, 0):
            to_log_later["systems/tok_s"] = token_count / program_time

        if jax.process_index() == 0:
            logger.config.update(to_log_later)
        if pbar is not None:
            pbar.close()

    return train_state, to_log_later


def main(config: sws.FinalConfig):
    _preparse_absl_flags()
    if jax.process_count() != 1:
        raise ValueError("train/grpo.py currently supports only single-process runs.")

    do_callback = getattr(config, "callback", None)
    do_lora = getattr(config, "lora", None)

    logger = None
    llm_client = None
    if jax.process_index() == 0:
        logger = make_logger(config.logger_name, config.logger.to_dict())
        logger.config.update(config.to_dict())
    try:
        rngs = jax.random.key(config.seed)
        model_rngs, train_rngs = jax.random.split(rngs)
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
                raise NotImplementedError(
                    "train/grpo.py same-process TPU sync does not support LoRA."
                )
            train_state = dataclasses.replace(
                train_state,
                model=(
                    train_state.model.stack()
                    if config.weights_impl == "stack"
                    else train_state.model
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

        vllm_config = config.vllm.to_dict()
        if "additional_config" in vllm_config:
            additional_config = dict(vllm_config["additional_config"])
        else:
            additional_config = {}
        if "sharding" in additional_config:
            sharding = dict(additional_config["sharding"])
        else:
            sharding = {}
        if "sharding_strategy" in sharding:
            sharding_strategy = dict(sharding["sharding_strategy"])
        else:
            sharding_strategy = {}
        sharding_strategy["device_indexes"] = mesh.device_ids.flatten().tolist()
        sharding["sharding_strategy"] = sharding_strategy
        additional_config["sharding"] = sharding
        vllm_config["additional_config"] = additional_config

        llm_client = make_llm_client(
            config.inference.mode,
            model=config.model.model_id,
            tokenizer=config.model.model_id,
            vllm_config=vllm_config,
        )
        # sync_inference_weights(train_state, llm_client)
        train_ds = make_rl_data(config, llm_client)
        _, metrics = train(
            config,
            train_state,
            train_ds,
            logger,
            llm_client,
            rngs=train_rngs,
        )
        return metrics
    finally:
        if llm_client is not None:
            asyncio.run(llm_client.close())
        if logger is not None:
            logger.finish()

if __name__ == "__main__":

    sws_run(main)
