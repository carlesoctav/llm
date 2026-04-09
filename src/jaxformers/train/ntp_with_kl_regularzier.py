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
from jaxtyping import PRNGKeyArray
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
from jaxformers.eval import make_eval
from jaxformers.logger import make_logger
from jaxformers.modeling_utils import TrainState
from jaxformers.models import make_model
from jaxformers.ops.cross_entropy.api import cross_entropy_loss
from jaxformers.optimizers import make_optimizer
from jaxformers.scheduler import make_scheduler
from jaxformers.sharding_utils import (
    from_logical_rules,
    make_logical_axis_rules,
    make_mesh,
    with_logical_axis,
)
from jaxformers.sws_utils import run as sws_run


DEFAULT_REDUCED = {
    "loss": "mean",
    "token": "sum",
    "batch": "sum",
    "sft.loss": "mean",
    "sft.token": "sum",
    "sft.batch": "sum",
    "kl.loss": "mean",
    "kl.token": "sum",
    "kl.batch": "sum",
}
DEFAULT_AUX = {
    "loss": (0.0, 0),
    "token": 0,
    "batch": 0,
    "sft.loss": (0.0, 0),
    "sft.token": 0,
    "sft.batch": 0,
    "kl.loss": (0.0, 0),
    "kl.token": 0,
    "kl.batch": 0,
}


@contextmanager
def train_state_context(train_state: TrainState):
    with jax.set_mesh(train_state.mesh), with_logical_axis(train_state.rule):
        yield


def predict_fn(train_state: TrainState, batch, *, forward_dtype):
    with train_state_context(train_state):
        hidden_states, _ = train_state.model(
            **batch["inputs"],
            dtype=forward_dtype,
            return_hidden_states=True,
        )
        return train_state.model.unembed(hidden_states)


def loss_fn(train_state: TrainState, batch, rngs=None, *, forward_dtype, loss_impl):
    del rngs
    with train_state_context(train_state):
        hidden_states, _ = train_state.model(
            **batch["inputs"],
            dtype=forward_dtype,
            return_hidden_states=True,
        )
        batch_shape = batch["labels"].shape
        loss = cross_entropy_loss(
            hidden_states.reshape(-1, hidden_states.shape[-1]),
            batch["labels"].reshape(-1),
            train_state.model.lm_head_w,
            reduction=None,
            implementation=loss_impl,
        )
        return {"loss": loss.reshape(batch_shape)}


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


def train_step(
    config: sws.FinalConfig,
    train_state: TrainState,
    base_train_state: TrainState | None,
    batch,
    *,
    rngs,
):
    sft_batch, kl_batch = batch

    def kl_loss_fn(train_model, freeze_model, kl_batch, rngs):
        model = tree_util.combine(train_model, freeze_model)
        forward_dtype = config.forward_dtype
        count = jnp.sum(kl_batch["loss_mask"])
        mask = kl_batch["loss_mask"]

        hidden_states, _ = model(
            **kl_batch["inputs"],
            rngs=rngs,
            dtype=forward_dtype,
            return_hidden_states=True,
        )
        log_probs = jax.nn.log_softmax(model.unembed(hidden_states), axis=-1)

        if train_state.is_lora:
            base_model = train_state.base_params
            base_hidden_states, _ = base_model(
                **kl_batch["inputs"],
                rngs=rngs,
                dtype=forward_dtype,
                return_hidden_states=True,
            )
            kl_log_probs = jax.nn.log_softmax(
                base_model.unembed(base_hidden_states),
                axis=-1,
            )
        else:
            base_model = base_train_state.model
            base_hidden_states, _ = base_model(
                **kl_batch["inputs"],
                rngs=rngs,
                dtype=forward_dtype,
                return_hidden_states=True,
            )
            kl_log_probs = jax.nn.log_softmax(
                base_model.unembed(base_hidden_states),
                axis=-1,
            )

        kl_loss = (
            optax.losses.kl_divergence_with_log_targets(log_probs, kl_log_probs) * mask
        ).sum()

        return kl_loss, {
            "loss": (kl_loss, count),
            "token": count,
            "batch": kl_batch["inputs"]["input_ids"].shape[0],
        }

    def sft_loss_fn(train_model, freeze_model, sft_batch, rngs):
        model = tree_util.combine(train_model, freeze_model)
        forward_dtype = config.forward_dtype

        hidden_states, _ = model(
            **sft_batch["inputs"],
            rngs=rngs,
            dtype=forward_dtype,
            return_hidden_states=True,
        )

        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        labels = sft_batch["labels"].reshape(-1)
        count = jnp.sum(sft_batch["loss_mask"])
        mask = sft_batch["loss_mask"].reshape(-1)
        loss = cross_entropy_loss(
            hidden_states,
            labels,
            (
                # The fused XLA chunked CE kernel currently assumes replicated
                # vocab weights; avoid forcing replication for the reference
                # implementation to match Tunix's fsdp-sharded embed/lm_head.
                jax.reshard(
                    model.lm_head_w,
                    from_logical_rules(("none", "none")),
                )
                if (config.loss_impl or None) != "reference"
                else model.lm_head_w
            ),
            reduction="sum",
            mask=mask,
            implementation=config.loss_impl or None,
        )

        batch_size = sft_batch["labels"].shape[0]
        aux = {"loss": (loss, count), "token": count, "batch": batch_size}
        return loss, aux

    train_model, freeze_model = tree_util.partition(
        train_state.model,
        train_state.train_mask,
    )

    if config.grad_accum > 1:
        sft_microbatch_size = sft_batch["labels"].shape[0] // config.grad_accum
        kl_microbatch_size = (
            kl_batch["inputs"]["input_ids"].shape[0] // config.grad_accum
        )
        sft_grad_fn = microbatch(
            jax.value_and_grad(sft_loss_fn, has_aux=True),
            argnums=2,
            microbatch_size=sft_microbatch_size,
        )

        kl_grad_fn = microbatch(
            jax.value_and_grad(kl_loss_fn, has_aux=True),
            argnums=2,
            microbatch_size=kl_microbatch_size,
        )
    else:
        sft_grad_fn = jax.value_and_grad(sft_loss_fn, has_aux=True)
        kl_grad_fn = jax.value_and_grad(kl_loss_fn, has_aux=True)

    (_, sft_aux), sft_grad = sft_grad_fn(
        train_model,
        freeze_model,
        sft_batch,
        rngs,
    )
    (_, kl_aux), kl_grad = kl_grad_fn(
        train_model,
        freeze_model,
        kl_batch,
        rngs,
    )

    sft_token_count = sft_aux["token"]
    kl_token_count = kl_aux["token"]

    sft_inv_token_count = (1 / sft_token_count).astype(config.forward_dtype)
    kl_inv_token_count = (1 / kl_token_count).astype(config.forward_dtype)
    sft_grad = jtu.tree_map(lambda g: g * sft_inv_token_count, sft_grad)
    kl_grad = jtu.tree_map(lambda g: g * kl_inv_token_count, kl_grad)

    grad = jtu.tree_map(
        lambda g1, g2: config.loss_ratio.sft_loss * g1 + config.loss_ratio.kl_loss * g2,
        sft_grad,
        kl_grad,
    )

    updates, nst = train_state.tx.update(
        grad,
        train_state.opt_state,
        train_state.model,
    )
    nmodel = tree_util.apply_updates(train_state.model, updates)
    total_aux = metric_utils.jittable_add_aux(
        sft_aux,
        kl_aux,
        reduce_method=DEFAULT_REDUCED,
    )
    aux = {"sft": sft_aux, "kl": kl_aux, **total_aux}

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
    base_train_state: TrainState | None,
    train_ds: Iterable,
    evaluators: dict[str, Callable[[TrainState], Any]] | None = None,
    logger=None,
    ckptr: CheckpointerWithInfo | None = None,
    *,
    rngs: PRNGKeyArray | None = None,
):
    train_iterator = iter(train_ds)
    step = train_state.step or 0
    global_aux = dict(DEFAULT_AUX)
    skip_eval = config.eval_every is None or evaluators is None
    first_step = True

    to_log_later = {}
    program_wall_t0 = None

    pbar = None
    if jax.process_index() == 0:
        total = config.max_train_step
        pbar = tqdm(
            total=total,
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

            batch = next(train_iterator)
            step_rngs = jax.random.fold_in(rngs, step) if rngs is not None else None
            if first_step:
                with (
                    jax.named_scope("compile train step"),
                    train_state_context(train_state),
                ):

                    @print_timing
                    def compile_train_step():
                        train_step_jit = jax.jit(
                            partial(train_step, config),
                            donate_argnums=(0,),
                        )
                        lower = train_step_jit.lower(
                            train_state,
                            base_train_state,
                            batch,
                            rngs=step_rngs,
                        )
                        train_step_fn = lower.compile()
                        return train_step_fn

                    train_step_fn = compile_train_step()
                    memory_stats = print_compiled_memory_stats(
                        train_step_fn.memory_analysis()
                    )
                    cost = print_flops(train_step_fn.cost_analysis())

                    to_log_later.update(memory_stats)
                    to_log_later.update(cost)

                    program_wall_t0 = time.monotonic()
                    train_state, aux = train_step_fn(
                        train_state,
                        base_train_state,
                        batch,
                        rngs=step_rngs,
                    )
                    first_step = False
            else:
                with (
                    jax.named_scope("train_step"),
                    jax.profiler.StepTraceAnnotation(f"train_step_{step}"),
                    train_state_context(train_state),
                ):
                    train_state, aux = train_step_fn(
                        train_state,
                        base_train_state,
                        batch,
                        rngs=step_rngs,
                    )

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
                    train_state,
                    callback_state=callback_state,
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
            if "program_time" in to_log_later:
                print(f"program_time: {to_log_later['program_time']:.3f}s")
            if "systems/tok_s" in to_log_later:
                print(f"tok/s: {to_log_later['systems/tok_s']:.2f}")
        if pbar is not None:
            pbar.close()
        if ckptr is not None:
            ckptr.close()

    return train_state, to_log_later


def main(config: sws.FinalConfig):
    _preparse_absl_flags()
    do_callback = getattr(config, "callback", None)
    do_load_state = getattr(config, "load_state", None)
    do_checkpoint = getattr(config, "checkpoint", None)
    do_lora = getattr(config, "lora", None)
    do_eval = getattr(config, "eval", None)

    logger = None
    if jax.process_index() == 0:
        logger = make_logger(config.logger_name, config.logger.to_dict())
        logger.config.update(config.to_dict())
    try:
        rngs = jax.random.key(config.seed) if config.seed is not None else None
        model_rngs, lora_rngs, train_rngs = (
            jax.random.split(rngs, 3) if rngs is not None else (None, None, None)
        )
        rule = make_logical_axis_rules(**config.parallel.to_dict())
        mesh = make_mesh(**config.parallel.to_dict())
        with jax.set_mesh(mesh), with_logical_axis(rule):
            model = make_model(
                config.model_name,
                **config.model.to_dict(),
                rngs=model_rngs,
            )
        train_state = TrainState(model, mesh, rule=rule)
        base_train_state = dataclasses.replace(train_state)

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
            if config.weights_impl == "stack":
                model = train_state.model.stack()
            elif config.weights_impl == "stack_block":
                model = train_state.model.stack_block()
            else:
                model = train_state.model
            train_state = dataclasses.replace(
                train_state,
                model=model,
            )
            train_state = make_optimizer(
                config.optimizer_name,
                train_state,
                scheduler,
                config.optimizer.to_dict(),
            )
            print_train_state_size(train_state)

            if do_lora:
                base_train_state = None
            else:
                if config.weights_impl == "stack":
                    base_model = base_train_state.model.stack()
                elif config.weights_impl == "stack_block":
                    base_model = base_train_state.model.stack_block()
                else:
                    base_model = base_train_state.model
                base_train_state = dataclasses.replace(
                    base_train_state,
                    model=base_model,
                )
                base_train_state = tree_util.copy(base_train_state)

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
                ckptr = make_checkpointer(
                    train_state,
                    **config.checkpoint.to_dict(),
                )
                train_state = ckptr.load_checkpoint(train_state)

            train_ds = make_data(
                config.data.to_dict(),
                mesh=train_state.mesh,
            )

            evaluators = None
            if do_eval:
                evaluators = make_eval(
                    config.eval.to_dict(),
                    {
                        "predict": partial(
                            predict_fn,
                            forward_dtype=config.forward_dtype,
                        ),
                        "loss": partial(
                            loss_fn,
                            forward_dtype=config.forward_dtype,
                            loss_impl=config.loss_impl,
                        ),
                    },
                )

        _, metrics = train(
            config,
            train_state,
            base_train_state,
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
