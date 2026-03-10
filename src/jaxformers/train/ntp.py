import dataclasses
import importlib
import sys
import time
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import orbax.checkpoint as ocp
import sws
from jax.experimental.rnn import PRNGKeyArray
from optax import microbatch
from tqdm.auto import tqdm

from jaxformers import tree_util
from jaxformers.benchmark_utils import print_compiled_memory_stats, print_flops
from jaxformers.callbacks.base import callback_chain
from jaxformers.data.next_token_prediction import transforms as ntp_transforms
from jaxformers.data.training import make_dataloader
from jaxformers.dispatch.lora import loraify
from jaxformers.logger import load as load_logger
from jaxformers.modeling_utils import logical_to_physical, Model
from jaxformers.models import make_model
from jaxformers.ops.cross_entropy.api import cross_entropy_loss
from jaxformers.optimizer_utils import (
    find_grad_norm,
    find_learning_rate,
    mask_trainable_lora,
)


DEFAULT_REDUCED = {"loss": "mean", "token": "sum"}
DEFAULT_AUX = {"loss": (0, 0), "token": 0}


def _preparse_absl_flags() -> None:
    """Avoid absl.flags crashing on this script's CLI args.

    Some dependencies (e.g. `tokamax`) lazily call `absl.flags.FLAGS(sys.argv)`,
    which raises `UnrecognizedFlagError` when our program is launched with
    non-absl flags like `--config` (used by `sws`).

    We pre-parse once with `known_only=True` so absl marks flags as parsed while
    ignoring unknown args.
    """

    try:
        from absl import flags
    except Exception:
        return

    if flags.FLAGS.is_parsed():
        return

    # Ensure tokamax' absl flags are registered before parsing, if available.
    try:
        import tokamax._src.config as _tokamax_config  # noqa: F401
    except Exception:
        pass

    flags.FLAGS(sys.argv, known_only=True)


def load_optimizer(config: sws.FinalConfig, model, opt_name, scheduler):
    optmizer_module = importlib.import_module(f"jaxformers.optimizers.{opt_name}")
    train_mask = None
    if getattr(model, "is_lora", False):
        train_mask = mask_trainable_lora(model.weights)

    train_weights, _ = tree_util.partition(model.weights, train_mask)
    optimizer_kwargs = config.optimizer.to_dict()
    tx = optmizer_module.make(scheduler, **optimizer_kwargs)
    opt_state = tx.init(train_weights)
    return dataclasses.replace(model, opt_state=opt_state, tx=tx, train_mask=train_mask)


def load_scheduler(config: sws.FinalConfig, sched_name):
    if sched_name is None:
        return config.learning_rate
    raise NotImplementedError


def load_dataset(config: sws.FinalConfig, data_name: str):
    dataset_module = importlib.import_module(f"jaxformers.data.{data_name}")

    train_dataset = dataset_module.load(config.data.load_kwargs)
    transforms = ntp_transforms(**config.data.transforms.to_dict())
    train_ds = make_dataloader(
        train_dataset, transforms, **config.train_loader.to_dict()
    )

    eval_ds = None
    if not config.skip_eval:
        eval_dataset = dataset_module.load(config.data.eval_data)
        eval_ds = make_dataloader(
            eval_dataset, transforms, **config.eval_loader.to_dict()
        )

    return train_ds, eval_ds


def train_step(config: sws.FinalConfig, model: Model, batch, callback_state, rngs):
    def loss_fn(train_weights, frozen_weights, batch, rngs):
        weights = tree_util.combine(train_weights, frozen_weights)
        forward_dtype = config.forward_dtype

        hidden_states = model.forward(
            weights, **batch["inputs"], rngs=rngs, dtype=forward_dtype
        )

        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        labels = batch["labels"].reshape(-1)

        if "assistant_masks" in batch["inputs"]:
            count = jnp.sum(
                batch["inputs"]["attention_mask"] * batch["inputs"]["assistant_masks"]
            )
            mask = (
                batch["inputs"]["attention_mask"] * batch["inputs"]["assistant_masks"]
            ).reshape(-1)
        else:
            count = jnp.sum(batch["inputs"]["attention_mask"])
            mask = (batch["inputs"]["attention_mask"]).reshape(-1)
        loss = cross_entropy_loss(
            hidden_states,
            labels,
            (
                # The fused XLA chunked CE kernel currently assumes replicated
                # vocab weights; avoid forcing replication for the reference
                # implementation to match Tunix's fsdp-sharded embed/lm_head.
                jax.reshard(
                    weights[model.lm_head_key],
                    logical_to_physical(("none", "none"), model.config.sharding_rules),
                )
                if (config.loss_implementation or None) != "reference"
                else weights[model.lm_head_key]
            ),
            reduction="sum",
            weight=mask,
            implementation=config.loss_implementation or None,
        )

        aux = {"loss": (loss, count), "token": count}

        return loss, aux

    train_weights, frozen_weights = tree_util.partition(model.weights, model.train_mask)

    if config.optimizer.grad_accum > 1:
        microbatch_size = (
            config.train_loader.global_batch_size // config.optimizer.grad_accum
        )
        grad_fn = microbatch(
            jax.value_and_grad(loss_fn, has_aux=True),
            argnums=2,
            microbatch_size=microbatch_size,
        )
    else:
        grad_fn = jax.vlaue_and_grad(loss_fn, has_aux=True)

    (loss, aux), grad = grad_fn(train_weights, frozen_weights, batch, rngs)

    token_count = aux["token"]
    inv_token_count = (1 / token_count).astype(jnp.bfloat16)
    grad = jtu.tree_map(lambda g: g * inv_token_count, grad)

    updates, nst = model.tx.update(
        grad, model.opt_state, model.weights, count=token_count
    )
    nweights = tree_util.apply_updates(model.weights, updates)
    if model.callback_state is not None:
        callback_state = model.callback_updates(
            model.callback_state, grad, updates, model.opt_state, model.weights
        )

    return dataclasses.replace(
        model, weights=nweights, opt_state=nst, callback_state=callback_state
    ), aux


def eval(model, eval_ds):
    raise NotImplementedError


def process_aux(accum_aux: dict[str, Any], namespace=""):
    def finalize(v: Any) -> Any:
        if isinstance(v, tuple):
            return v[0] / v[1]
        return v

    return {f"{namespace}/{k}": finalize(v) for k, v in accum_aux.items()}


def add_aux(accum_aux, aux):
    is_tuple = lambda x: isinstance(x, tuple)

    def f(path, accum_leaf, leaf):
        method = DEFAULT_REDUCED[jtu.keystr(path, simple=True)]
        match method:
            case "max":
                return jnp.maximum(accum_leaf, leaf)
            case "sum":
                return accum_leaf + leaf
            case "min":
                return jnp.minimum(accum_leaf, leaf)
            case "mean":
                return (accum_leaf[0] + leaf[0], accum_leaf[1] + leaf[1])

    return jtu.tree_map_with_path(f, accum_aux, aux, is_leaf=is_tuple)


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
    train_ds,
    eval_ds,
    logger,
    rngs: PRNGKeyArray | None = None,
):
    train_iterator = iter(train_ds)
    step = model.step or 0
    global_aux = dict(DEFAULT_AUX)
    skip_eval = config.skip_eval or config.eval_every is None or eval_ds is None
    first_step = True
    need_save = config.checkpoints.save_interval_steps > 0

    if need_save:
        ckpt_options = ocp.CheckpointManagerOptions(
            **config.checkpoint_options.to_dict()
        )
        ckpt_manager = ocp.CheckpointManager(config.ckpt_path, options=ckpt_options)

    to_log_later = {}
    program_wall_t0 = None
    first_compile_time = None
    warn_assitant_loss = config.data.transforms.assistant_loss

    pbar = None
    if jax.process_index() == 0:
        total = getattr(config, "max_train_step", None)
        pbar = tqdm(
            total=total,
            initial=int(step),
            desc="train",
            unit="step",
            dynamic_ncols=True,
        )

    try:
        while step < config.max_train_step:
            if not skip_eval and (step % config.eval_every_n_steps) == 0:
                eval_aux = eval(model, eval_ds)
                processed_aux = process_aux(eval_aux)

            try:
                batch = next(train_iterator)
                if warn_assitant_loss and "assistant_masks" in batch["inputs"]:
                    check_ast_token = np.any(batch["inputs"]["assistant_masks"])
                    if not check_ast_token and jax.process_index() == 0:
                        raise RuntimeError(
                            "assistant_loss=True was requested but no assistant token was found. "
                            "This can occur if the chat template does not distinguish assistant vs user tokens "
                            "or if truncation (max_length) removed the assistant token. "
                            "please fix this issue before proceeding"
                        )
                    warn_assitant_loss = False
            except StopIteration:
                if jax.process_index() == 0:
                    print("dataloader is exhausted")
                break

            loop_rngs = jax.random.fold_in(rngs, step) if rngs is not None else None
            if first_step:
                with jax.named_scope("compile train step"):
                    start_time = time.monotonic()
                    train_step_jit = jax.jit(
                        partial(train_step, config),
                        donate_argnums=(0,),
                    )
                    lower = train_step_jit.lower(model, batch, loop_rngs)
                    train_step_fn = lower.compile()
                    first_compile_time = time.monotonic() - start_time

                    if jax.process_index() == 0:
                        print("compile time: ", first_compile_time)
                        memory_stats = print_compiled_memory_stats(
                            train_step_fn.memory_analysis()
                        )
                        cost = print_flops(train_step_fn.cost_analysis())

                        to_log_later.update(memory_stats)
                        to_log_later.update(cost)

                    program_wall_t0 = time.monotonic()
                    model, aux = train_step_fn(model, batch, loop_rngs)
                    first_step = False
            else:
                with (
                    jax.named_scope("train_step"),
                    jax.profiler.StepTraceAnnotation(f"train_step_{step}"),
                ):
                    model, aux = train_step_fn(model, batch, loop_rngs)

            callback_output, callback_state = model.callback_process(
                model.callback_state, aux
            )
            dataclasses.replace(model, callback_state=callback_state)
            global_aux = add_aux(global_aux, aux)
            processed_aux = process_aux(aux, "step")
            cum_processed_aux = process_aux(global_aux, "cum")
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
            if need_save:
                ckpt_manager.save(
                    step,
                    args=ocp.args.Composite(
                        weights=ocp.args.StandardSave(model.weights),
                        opt_state=ocp.args.StandardSave(model.opt_state),
                        step=ocp.args.JsonSave(int(step)),
                    ),
                )
            step += 1

    finally:
        to_log_later["program_time"] = time.monotonic() - program_wall_t0
        to_log_later["compile_time"] = first_compile_time
        to_log_later.update(process_aux(global_aux, "cum"))
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
        if ckpt_manager is not None:
            ckpt_manager.close()

    return model, to_log_later


def create_logger(config: sws.FinalConfig):
    if jax.process_index() != 0:
        return load_logger(config, "noop")
    logger_name = getattr(config, "logger_name", "noop")
    return load_logger(config, logger_name)


def load_callback(config: sws.FinalConfig):
    callbacks = []
    for callback in config.callback:
        if isinstance(callback, str):
            callback_lib = importlib.import_module(f"jaxformers.callbacks.{callback}")
            callbacks.append(callback_lib.make())
        elif isinstance(callback, tuple):
            callback_lib = importlib.import_module(
                f"jaxformers.callbacks.{callback[0]}"
            )
            callbacks.append(callback_lib.make(**callback[1]))
        else:
            raise ValueError(
                f"Invalid callback specification: {callback!r}. "
                "Expected either a string callback name (e.g. 'my_callback') "
                "or a tuple of the form (callback_name, kwargs_dict)."
            )
    return callback_chain(*callbacks)


def main(config: sws.FinalConfig):
    _preparse_absl_flags()
    logger = create_logger(config)
    try:
        rngs = jax.random.key(config.train_seed) if config.train_seed else None
        if not config.random_init and not config.resume:
            model = make_model(config.model_name, config.init_model, config.model_config.to_dict())
            print("DEBUGPRINT {model}:", model)
            scheduler = load_scheduler(config, config.lr_scheduler_name)
            if config.use_lora:
                if config.random_init_lora:
                    rngs, lora_rngs = jax.random.split(rngs, 2)
                    model = loraify(model, **config.lora.to_dict(), rngs=lora_rngs)
                else:
                    raise NotImplementedError
            t0 = time.monotonic()
            model = load_optimizer(config, model, config.optimizer_name, scheduler)
            diff = time.monotonic() - t0
            print(f"Created optimizer and its state in {diff:.2f} seconds.")
        else:
            raise NotImplementedError

        train_ds, eval_ds = load_dataset(config, config.data_name)
        _, metrics = train(config, model, train_ds, eval_ds, logger, rngs)
        return metrics
    finally:
        if jax.process_index() == 0:
            logger.finish()


if __name__ == "__main__":
    sws.run(main)
