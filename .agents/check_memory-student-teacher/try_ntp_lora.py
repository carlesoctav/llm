from __future__ import annotations

import dataclasses
import sys
import time
from functools import partial
from pathlib import Path
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
from jaxformers.benchmark_utils import (
    print_compiled_memory_stats,
    print_flops,
    print_timing,
    print_train_state_size,
)
from jaxformers.callbacks import make_callbacks
from jaxformers.data import make_dataset
from jaxformers.dispatch.lora import LoraArray, make_lora
from jaxformers.logger import make_logger
from jaxformers.modeling_utils import logical_to_physical, Model
from jaxformers.models import make_model, prepare_weights as prepare_model_weights
from jaxformers.ops.cross_entropy.api import cross_entropy_loss
from jaxformers.optimizers import make_optimizer
from jaxformers.print_utils import tree_pprint
from jaxformers.scheduler import make_scheduler
from jaxformers.sws_utils import merge_config_builders, run as sws_run


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATHS = [
    str(ROOT / "experiments/tunix-repro/name_base_config.py"),
    str(ROOT / "experiments/tunix-repro/model_base_config.py"),
    str(ROOT / "experiments/tunix-repro/data_base_config.py"),
    str(ROOT / "experiments/tunix-repro/optimizer_base_config.py"),
    str(ROOT / "experiments/tunix-repro/scheduler_base_config.py"),
]
DEFAULT_REDUCED = {
    "loss": "mean",
    "token": "sum",
    "batch": "sum",
    "teacher_probe": "mean",
}
DEFAULT_AUX = {
    "loss": (0.0, 0),
    "token": 0,
    "batch": 0,
    "teacher_probe": (0.0, 0),
}
DEFAULT_LORA_PATHS = [
    "*.q_proj.weight",
    "*.k_proj.weight",
    "*.v_proj.weight",
    "*.o_proj.weight",
    "*.gate_proj.weight",
    "*.up_proj.weight",
    "*.down_proj.weight",
]


def get_config() -> sws.Config:
    config = merge_config_builders(CONFIG_PATHS)

    config.init_lora = "random"
    config.lora.rank = 256
    config.lora.alpha = 512
    config.lora.weights_path = list(DEFAULT_LORA_PATHS)

    config.model.additional_config.forward_impl = "scan_layer"
    config.model.additional_config.remat_layer = True

    config.max_train_step = 1000
    config.logger_name = "noop"
    config.callback_name = []
    config.checkpoint_options.save_interval_steps = 0
    config.data.transforms.chat_template_path = str(ROOT / "temp/think.jinja")

    return config


def _preparse_absl_flags() -> None:
    """Avoid absl.flags crashing on this script's CLI args."""

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


def _to_host(tree):
    tree = jax.device_get(tree)

    def _item(value):
        if isinstance(value, np.ndarray) and value.shape == ():
            return value.item()
        if isinstance(value, np.generic):
            return value.item()
        return value

    return jtu.tree_map(_item, tree)


def process_aux(accum_aux: dict[str, Any], namespace=""):
    def finalize(v: Any) -> Any:
        if isinstance(v, tuple):
            numer, denom = v
            return numer / denom if denom else 0.0
        return v

    prefix = f"{namespace}/" if namespace else ""
    return {f"{prefix}{k}": finalize(v) for k, v in accum_aux.items()}


def add_aux(accum_aux, aux):
    is_tuple = lambda x: isinstance(x, tuple)

    def f(path, accum_leaf, leaf):
        method = DEFAULT_REDUCED[jtu.keystr(path, simple=True)]
        match method:
            case "max":
                return max(accum_leaf, leaf)
            case "sum":
                return accum_leaf + leaf
            case "min":
                return min(accum_leaf, leaf)
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


def _strip_lora_weights(weights):
    is_lora_array = lambda x: isinstance(x, LoraArray)
    return jtu.tree_map(
        lambda leaf: leaf._w if isinstance(leaf, LoraArray) else leaf,
        weights,
        is_leaf=is_lora_array,
    )


def make_base_model_from_train_model(train_model: Model) -> Model:
    base_weights = _strip_lora_weights(train_model.weights)
    return dataclasses.replace(
        train_model,
        weights=base_weights,
        is_lora=False,
        train_mask=None,
        tx=None,
        opt_state=None,
        callback_state=None,
        callbacks=None,
    )


def print_shared_lora_summary(train_model: Model, base_model: Model) -> None:
    shared_count = 0
    total_lora = 0

    for key, train_leaf in train_model.weights.items():
        if not isinstance(train_leaf, LoraArray):
            continue
        total_lora += 1
        if train_leaf._w is base_model.weights[key]:
            shared_count += 1

    print(f"LoRA/base shared leaves: {shared_count}/{total_lora}")


def train_step(
    config: sws.FinalConfig,
    train_model: Model,
    base_model: Model,
    batch,
    *,
    rngs,
):
    def loss_fn(train_weights, frozen_weights, batch, rngs):
        weights = tree_util.combine(train_weights, frozen_weights)
        forward_dtype = config.forward_dtype

        hidden_states = train_model.forward(
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
            mask = batch["inputs"]["attention_mask"].reshape(-1)

        loss = cross_entropy_loss(
            hidden_states,
            labels,
            (
                jax.reshard(
                    weights[train_model.lm_head_key],
                    logical_to_physical(
                        ("none", "none"),
                        train_model.config.sharding_rules,
                    ),
                )
                if (config.loss_implementation or None) != "reference"
                else weights[train_model.lm_head_key]
            ),
            reduction="sum",
            weight=mask,
            implementation=config.loss_implementation or None,
        )

        batch_size = batch["labels"].shape[0]
        aux = {"loss": (loss, count), "token": count, "batch": batch_size}
        return loss, aux

    train_weights, frozen_weights = tree_util.partition(
        train_model.weights,
        train_model.train_mask,
    )

    if config.grad_accum > 1:
        microbatch_size = config.train_loader.global_batch_size // config.grad_accum
        grad_fn = microbatch(
            jax.value_and_grad(loss_fn, has_aux=True),
            argnums=2,
            microbatch_size=microbatch_size,
        )
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    (loss, aux), grad = grad_fn(train_weights, frozen_weights, batch, rngs)

    token_count = aux["token"]
    inv_token_count = (1 / token_count).astype(config.forward_dtype)
    grad = jtu.tree_map(lambda g: g * inv_token_count, grad)

    updates, nst = train_model.tx.update(
        grad,
        train_model.opt_state,
        train_model.weights,
        count=token_count,
    )
    nweights = tree_util.apply_updates(
        train_model.weights,
        updates,
        config.forward_dtype,
    )
    teacher_hidden = base_model.forward(
        base_model.weights,
        **batch["inputs"],
        rngs=rngs,
        dtype=config.forward_dtype,
    )
    teacher_probe = teacher_hidden[..., 0].mean().astype(jnp.float32)
    aux = {
        **aux,
        "teacher_probe": (
            teacher_probe,
            jnp.asarray(1, dtype=jnp.int32),
        ),
    }

    callback_state = train_model.callback_state
    if train_model.callback_state is not None:
        callback_state = train_model.callbacks.update(
            train_model.callback_state,
            grad,
            updates,
            nst,
            nweights,
            aux,
        )

    return dataclasses.replace(
        train_model,
        weights=nweights,
        opt_state=nst,
        callback_state=callback_state,
    ), aux


def eval(model, eval_ds):
    raise NotImplementedError


def train(
    config,
    train_model: Model,
    base_model: Model,
    train_ds,
    eval_ds,
    logger,
    *,
    rngs: PRNGKeyArray | None = None,
):
    train_iterator = iter(train_ds)
    step = train_model.step or 0
    global_aux = dict(DEFAULT_AUX)
    skip_eval = config.skip_eval or config.eval_every is None or eval_ds is None
    first_step = True
    need_save = config.checkpoint_options.save_interval_steps > 0
    ckpt_manager = None

    if need_save:
        ckpt_options = ocp.CheckpointManagerOptions(
            **config.checkpoint_options.to_dict()
        )
        ckpt_manager = ocp.CheckpointManager(config.ckpt_path, options=ckpt_options)

    to_log_later = {}
    program_wall_t0 = None
    warn_assitant_loss = config.data.transforms.assistant_loss

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
            if not skip_eval and (step % config.eval_every) == 0:
                eval_aux = eval(train_model, eval_ds)
                _ = process_aux(jax.device_get(eval_aux))

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

                    @print_timing
                    def compile_train_step():
                        train_step_jit = jax.jit(
                            partial(train_step, config),
                            # donate_argnums=(0, 1),
                        )
                        lower = train_step_jit.lower(
                            train_model,
                            base_model,
                            batch,
                            rngs=loop_rngs,
                        )
                        return lower.compile()

                    train_step_fn = compile_train_step()
                    memory_stats = print_compiled_memory_stats(
                        train_step_fn.memory_analysis()
                    )
                    cost = print_flops(train_step_fn.cost_analysis())

                    if memory_stats is not None:
                        to_log_later.update(memory_stats)
                    if cost is not None:
                        to_log_later.update(cost)

                    program_wall_t0 = time.monotonic()
                    train_model, aux = train_step_fn(
                        train_model,
                        base_model,
                        batch,
                        rngs=loop_rngs,
                    )
                    first_step = False
            else:
                with (
                    jax.named_scope("train_step"),
                    jax.profiler.StepTraceAnnotation(f"train_step_{step}"),
                ):
                    train_model, aux = train_step_fn(
                        train_model,
                        base_model,
                        batch,
                        rngs=loop_rngs,
                    )

            callback_output = {}
            if train_model.callback_state is not None:
                callback_output, callback_state = train_model.callbacks.process(
                    {},
                    train_model.callback_state,
                    aux,
                )
                train_model = dataclasses.replace(
                    train_model,
                    callback_state=callback_state,
                )
            host_aux = _to_host(aux)
            global_aux = add_aux(global_aux, host_aux)
            processed_aux = process_aux(host_aux, "step")
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
                        weights=ocp.args.StandardSave(train_model.weights),
                        opt_state=ocp.args.StandardSave(train_model.opt_state),
                        step=ocp.args.JsonSave(int(step)),
                    ),
                )
            step += 1

    finally:
        if program_wall_t0 is not None:
            to_log_later["program_time"] = time.monotonic() - program_wall_t0
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

    return train_model, to_log_later


def main(config: sws.FinalConfig):
    _preparse_absl_flags()
    do_callback = hasattr(config, "callback_name") and bool(config.callback_name)
    do_lora = hasattr(config, "lora")

    logger = None
    if jax.process_index() == 0:
        logger = make_logger(config.logger_name, config.logger.to_dict())
        logger.config.update(config.to_dict())
    try:
        rngs = jax.random.key(config.seed) if config.seed else None
        model_rngs, lora_rngs, train_rngs = (
            jax.random.split(rngs, 3) if rngs is not None else (None, None, None)
        )
        train_model = make_model(
            config.model_name,
            config.init_model,
            config.model.to_dict(),
            rngs=model_rngs,
        )

        def copy_tree(tree):
            def _f(leaf):
                return leaf.copy()

            return jtu.tree_map(_f, tree)

        # base_model = dataclasses.replace(
        #     train_model, weights=copy_tree(train_model.weights)
        # )
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
            train_model = make_lora(
                train_model,
                config.init_lora,
                config.lora.to_dict(),
                rngs=lora_rngs,
            )
        train_model = prepare_model_weights(config.model_name, train_model)
        base_model = make_base_model_from_train_model(train_model)
        # base_model = prepare_model_weights(config.model_name, base_model)
        tree_pprint(base_model.weights)
        print_shared_lora_summary(train_model, base_model)

        train_model = make_optimizer(
            config.optimizer_name,
            train_model,
            scheduler,
            config.optimizer.to_dict(),
        )
        print_train_state_size(train_model)

        if do_callback:
            callbacks = make_callbacks(config.callback_name, config.callback.to_dict())
            train_model = dataclasses.replace(
                train_model,
                callback_state=callbacks.init(
                    train_model.weights,
                    train_model.opt_state,
                ),
                callbacks=callbacks,
            )

        train_ds = make_dataset(
            config.data.source_name,
            config.data.source.to_dict(),
            config.data.transforms_name,
            config.data.transforms.to_dict(),
            config.train_loader_name,
            config.train_loader.to_dict(),
        )
        eval_ds = None
        _, metrics = train(
            config,
            train_model,
            base_model,
            train_ds,
            eval_ds,
            logger,
            rngs=train_rngs,
        )
        return metrics
    finally:
        if logger is not None:
            logger.finish()


if __name__ == "__main__":
    sws_run(main)
