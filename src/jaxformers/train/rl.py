from __future__ import annotations

import dataclasses
import sys
import time
from contextlib import nullcontext
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
from jaxformers.benchmark_utils import print_compiled_memory_stats, print_flops
from jaxformers.callbacks import make_callbacks
from jaxformers.data.rl import make_rl_data_loader, make_task
from jaxformers.data.rl_tokenize_transforms import make_tokenize_rl_input
from jaxformers.dispatch.lora import make_lora
from jaxformers.distributed.parallel import check_mesh_axis_for_inference
from jaxformers.logger import make_logger
from jaxformers.modeling_utils import logical_to_physical, Model
from jaxformers.models import make_model, prepare_weights as prepare_model_weights
from jaxformers.optimizers import make_optimizer, make_scheduler
from jaxformers.rollout import make_rollout_engine
from jaxformers.sws_utils import run as sws_run


DEFAULT_REDUCED = {
    "loss": "mean",
    "reward": "mean",
    "token": "sum",
    "batch": "sum",
}
DEFAULT_AUX = {
    "loss": (0.0, 0.0),
    "reward": (0.0, 0.0),
    "token": 0.0,
    "batch": 0.0,
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


def _add_aux(accum_aux, aux):
    is_tuple = lambda x: isinstance(x, tuple)

    def f(path, accum_leaf, leaf):
        method = DEFAULT_REDUCED[jtu.keystr(path, simple=True)]
        match method:
            case "sum":
                return accum_leaf + leaf
            case "mean":
                return (accum_leaf[0] + leaf[0], accum_leaf[1] + leaf[1])
            case _:
                raise ValueError(f"Unsupported reduction method: {method!r}")

    return jtu.tree_map_with_path(f, accum_aux, aux, is_leaf=is_tuple)


def _process_aux(accum_aux: dict[str, Any], namespace: str = ""):
    def _to_host(value: Any) -> Any:
        value = jax.device_get(value)
        if isinstance(value, np.ndarray) and value.shape == ():
            return value.item()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def finalize(v: Any) -> Any:
        if isinstance(v, tuple):
            numer = _to_host(v[0])
            denom = max(_to_host(v[1]), 1)
            return numer / denom
        return _to_host(v)

    return {f"{namespace}/{k}": finalize(v) for k, v in accum_aux.items()}


def _pbar_display(metrics: dict[str, Any]) -> dict[str, Any]:
    def to_py(value: Any) -> Any:
        value = jax.device_get(value)
        if isinstance(value, np.ndarray) and value.shape == ():
            return value.item()
        if isinstance(value, np.generic):
            return value.item()
        return value

    return {k: to_py(v) for k, v in metrics.items()}


def _first_defined(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _build_rollout_config(config: sws.FinalConfig) -> dict[str, Any]:
    rollout_config = config.rollout.to_dict() if hasattr(config, "rollout") else {}
    train_max_seq_len = _first_defined(
        getattr(config, "train_max_seq_len", None),
        getattr(getattr(config, "data", None), "transforms", None)
        and getattr(config.data.transforms, "max_length", None),
    )
    rollout_max_seq_len = _first_defined(
        rollout_config.get("max_seq_len"),
        int(train_max_seq_len) + 1 if train_max_seq_len is not None else None,
    )
    if rollout_max_seq_len is None:
        raise ValueError("RL training requires either rollout.max_seq_len or train_max_seq_len.")

    return {
        "temperature": _first_defined(
            rollout_config.get("temperature"),
            getattr(config, "sampling_temperature", None),
            1.0,
        ),
        "top_k": _first_defined(
            rollout_config.get("top_k"),
            -1,
        ),
        "top_p": _first_defined(
            rollout_config.get("top_p"),
            1.0,
        ),
        "max_decode_steps": _first_defined(
            rollout_config.get("max_decode_steps"),
            rollout_config.get("max_decode_step"),
            getattr(config, "sampling_max_decode_steps", None),
            256,
        ),
        "decode_chunk_size": _first_defined(
            rollout_config.get("decode_chunk_size"),
            getattr(config, "sampling_intermediate_decode_steps", None),
        ),
        "max_seq_len": int(rollout_max_seq_len),
        "max_input_len": _first_defined(
            rollout_config.get("max_input_len"),
            getattr(config, "sampling_max_input_len", None),
        ),
        "num_samples_per_example": int(
            _first_defined(
                rollout_config.get("num_samples_per_example"),
                rollout_config.get("num_sampels_per_examples"),
                getattr(config, "num_samples_per_example", None),
                1,
            )
        ),
        "min_prefill_size": int(
            _first_defined(
                rollout_config.get("min_prefill_size"),
                256,
            )
        ),
        "prefill_size": _first_defined(
            rollout_config.get("prefill_size"),
            getattr(config, "sampling_prefill_size", None),
        ),
    }


def _build_tasks(config: sws.FinalConfig):
    task_specs = getattr(config, "tasks", None)
    if task_specs is None and hasattr(config, "data"):
        task_specs = getattr(config.data, "tasks", None)

    if not task_specs:
        raise ValueError("RL training requires config.tasks or config.data.tasks.")

    tasks = []
    for task_spec in task_specs:
        task_spec = dict(task_spec)
        task_name = task_spec.pop("name")
        tasks.append(make_task(task_name, **task_spec))
    return tasks


def _default_chat_template_path() -> str | None:
    template_path = Path("temp/think.jinja")
    if template_path.exists():
        return str(template_path)
    return None


def _configure_rollout_model(model: Model) -> Model:
    additional_config = dict(getattr(model.config, "additional_config", {}))
    additional_config["remat_layer"] = False
    additional_config["forward_impl"] = "loop"
    additional_config["sequence_parallelism"] = False
    model.config.additional_config = additional_config
    return model


def _build_rollout_parallel_dims(config: sws.FinalConfig, model: Model) -> dict[str, int]:
    rollout_config = config.rollout.to_dict() if hasattr(config, "rollout") else {}
    rollout_parallel_dims = rollout_config.get("parallel_dims")
    if rollout_parallel_dims is not None:
        parallel_dims = {k: int(v) for k, v in dict(rollout_parallel_dims).items()}
        check_mesh_axis_for_inference(parallel_dims)
        return parallel_dims

    train_parallel_dims = getattr(model.config, "parallel_dims", None)
    if train_parallel_dims is None:
        train_parallel_dims = getattr(config.model, "parallel_dims", None)
    if train_parallel_dims is None:
        raise ValueError("Could not infer train parallel dims for rollout mesh.")

    total_devices = 1
    for axis_size in dict(train_parallel_dims).values():
        total_devices *= int(axis_size)

    parallel_dims = {
        "dp_replicate": 1,
        "dp_shard": 1,
        "cp": 1,
        "tp": int(total_devices),
    }
    check_mesh_axis_for_inference(parallel_dims)
    return parallel_dims


def rl_train_step(config: sws.FinalConfig, model: Model, batch, rngs):
    def loss_fn(train_weights, frozen_weights, batch, rngs):
        weights = tree_util.combine(train_weights, frozen_weights)
        forward_dtype = config.forward_dtype

        input_ids = batch["token_ids"][:, :-1]
        attention_mask = batch["attention_mask"][:, :-1]
        labels = batch["token_ids"][:, 1:]
        target_mask = batch["attention_mask"][:, 1:] & batch["generation_mask"][:, 1:]
        reward = batch["rewards"][:, None].astype(jnp.float32)

        hidden_states = model.forward(
            weights,
            input_ids=input_ids,
            attention_mask=attention_mask,
            rngs=rngs,
            dtype=forward_dtype,
        )
        logits = model.unembed(
            weights,
            hidden_states,
            dtype=jnp.float32,
        )
        logits = jax.reshard(
            logits,
            logical_to_physical(("batch", "context", "none"), model.config.sharding_rules),
        )
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        token_logprobs = jnp.take_along_axis(
            log_probs,
            labels[..., None],
            axis=-1,
        ).squeeze(-1)

        weighted_loss = -(token_logprobs * reward * target_mask)
        loss = jnp.sum(weighted_loss)
        token_count = jnp.maximum(jnp.sum(target_mask), 1)
        batch_size = batch["token_ids"].shape[0]
        aux = {
            "loss": (loss, token_count),
            "reward": (jnp.sum(batch["rewards"]), batch_size),
            "token": token_count,
            "batch": batch_size,
        }
        return loss, aux

    train_weights, frozen_weights = tree_util.partition(model.weights, model.train_mask)

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
    grad = jtu.tree_map(lambda g: g * (1 / token_count).astype(jnp.bfloat16), grad)

    updates, nst = model.tx.update(
        grad,
        model.opt_state,
        model.weights,
        count=token_count,
    )
    nweights = tree_util.apply_updates(model.weights, updates)
    callback_state = model.callback_state
    if model.callback_state is not None:
        callback_state = model.callbacks.update(
            model.callback_state,
            grad,
            updates,
            nst,
            nweights,
            aux,
        )

    return dataclasses.replace(
        model,
        weights=nweights,
        opt_state=nst,
        callback_state=callback_state,
    ), aux


def train(
    config,
    model: Model,
    rl_loader,
    logger,
    rngs: PRNGKeyArray | None = None,
):
    step = model.step or 0
    global_aux = dict(DEFAULT_AUX)
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
    first_compile_time = None
    pbar = None
    if jax.process_index() == 0:
        pbar = tqdm(
            total=config.max_train_step,
            initial=int(step),
            desc="rl",
            unit="step",
            dynamic_ncols=True,
        )

    mesh_ctx = jax.set_mesh(model.mesh) if model.mesh is not None else nullcontext()
    try:
        with mesh_ctx:
            while step < config.max_train_step:
                rl_loader.set_params(model.weights)
                batch, rollout_metrics = next(rl_loader)
                loop_rngs = jax.random.fold_in(rngs, step) if rngs is not None else None

                if first_step:
                    with jax.named_scope("compile rl train step"):
                        start_time = time.monotonic()
                        train_step_jit = jax.jit(
                            partial(rl_train_step, config),
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
                        jax.named_scope("rl_train_step"),
                        jax.profiler.StepTraceAnnotation(f"rl_train_step_{step}"),
                    ):
                        model, aux = train_step_fn(model, batch, loop_rngs)

                callback_output = {}
                if model.callback_state is not None:
                    callback_output, callback_state = model.callbacks.process(
                        {},
                        model.callback_state,
                        aux,
                    )
                    model = dataclasses.replace(model, callback_state=callback_state)

                global_aux = _add_aux(global_aux, aux)
                processed_aux = _process_aux(aux, "step")
                cum_processed_aux = _process_aux(global_aux, "cum")
                step_metrics = {
                    **processed_aux,
                    **rollout_metrics,
                    **callback_output,
                }

                if jax.process_index() == 0:
                    logger.log(step_metrics, step=step)
                    logger.log(cum_processed_aux, step=step)
                    if pbar is not None:
                        pbar.set_postfix(
                            _pbar_display(
                                {
                                    **step_metrics,
                                    **cum_processed_aux,
                                }
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
        if program_wall_t0 is not None:
            to_log_later["program_time"] = time.monotonic() - program_wall_t0
        to_log_later["compile_time"] = first_compile_time
        to_log_later.update(_process_aux(global_aux, "cum"))
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


def main(config: sws.FinalConfig):
    _preparse_absl_flags()

    logger = None
    if jax.process_index() == 0:
        logger = make_logger(config.logger_name, config.logger.to_dict())
        logger.config.update(config.to_dict())
    try:
        rngs = jax.random.key(config.train_seed) if config.train_seed else None
        model = make_model(config.model_name, config.init_model, config.model.to_dict())
        scheduler = make_scheduler(config.lr_scheduler_name, config.learning_rate)
        model = make_lora(model, config.init_lora, config.lora.to_dict(), rngs=rngs)
        model = prepare_model_weights(config.model_name, model)
        model = make_optimizer(
            config.optimizer_name,
            model,
            scheduler,
            config.optimizer.to_dict(),
        )

        callbacks = make_callbacks(config.callback)
        if callbacks is not None:
            model = dataclasses.replace(
                model,
                callback_state=callbacks.init(model.weights, model.opt_state),
                callbacks=callbacks,
            )

        rollout_config = _build_rollout_config(config)
        rollout_model_config = dict(config.model.to_dict())
        rollout_model_config["parallel_dims"] = _build_rollout_parallel_dims(config, model)
        rollout_model_config["devices"] = list(np.asarray(model.mesh.devices).reshape(-1))
        rollout_additional_config = dict(rollout_model_config.get("additional_config", {}))
        rollout_additional_config["remat_layer"] = False
        rollout_additional_config["forward_impl"] = "loop"
        rollout_additional_config["sequence_parallelism"] = False
        rollout_model_config["additional_config"] = rollout_additional_config
        rollout_model = make_model(
            config.model_name,
            config.init_model,
            rollout_model_config,
        )
        rollout_model = _configure_rollout_model(rollout_model)
        rollout_model = make_lora(
            rollout_model,
            config.init_lora,
            config.lora.to_dict(),
            rngs=rngs,
        )
        rollout_model = prepare_model_weights(config.model_name, rollout_model)
        rollout_engine = make_rollout_engine(
            getattr(config, "rollout_name", "simple"),
            model=rollout_model,
            rollout_config=rollout_config,
            params=rollout_model.weights,
            forward_dtype=config.forward_dtype,
        )
        rollout_engine.set_params(model.weights)

        chat_template_path = _default_chat_template_path()
        if hasattr(config, "data") and hasattr(config.data, "transforms"):
            chat_template_path = getattr(
                config.data.transforms,
                "chat_template_path",
                chat_template_path,
            )

        rl_loader = make_rl_data_loader(
            tasks=_build_tasks(config),
            rollout_engine=rollout_engine,
            reward_name=getattr(config, "reward_name", None),
            reward_config=config.reward.to_dict() if hasattr(config, "reward") else {},
            loader_config=config.train_loader.to_dict(),
            tokenizer_transform=make_tokenize_rl_input(
                tokenizer=model.tokenizer,
                chat_template_path=chat_template_path,
            ),
        )

        _, metrics = train(config, model, rl_loader, logger, rngs)
        return metrics
    finally:
        if logger is not None:
            logger.finish()


if __name__ == "__main__":
    sws_run(main)
