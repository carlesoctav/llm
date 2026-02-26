import dataclasses
import importlib
import json
import sys
import time
from functools import partial, reduce
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import orbax.checkpoint as ocp
import quax._core as qc
import sws
from jax.experimental.rnn import PRNGKeyArray
from rich.themes import DEFAULT
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import wandb
from jaxformers import tree_util
from jaxformers.benchmark_utils import print_compiled_memory_stats
from jaxformers.data.next_token_prediction import transforms as ntp_transforms
from jaxformers.data.training import make_dataloader
from jaxformers.dispatch.lora import loraify
from jaxformers.modeling_utils import logical_to_physical, Model
from jaxformers.ops.cross_entropy.api import cross_entropy_loss
from jaxformers.optimizer_utils import (
    find_grad_norm,
    find_learning_rate,
    mask_trainable_lora,
)


DEFAULT_REDUCED = {"loss": "mean", "token_count": "sum"}
DEFAULT_AUX = {"loss": (0, 0), "token_count": 0}


class _NoopLoggerConfig:
    def update(self, values: dict[str, Any]) -> None:
        _ = values


class NoopLogger:
    def __init__(self):
        self.config = _NoopLoggerConfig()

    def log(self, values: dict[str, Any], *, step: int) -> None:
        _ = (values, step)

    def finish(self) -> None:
        return


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


# orig = qc.Value.default
# def dbg(prim, values, params):
#     if any(isinstance(v, LoraArray) for v in values):
#         print("quax fallback primtiive", prim)
#     return orig(prim, values, params)


# qc.Value.default = staticmethod(dbg)
#
# def get_config():
#     config = sws.Config()

#     config.resume = False
#     config.random_init = False
#     config.skip_eval = True

#     config.use_lora = True
#     config.random_init_lora = True

#     config.exp_name = "test1"
#     config.dir = "gs://carles-git-good"
#     config.ckpt_path = lambda: f"{config.dir}/{config.exp_name}"
#     config.train_seed = 42
#     config.eval_every = None
#     config.max_train_step = 1000
#     config.forward_dtype = lambda: jnp.bfloat16
#     config.loss_implementation = "reference"

#     config.model_name = "qwen3"
#     config.model.parallel_dims = {"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 4}
#     config.model.model_id = "Qwen/Qwen3-0.6B"
#     config.model.additional_config.remat_layer = True
#     config.model.additional_config.attn_implementation = "sdpa"
#     config.model.additional_config.sequence_parallelism = True
#     config.model.additional_config.loss_parallel = False

#     config.model.devices = jax.devices()
#     config.model.param_dtype = lambda: jnp.bfloat16

#     # config.lora.rank = 64
#     # config.lora.alpha = 1
#     # config.lora.weights_path = [
#     #     "*.q_proj.weight",
#     #     "*.k_proj.weight",
#     #     "*.v_proj.weight",
#     #     "*.o_proj.weight",
#     #     "*.gate_proj.weight",
#     #     "*.up_proj.weight",
#     #     "*.down_proj.weight",
#     # ]

#     config.lr_scheduler_name = None
#     config.learning_rate = 1e-5

#     config.optimizer_name = "adam"
#     config.optimizer.max_grad_norm = 1.0
#     config.optimizer.grad_accum = 1

#     config.data_name = "huggingface"
#     config.data.load_kwargs = [
#         {
#             "path": "allenai/Dolci-Instruct-SFT",
#             "split": "train",
#             "streaming": False,
#         }
#     ]
#     config.data.transforms.column = "messages"
#     config.data.transforms.max_length = 8192
#     config.data.transforms.tokenizer = lambda: AutoTokenizer.from_pretrained(
#         config.model.model_id
#     )
#     config.data.transforms.assistant_loss = False
#     config.data.transforms.is_tokenized = False
#     config.data.transforms.is_chat = True
#     config.data.transforms.chat_template_path = None
#     config.data.transforms.packing = True
#     config.data.transforms.packing_bins = 64

#     # total batch size is global_batch_size * config.optimizer.grad_accum
#     config.train_loader.global_batch_size = 8
#     config.train_loader.seed = 42

#     # config.train_loader.pspec =
#     # config.train_loader.mesh =
#     # config.train_loader.num_epochs =
#     # config.train_loader.dataset_weights =
#     # config.train_loader.dataloading_host_index =
#     # config.train_loader.dataloading_host_count =
#     # config.train_loader.is_not_sharded =
#     # config.train_loader.read_num_threads =
#     # config.train_loader.read_prefetch_buffer_size =
#     # config.train_loader.shuffle =
#     # config.train_loader.shuffle_buffer_size =
#     # config.train_loader.worker_count =
#     # config.train_loader.worker_buffer_size =
#     # config.train_loader.drop_remainder =

#     config.log.grad_norm = True
#     config.log.learning_rate = True

#     # see orbax checkpointmanager options
#     config.checkpoint_options.save_interval_steps = 10000
#     config.checkpoint_options.max_to_keep = 1

#     config.wandb.project = "test-training"
#     config.wandb.name = lambda: config.exp_name
#     # config.wandb.entity =
#     # config.wandb.dir =
#     # config.wandb.id =
#     # config.wandb.notes =
#     # config.wandb.tags =

#     return config


def load_model(config: sws.FinalConfig, name: str):
    model_module = importlib.import_module(f"jaxformers.models.{name}")
    model = model_module.load(**config.model.to_dict())
    return model


def load_optimizer(config: sws.FinalConfig, model, opt_name, scheduler):
    optmizer_module = importlib.import_module(f"jaxformers.optimizers.{opt_name}")
    train_mask = None
    if getattr(model, "is_lora", False):
        train_mask = mask_trainable_lora(model.weights)

    train_weights, _ = tree_util.partition(model.weights, train_mask)
    tx = optmizer_module.make(scheduler, **config.optimizer.to_dict())
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


def train_step(config: sws.FinalConfig, model: Model, batch, rngs):
    def loss_fn(train_weights, frozen_weights, batch, rngs):
        weights = tree_util.combine(train_weights, frozen_weights)

        hidden_states = model.forward(weights, **batch["inputs"], rngs=rngs)
        hidden_states = hidden_states.reshape(
            -1,
            hidden_states.shape[-1],
            out_sharding=logical_to_physical(
                ("batch", "context", "none"), model.config.sharding_rules
            ),
        )
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
            jax.reshard(
                weights[model.lm_head_key],
                logical_to_physical(("none", "none"), model.config.sharding_rules),
            ),
            reduction="sum",
            weight=mask,
            implementation=config.loss_implementation or None,
        )

        aux = {"loss": (loss, count), "token_count": count}

        return loss, aux

    train_weights, frozen_weights = tree_util.partition(model.weights, model.train_mask)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    k = config.optimizer.grad_accum
    c = model.opt_state[0].count
    emit = c == (k - 1)

    (loss, aux), grad = grad_fn(train_weights, frozen_weights, batch, rngs)
    token_count = aux["token_count"]

    updates, nst = model.tx.update(
        grad, model.opt_state, model.weights, count=token_count
    )

    nst = (nst[0], nst[1]) + jtu.tree_map(
        lambda nst, st: jnp.where(emit, nst, st), nst[2:], model.opt_state[2:]
    )

    nweights = tree_util.apply_updates(model.weights, updates)

    return dataclasses.replace(model, weights=nweights, opt_state=nst), aux


def eval(model, eval_ds):
    raise NotImplementedError


def process_aux(accum_aux: dict[str, Any], namespace=""):
    return {
        f"{namespace}/{k}": v[0] / v[1] if isinstance(v, tuple) else v
        for k, v in accum_aux.items()
    }


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


def _jsonable(value: Any) -> Any:
    value = jax.device_get(value)
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


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
    mini_step = 0
    accum_aux = dict(DEFAULT_AUX)
    global_aux = dict(DEFAULT_AUX)
    skip_eval = config.skip_eval or config.eval_every is None or eval_ds is None
    first_step = True

    ckpt_options = ocp.CheckpointManagerOptions(**config.checkpoint_options.to_dict())
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
            desc=f"train[p{jax.process_index()}]",
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
            if step == 0 and first_step:
                with jax.named_scope("compile train step"):
                    start_time = time.monotonic()
                    train_step_fn = (
                        jax.jit(partial(train_step, config))
                        .lower(model, batch, loop_rngs)
                        .compile()
                    )
                    first_compile_time = time.monotonic() - start_time

                    program_wall_t0 = time.monotonic()
                    model, aux = train_step_fn(model, batch, loop_rngs)
                    if jax.process_index() == 0:
                        print("compile time: ", first_compile_time)
                        compiled_analysis = train_step_fn.memory_analysis()
                        memory_stats = print_compiled_memory_stats(compiled_analysis)
                        to_log_later.update(memory_stats)
                    first_step = False
            else:
                with (
                    jax.named_scope("train_step"),
                    jax.profiler.StepTraceAnnotation(f"train_step_{step}"),
                ):
                    model, aux = train_step_fn(model, batch, loop_rngs)

            accum_aux = add_aux(accum_aux, aux)
            global_aux = add_aux(global_aux, accum_aux)
            emit = mini_step == (config.optimizer.grad_accum - 1)
            if emit:
                processed_aux = process_aux(accum_aux, "step")
                cum_processed_aux = process_aux(global_aux, "cum")
                if config.log.learning_rate:
                    processed_aux.update(find_learning_rate(model.opt_state))
                if config.log.grad_norm:
                    processed_aux.update(find_grad_norm(model.opt_state))
                if jax.process_index() == 0:
                    logger.log(processed_aux, step=step)
                    logger.log(cum_processed_aux, step=step)
                    if pbar is not None:
                        pbar.set_postfix(
                            pbar_display({**cum_processed_aux, **processed_aux})
                        )
                        pbar.update(1)
                accum_aux = dict(DEFAULT_AUX)
                ckpt_manager.save(
                    step,
                    args=ocp.args.Composite(
                        weights=ocp.args.StandardSave(model.weights),
                        opt_state=ocp.args.StandardSave(model.opt_state),
                        step=ocp.args.JsonSave(int(step)),
                    ),
                )

            mini_step = (mini_step + 1) % config.optimizer.grad_accum
            step = emit * (step + 1) + (1 - emit) * step
    finally:
        if program_wall_t0 is not None:
            to_log_later["program_time"] = time.monotonic() - program_wall_t0
        if first_compile_time is not None:
            to_log_later["compile_time"] = first_compile_time
        final_cum = pbar_display(process_aux(global_aux, "cum"))
        to_log_later.update(final_cum)
        token_count = final_cum.get("cum/token_count")
        if token_count is not None and to_log_later.get("program_time"):
            to_log_later["tokens_per_s"] = token_count / to_log_later["program_time"]
        if jax.process_index() == 0:
            logger.config.update(to_log_later)
        if pbar is not None:
            pbar.close()
        ckpt_manager.close()

    print("DEBUGPRINT {to_log_later}:", to_log_later)
    return model, to_log_later


def _create_logger(config: sws.FinalConfig):
    log_name = getattr(config, "log_name", "wandb")
    if log_name == "noop":
        return NoopLogger()
    if log_name == "wandb":
        log_config = config.to_dict()
        log_config.pop("wandb", None)
        return wandb.init(**config.wandb.to_dict(), config=log_config)
    raise ValueError("log_name must be either 'wandb' or 'noop'")


def main(config: sws.FinalConfig):
    print(config.learning_rate)
    print(type(config.learning_rate))
    _preparse_absl_flags()
    result: dict[str, Any] = {
        "status": "error",
        "error": None,
        "exp_name": getattr(config, "exp_name", None),
        "optimizer_name": getattr(config, "optimizer_name", None),
        "loss_implementation": getattr(config, "loss_implementation", None),
        "max_length": None,
    }
    logger = None
    try:
        try:
            result["max_length"] = config.data.transforms.max_length
        except Exception:
            result["max_length"] = None

        rngs = jax.random.key(config.train_seed) if config.train_seed else None

        if not config.random_init and not config.resume:
            model = load_model(config, config.model_name)
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
        logger = _create_logger(config)
        _, metrics = train(config, model, train_ds, eval_ds, logger, rngs)
        result["status"] = "ok"
        result["metrics"] = _jsonable(metrics)
    except BaseException as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if logger is not None and hasattr(logger, "finish"):
            try:
                logger.finish()
            except Exception:
                pass
        print("NTP_RESULT", json.dumps(_jsonable(result), sort_keys=True))


if __name__ == "__main__":
    sws.run(main)
