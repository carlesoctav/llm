import dataclasses
import importlib
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
from transformers import AutoTokenizer

import wandb
from jaxformers.benchmark_utils import print_compiled_memory_stats
from jaxformers.data.next_token_prediction import transforms as ntp_transforms
from jaxformers.data.training import make_dataloader
from jaxformers.dispatch.lora import loraify
from jaxformers.modeling_utils import Model
from jaxformers.optimizer_utils import (
    find_grad_norm,
    find_learning_rate,
    mask_non_lora,
)


DEFAULT_REDUCED = {"loss": "mean", "token_count": "sum"}
DEFAULT_AUX = {"loss": (0, 0), "token_count": 0}


# orig = qc.Value.default
# def dbg(prim, values, params):
#     if any(isinstance(v, LoraArray) for v in values):
#         print("quax fallback primtiive", prim)
#     return orig(prim, values, params)


# qc.Value.default = staticmethod(dbg)
#
def get_config():
    config = sws.Config()

    config.resume = False
    config.random_init = False
    config.skip_eval = True

    config.use_lora = True
    config.random_init_lora = True

    config.exp_name = "test1"
    config.dir = "gs://carles-git-good"
    config.ckpt_path = lambda: f"{config.dir}/{config.exp_name}"
    config.train_seed = 42
    config.eval_every = None
    config.max_train_step = 1000
    config.forward_dtype = lambda: jnp.bfloat16

    config.model_name = "qwen3"
    config.model.model_id = "Qwen/Qwen3-4B-Instruct-2507"
    config.model.parallel_dims = {"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 4}
    # config.model.model_id = "Qwen/Qwen3-0.6B"
    config.model.additional_config.gradient_checkpointing = True
    config.model.additional_config.attn_implementation = "sdpa"
    config.model.additional_config.sequence_parallelism = True
    config.model.additional_config.loss_parallel = False

    config.model.devices = jax.devices()
    config.model.param_dtype = lambda: jnp.bfloat16

    config.lora.rank = 8
    config.lora.alpha = 1
    config.lora.weights_path = [
        "*.q_proj.weight",
        "*.k_proj.weight",
        "*.v_proj.weight",
        "*.o_proj.weight",
        "*.gate_proj.weight",
        "*.up_proj.weight",
        "*.down_proj.weight",
    ]

    config.lr_scheduler_name = None
    config.learning_rate = 1e-5

    config.optimizer_name = "adam"
    config.optimizer.max_grad_norm = 1.0
    config.optimizer.grad_accum = 1

    config.data_name = "huggingface"
    config.data.load_kwargs = [
        {
            "path": "carlesoctav/skripsi_UI_membership_30K",
            # "name": "GovReport",
            "split": "train",
            # "streaming": True,
        }
    ]
    config.data.transforms.column = "id_title"
    config.data.transforms.max_length = 512
    config.data.transforms.packing = True
    config.data.transforms.tokenizer = lambda: AutoTokenizer.from_pretrained(
        config.model.model_id
    )
    config.data.transforms.is_tokenized = False
    config.data.transforms.packing_bins = 64

    # total batch size is global_batch_size * config.optimizer.grad_accum
    config.train_loader.global_batch_size = 8
    config.train_loader.seed = 42

    # config.train_loader.pspec =
    # config.train_loader.mesh =
    # config.train_loader.num_epochs =
    # config.train_loader.dataset_weights =
    # config.train_loader.dataloading_host_index =
    # config.train_loader.dataloading_host_count =
    # config.train_loader.is_not_sharded =
    # config.train_loader.read_num_threads =
    # config.train_loader.read_prefetch_buffer_size =
    # config.train_loader.shuffle =
    # config.train_loader.shuffle_buffer_size =
    # config.train_loader.worker_count =
    # config.train_loader.worker_buffer_size =
    # config.train_loader.drop_remainder =

    config.log.grad_norm = True
    config.log.learning_rate = True

    # see orbax checkpointmanager options
    config.checkpoint_options.save_interval_steps = 1000
    config.checkpoint_options.max_to_keep = 1

    config.wandb.project = "test-training"
    config.wandb.name = lambda: config.exp_name
    # config.wandb.entity =
    # config.wandb.dir =
    # config.wandb.id =
    # config.wandb.notes =
    # config.wandb.tags =

    return config


def load_model(config: sws.FinalConfig, name: str):
    model_module = importlib.import_module(f"jaxformers.models.{name}")
    model = model_module.load(**config.model.to_dict())
    return model


def load_optimizer(config: sws.FinalConfig, model, opt_name, scheduler):
    optmizer_module = importlib.import_module(f"jaxformers.optimizers.{opt_name}")
    freeze_mask = None
    if getattr(model, "is_lora", False):
        freeze_mask = mask_non_lora(model.weights)
        print("DEBUGPRINT {freeze_mask}:", freeze_mask)

    tx = optmizer_module.make(
        scheduler, **config.optimizer.to_dict(), freeze_mask=freeze_mask
    )
    opt_state = tx.init(model.weights)
    return dataclasses.replace(model, opt_state=opt_state, tx=tx)


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
    def loss_fn(weights, batch, rngs):
        logits = model.forward(weights, **batch["inputs"], rngs=rngs)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch["labels"]
        )  # [b,t]
        count = jnp.sum(batch["inputs"]["attention_mask"])
        loss = jnp.sum(loss * batch["inputs"]["attention_mask"])
        aux = {"loss": (loss, count), "token_count": count}

        return loss, aux

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    k = config.optimizer.grad_accum
    apply_every_state = model.opt_state[0]
    if isinstance(apply_every_state, optax.MaskedState):
        apply_every_state = apply_every_state.inner_state
    c = apply_every_state.count
    emit = c == (k - 1)

    (loss, aux), grad = grad_fn(model.weights, batch, rngs)
    token_count = aux["token_count"]

    updates, nst = model.tx.update(
        grad, model.opt_state, model.weights, count=token_count
    )

    nst = (nst[0], nst[1]) + jtu.tree_map(
        lambda nst, st: jnp.where(emit, nst, st), nst[2:], model.opt_state[2:]
    )

    nweights = optax.apply_updates(model.weights, updates)

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

    try:
        while step < config.max_train_step:
            if not skip_eval and (step % config.eval_every_n_steps) == 0:
                eval_aux = eval(model, eval_ds)
                processed_aux = process_aux(eval_aux)

            try:
                batch = next(train_iterator)
            except StopIteration:
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
                    model, aux = train_step_fn(model, batch, loop_rngs)
                    print("compile time: ", first_compile_time)
                    compiled_analysis = train_step_fn.memory_analysis()
                    print_compiled_memory_stats(compiled_analysis)
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
        ckpt_manager.close()

    return model


def main(config: sws.FinalConfig):
    rngs = jax.random.key(config.train_seed) if config.train_seed else None

    if not config.random_init and not config.resume:
        # reinventing flax lol
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

    log_config_wandb = config.to_dict()
    del log_config_wandb["wandb"]
    logger = wandb.init(**config.wandb.to_dict(), config=log_config_wandb)

    train(config, model, train_ds, eval_ds, logger, rngs)


if __name__ == "__main__":
    sws.run(main)
