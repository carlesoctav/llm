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
from transformers import AutoTokenizer

import wandb
from jaxformers.benchmark_utils import print_compiled_memory_stats
from jaxformers.data.next_token_prediction import transforms as ntp_transforms
from jaxformers.data.training import make_dataloader
from jaxformers.dispatch.lora import LoraArray, loraify
from jaxformers.modeling_utils import Model
from jaxformers.optimizer_utils import (
    find_apply_every_count,
    _freeze_non_accum_states,
    lora_only_param_labels,
)
from jaxformers.optimizers.log_grad_norm import get_logged_grad_norm


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
    # config.model.parallel_dims = {"dp_replicate": 4, "dp_shard": 1, "cp": 1, "tp": 1}
    # config.model.additional_config = {"attn_implementation": "eager", "loss_parallel": False, "gradient_checkpointing": True}
    config.model.additional_config.gradient_checkpointing = False
    config.model.additional_config.attn_implementation = "sdpa"
    config.model.additional_config.sequence_parallelism = True
    config.model.additional_config.loss_parallel = False

    config.model.devices = jax.devices()
    config.model.param_dtype = lambda: jnp.bfloat16

    config.random_init_lora = True
    config.lora.rank = 64
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
    # config.optimizer.use_grad_mean = True

    # adam related
    # config.optimizer.b1
    # config.optimizer.b2
    # config.optimizer.eps

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

    # plesae don't change this
    config.reduced = {"loss": "mean", "token_count": "sum"}

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
    base_tx = optmizer_module.make(scheduler, **config.optimizer.to_dict())
    if getattr(model, "is_lora", False):
        tx = optax.masked(base_tx, mask = lora_only_param_labels(model.weights))
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
    if k == 1:
        emit = True
    else:
        count = find_apply_every_count(model.opt_state)
        if count is None:
            raise RuntimeError(
                "grad_accum > 1 but couldn't find optax.apply_every state in opt_state"
            )
        emit = (count % k) == (k - 1)

    (loss, aux), grad = grad_fn(model.weights, batch, rngs)

    token_count = aux["token_count"]
    updates, nst = model.tx.update(
        grad, model.opt_state, model.weights, token_count=token_count
    )
    if k != 1:
        nst = _freeze_non_accum_states(emit, nst, model.opt_state)
    nweights = optax.apply_updates(model.weights, updates)

    return dataclasses.replace(model, weights=nweights, opt_state=nst), aux


def eval(model, eval_ds):
    raise NotImplementedError


def process_metrics(config: sws.FinalConfig, accum_aux: list[dict[str, Any]]):
    def process_single(path, *value):
        method = config.reduced[jtu.keystr(path, simple=True)]
        if method == "mean":
            if isinstance(value[0], tuple):
                red = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), value)
                return red[0] / red[1]
        elif method == "sum":
            return np.sum(*value)

    reduced = jtu.tree_map_with_path(
        process_single, *accum_aux, is_leaf=lambda x: isinstance(x, tuple)
    )
    return reduced


def train(
    config,
    model: Model,
    train_ds,
    eval_ds,
    scheduler,
    logger,
    rngs: PRNGKeyArray | None = None,
):
    train_iterator = iter(train_ds)
    step = model.step or 0
    mini_step = 0
    accum_aux = []
    skip_eval = config.skip_eval or config.eval_every is None or eval_ds is None

    ckpt_options = ocp.CheckpointManagerOptions(**config.checkpoint_options.to_dict())
    ckpt_manager = ocp.CheckpointManager(config.ckpt_path, options=ckpt_options)

    try:
        while step < config.max_train_step:
            if not skip_eval and (step % config.eval_every_n_steps) == 0:
                eval_aux = eval(model, eval_ds)
                processed_aux = process_metrics(config, eval_aux)

            try:
                batch = next(train_iterator)
            except StopIteration:
                print("dataloader is exhausted")
                break

            loop_rngs = jax.random.fold_in(rngs, step) if rngs is not None else None
            if step == 0:
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
            else:
                with (
                    jax.named_scope("train_step"),
                    jax.profiler.StepTraceAnnotation(f"train_step_{step}"),
                ):
                    model, aux = train_step_fn(model, batch, loop_rngs)

            emit = mini_step == (config.optimizer.grad_accum - 1)
            accum_aux.append(aux)
            if emit:
                processed_aux = process_metrics(config, accum_aux)
                if config.log.learning_rate:
                    processed_aux["learning_rate"] = (
                        scheduler(step)
                        if isinstance(scheduler, Callable)
                        else scheduler
                    )
                if config.log.grad_norm:
                    logged = get_logged_grad_norm(model.opt_state)
                    processed_aux["grad_norm"] = logged
                if jax.process_index() == 0:
                    logger.log(processed_aux, step=step)
                accum_aux = []

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
        if config.lora is not None:
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

    train(config, model, train_ds, eval_ds, scheduler, logger, rngs)


if __name__ == "__main__":
    sws.run(main)
