import copy
import dataclasses
import importlib
import time
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import orbax.checkpoint as ocp
import sws
import wandb

from jaxformers.data.next_token_prediction import transforms as ntp_transforms
from jaxformers.data.training import make_dataloader
from jaxformers.modeling_utils import Model
optax.tree_utils.tree_l2_norm


def get_config():
    config = sws.Config()

    config.resume = False
    config.random_init = False
    config.skip_eval = True
    config.exp_name = "test1"
    config.dir = "gs://carles-git-good"
    config.ckpt_path = lambda: f"{config.dir}/{config.exp_name}"
    config.eval_every = None
    config.train_seed = None
    config.data_seed = None

    config.model_name = "qwen3"
    config.model.model_id = "Qwen/Qwen3-4B-Instruct-2507"
    config.model.parallel_dims = {"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 4}
    config.model.devices = jax.devices()
    config.model.param_dtype = lambda: jnp.float32

    config.lr_scheduler_name = None
    config.learning_rate = 1e-5

    config.optimizer_name = "adam"
    config.optimizer.max_grad_norm = 1.0
    config.optimizer.grad_accum = 1
    config.optimizer.use_grad_accum_mean = True

    # adam related
    # config.optimizer.b1
    # config.optimizer.b2
    # config.optimizer.eps

    config.data_name = "huggingface"

    config.log.grad_norm = True
    config.log.learning_rate = True

    # see orbax checkpointmanager options
    config.checkpoint_options = {}

    config.reduced = lambda: {
        "total_tokens": "mean",
        "loss": "mean",
        "grad_norm": "mean",
    }

    return config


def load_model(name: str, config: sws.FinalConfig):
    model_module = importlib.import_module(f"jaxformers.models.{name}")
    model = model_module.load(**config.model.to_dict())
    return model


def load_optimizer(model, opt_name, scheduler, config: sws.FinalConfig):
    optmizer_module = importlib.import_module(f"jaxformers.optimizers.{opt_name}")
    tx = optmizer_module.make(scheduler, **config.optimizer.to_dict())
    opt_state = tx.init(model.weights)
    print("DEBUGPRINT {opt_state}:", opt_state)
    return dataclasses.replace(model, opt_state=opt_state, tx=tx)


def load_scheduler(sched_name, config: sws.FinalConfig):
    if sched_name is None:
        return config.learning_rate
    raise NotImplementedError


def load_dataset(data_name: str, config: sws.FinalConfig):
    dataset_module = importlib.import_module(f"jaxformers.data.{data_name}")

    train_dataset = dataset_module.load(config.data.train_data)
    transforms = ntp_transforms(**config.transforms.to_dict())
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


def train_step(config: sws.FinalConfig, model: Model, batch, step: int, rngs):
    def loss_fn(weights, batch, rngs):
        logits = model.forward(weights, **batch, rngs=rngs)  # [b,t, v]
        logits = logits[:, :-1, :]
        labels = jax.nn.one_hot(batch["input_ids"][:, 1:], logits.shape[-1])
        loss = optax.safe_softmax_cross_entropy(logits, labels)  # [b,t-1, v]
        aux = {"loss": loss}

        return loss, aux

    emit = step == (config.grad_accum - 1)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    (loss, aux), grad = grad_fn(model.weights, batch, rngs)
    updates, nst = model.tx.update(grad, model.opt_state, model.weights)
    nst = jtu.tree_map(lambda st, nst: jnp.where(emit, nst, st), model.opt_state, nst)

    nweights = optax.apply_updates(model.weights, updates)
    return dataclasses.replace(model, weights=nweights, opt_state=nst), aux


def eval(model, eval_ds):
    raise NotImplementedError


def process_metrics(config: sws.FinalConfig, accum_aux: list[dict[str, Any]]):
    def process_single(config: sws.FinalConfig, path, *value):
        method = config[jtu.keystr(path)]
        if method == "mean":
            return np.mean(*value)
        elif method == "sum":
            return np.sum(*value)

    reduced = jtu.tree_map_with_path(partial(process_single, config), *accum_aux)
    return reduced


def mini_train_step(config: sws.FinalConfig):
    pass


def train(
    model: Model,
    train_ds,
    eval_ds,
    scheduler,
    logger,
    rngs,
    config,
):
    train_iterator = iter(train_ds)
    step = model.step or 0
    mini_step = 0
    accum_aux = []
    skip_eval = config.skip_eval or config.eval_every_n_steps is None or eval_ds is None

    ckpt_options = ocp.CheckpointManagerOptions(**config.checkpoint_options.to_dict())
    ckpt_manager = ocp.CheckpointManager(config.ckpt_path, ckpt_options)

    first_batch = next(train_iterator)

    with jax.named_scope("compile train step"):
        start_time = time.monotonic()
        train_step_fn = (
            jax.jit(partial(train_step, config))
            .lower(model, first_batch, rngs)
            .compile()
        )
        first_compile_time = time.monotonic() - start_time
        print("compile time: ", first_compile_time)

    # for the first iteration
    model, aux = train_step_fn(model, first_batch, rngs)
    emit = mini_step == (config.grad_accum - 1)

    if emit:
        processed_aux = process_metrics(accum_aux)
        if config.log.learning_rate:
            processed_aux["learning_rate"] = scheduler(step)
        if config.log.grad_norm:
            processed_aux["grad_norm"] = optax.tree_utils.tree_l2_norm(model.opt_state[0].grad_acc)
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

    # new state
    mini_step = (mini_step + 1) % config.grad_accum
    step = emit * (step + 1) + (1 - emit) * step

    while step < config.max_train_step:
        if not skip_eval and (step % config.eval_every_n_steps) == 0:
            eval_aux = eval(model, eval_ds)
            processed_aux = process_metrics(eval_aux)

        batch = next(train_iterator)
        with (
            jax.named_scope("train_step"),
            jax.profiler.StepTraceAnnotation(f"train_step_{step}"),
        ):
            model, aux = train_step_fn(model, batch, rngs)

        emit = mini_step == (config.grad_accum - 1)
        accum_aux.append(aux)

        if emit:
            processed_aux = process_metrics(accum_aux)
            if config.log.learning_rate:
                processed_aux["learning_rate"] = scheduler(step)
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

        mini_step = (mini_step + 1) % config.grad_accum
        step = emit * (step + 1) + (1 - emit) * step

    return model


def main(config: sws.FinalConfig):
    if not config.random_init and not config.resume:
        # reinventing flax lol
        model = load_model(config.model_name, config)
        scheduler = load_scheduler(config.lr_scheduler_name, config)
        model = load_optimizer(model, config.optimizer_name, scheduler, config)
    else:
        raise NotImplementedError

    train_ds, eval_ds = load_dataset(config.data_name, config)

    rngs = jax.random.key(config.train_seed) if config.train_seed else None

    log_config_wandb = config.to_dict()
    del log_config_wandb["wandb"]

    logger = wandb.init(**config.wandb, config = log_config_wandb)

    train(model, train_ds, eval_ds, scheduler, logger, rngs, config)

    print("DEBUGPRINT {model}:", model)


if __name__ == "__main__":
    sws.run(main)
    # sws.run(main)
