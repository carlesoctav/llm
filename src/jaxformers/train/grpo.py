from __future__ import annotations

import dataclasses
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
from tqdm.auto import tqdm

from jaxformers import tree_util
from jaxformers.benchmark_utils import print_compiled_memory_stats, print_flops
from jaxformers.data import grpo as grpo_data
from jaxformers.inference import LLM
from jaxformers.modeling_utils import logical_to_physical, Model
from jaxformers.ops.cross_entropy.api import cross_entropy_loss
from jaxformers.optimizer_utils import find_grad_norm, find_learning_rate
from jaxformers.train.ntp import (
    _preparse_absl_flags,
    create_logger,
    load_model,
    load_optimizer,
    load_scheduler,
    pbar_display,
)


DEFAULT_REDUCED = {
    "loss": "mean",
    "token_count": "sum",
    "reward": "mean",
    "completion_length": "mean",
    "kl": "mean",
}
DEFAULT_AUX = {
    "loss": (0, 0),
    "token_count": 0,
    "reward": (0, 0),
    "completion_length": (0, 0),
    "kl": (0, 0),
}


def process_aux(accum_aux: dict[str, Any], namespace=""):
    def finalize(v: Any) -> Any:
        if isinstance(v, tuple):
            return v[0] / jnp.clip(v[1], min=1)
        return v

    return {f"{namespace}/{k}": finalize(v) for k, v in accum_aux.items()}


def add_aux(accum_aux, aux):
    is_tuple = lambda x: isinstance(x, tuple)

    def f(path, accum_leaf, leaf):
        method = DEFAULT_REDUCED[jtu.keystr(path, simple=True)]
        match method:
            case "sum":
                return accum_leaf + leaf
            case "mean":
                return (accum_leaf[0] + leaf[0], accum_leaf[1] + leaf[1])
        raise ValueError(f"Unsupported reduction {method!r}")

    return jtu.tree_map_with_path(f, accum_aux, aux, is_leaf=is_tuple)


def create_rollout_llm(config: sws.FinalConfig) -> LLM:
    rollout_config = config.rollout
    return LLM(
        model=rollout_config.model_id,
        tokenizer=getattr(rollout_config, "tokenizer", None),
        tensor_parallel_size=rollout_config.tensor_parallel_size,
        max_num_batched_tokens=rollout_config.max_num_batched_tokens,
        max_num_seqs=rollout_config.max_num_seqs,
        max_model_len=rollout_config.max_model_len,
        gpu_memory_utilization=rollout_config.gpu_memory_utilization,
        dtype=rollout_config.dtype,
        trust_remote_code=getattr(rollout_config, "trust_remote_code", False),
        download_dir=getattr(rollout_config, "download_dir", None),
        model_impl=getattr(rollout_config, "model_impl", "vllm"),
        vllm_model_impl=getattr(rollout_config, "vllm_model_impl", "vllm"),
    )


def load_rollout_dataset(
    config: sws.FinalConfig,
    rollout_llm: LLM,
    model: Model,
):
    dp_shard = int(getattr(config.model.parallel_dims, "dp_shard", 1))
    effective_batch_size = int(config.train_loader.global_batch_size) * int(
        config.rollout.num_generations
    )
    if effective_batch_size % dp_shard != 0:
        raise ValueError(
            "GRPO effective batch size must be divisible by model dp_shard. "
            f"Got global_batch_size={config.train_loader.global_batch_size}, "
            f"num_generations={config.rollout.num_generations}, "
            f"effective_batch_size={effective_batch_size}, dp_shard={dp_shard}."
        )

    datasets = grpo_data.load(config.data.load_kwargs)
    pad_token_id = model.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")

    return grpo_data.GRPORolloutDataset(
        datasets=datasets,
        llm=rollout_llm,
        pad_token_id=int(pad_token_id),
        prompt_batch_size=config.train_loader.global_batch_size,
        num_generations=config.rollout.num_generations,
        max_tokens=config.rollout.max_tokens,
        max_model_len=config.rollout.max_model_len,
        prompt_column=getattr(config.data, "prompt_column", None),
        messages_column=getattr(config.data, "messages_column", None),
        answer_column=getattr(config.data, "answer_column", None),
        prompt_template=getattr(config.data, "prompt_template", None),
        system_prompt=getattr(config.data, "system_prompt", None),
        temperature=config.rollout.temperature,
        top_p=config.rollout.top_p,
        top_k=config.rollout.top_k,
        drop_last=getattr(config.data, "drop_last", True),
    )


def compute_completion_logps(
    config: sws.FinalConfig,
    model: Model,
    weights,
    batch,
    rngs,
):
    hidden_states = model.forward(
        weights,
        **batch["inputs"],
        rngs=rngs,
        dtype=config.forward_dtype,
    )
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    labels = batch["labels"].reshape(-1)

    lm_head = (
        jax.reshard(
            weights[model.lm_head_key],
            logical_to_physical(("none", "none"), model.config.sharding_rules),
        )
        if (config.loss_implementation or None) != "reference"
        else weights[model.lm_head_key]
    )
    implementation = config.loss_implementation or None
    if implementation == "xla_chunked":
        implementation = ("xla_chunked", "reference")
    losses = cross_entropy_loss(
        hidden_states,
        labels,
        lm_head,
        reduction=None,
        implementation=implementation,
    )
    return -losses.reshape(batch["labels"].shape)


def compute_logps_step(
    config: sws.FinalConfig,
    model: Model,
    batch,
    rngs,
):
    return compute_completion_logps(config, model, model.weights, batch, rngs)


def train_step(
    config: sws.FinalConfig,
    model: Model,
    batch,
    rngs,
):
    epsilon = float(config.grpo.epsilon)
    epsilon_high = float(getattr(config.grpo, "epsilon_high", epsilon))
    beta = float(config.grpo.beta)

    def loss_fn(train_weights, frozen_weights, batch, rngs):
        weights = tree_util.combine(train_weights, frozen_weights)
        per_token_logps = compute_completion_logps(config, model, weights, batch, rngs)
        old_logps = batch["old_logps"]
        loss_mask = batch["loss_mask"]
        advantages = batch["advantages"][:, None]

        log_ratio = per_token_logps - old_logps
        coef_1 = jnp.exp(log_ratio)
        coef_2 = jnp.clip(coef_1, 1 - epsilon, 1 + epsilon_high)
        per_token_loss = -jnp.minimum(coef_1 * advantages, coef_2 * advantages)

        if beta != 0.0:
            ref_logps = batch["ref_logps"]
            kl = jnp.exp(ref_logps - per_token_logps) - (ref_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + beta * kl
            kl_sum = jnp.sum(kl * loss_mask)
        else:
            kl_sum = jnp.zeros([], dtype=jnp.float32)

        token_count = jnp.sum(loss_mask)
        loss = jnp.sum(per_token_loss * loss_mask)
        batch_count = jnp.asarray(batch["rewards"].shape[0], dtype=jnp.float32)
        aux = {
            "loss": (loss, token_count),
            "token_count": token_count,
            "reward": (jnp.sum(batch["rewards"]), batch_count),
            "completion_length": (
                jnp.sum(batch["completion_lengths"].astype(jnp.float32)),
                batch_count,
            ),
            "kl": (kl_sum, token_count),
        }
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
        lambda new_state, old_state: jnp.where(emit, new_state, old_state),
        nst[2:],
        model.opt_state[2:],
    )
    nweights = tree_util.apply_updates(model.weights, updates)
    return dataclasses.replace(model, weights=nweights, opt_state=nst), aux


def enrich_batch_with_logps(
    batch: dict[str, Any],
    actor_logps,
    ref_logps=None,
) -> dict[str, Any]:
    enriched = dict(batch)
    enriched["old_logps"] = np.asarray(jax.device_get(actor_logps), dtype=np.float32)
    if ref_logps is None:
        enriched["ref_logps"] = np.zeros_like(enriched["old_logps"], dtype=np.float32)
    else:
        enriched["ref_logps"] = np.asarray(jax.device_get(ref_logps), dtype=np.float32)
    return enriched


def train(
    config,
    model: Model,
    reference_model: Model | None,
    rollout_llm: LLM,
    train_ds,
    logger,
    rngs: PRNGKeyArray | None = None,
):
    train_iterator = iter(train_ds)
    step = model.step or 0
    mini_step = 0
    accum_aux = dict(DEFAULT_AUX)
    global_aux = dict(DEFAULT_AUX)
    first_step = True
    train_step_fn = None
    compute_logps_fn = None

    ckpt_options = ocp.CheckpointManagerOptions(**config.checkpoint_options.to_dict())
    ckpt_manager = ocp.CheckpointManager(config.ckpt_path, options=ckpt_options)
    to_log_later = {}
    program_wall_t0 = None
    first_compile_time = None

    pbar = None
    if jax.process_index() == 0:
        pbar = tqdm(
            total=getattr(config, "max_train_step", None),
            initial=int(step),
            desc="grpo",
            unit="step",
            dynamic_ncols=True,
        )

    try:
        while step < config.max_train_step:
            try:
                raw_batch = next(train_iterator)
            except StopIteration:
                if jax.process_index() == 0:
                    print("rollout dataset is exhausted")
                break

            loop_rngs = jax.random.fold_in(rngs, step) if rngs is not None else None

            if first_step:
                with jax.named_scope("compile grpo helpers"):
                    start_time = time.monotonic()
                    compute_logps_fn = (
                        jax.jit(partial(compute_logps_step, config))
                        .lower(model, raw_batch, loop_rngs)
                        .compile()
                    )
                    prepared_batch = enrich_batch_with_logps(
                        raw_batch,
                        compute_logps_fn(model, raw_batch, loop_rngs),
                        (
                            compute_logps_fn(reference_model, raw_batch, loop_rngs)
                            if reference_model is not None and float(config.grpo.beta) != 0.0
                            else None
                        ),
                    )
                    train_step_fn = (
                        jax.jit(partial(train_step, config), donate_argnums=(0,))
                        .lower(model, prepared_batch, loop_rngs)
                        .compile()
                    )
                    first_compile_time = time.monotonic() - start_time
                    program_wall_t0 = time.monotonic()
                    if jax.process_index() == 0:
                        print("compile time: ", first_compile_time)
                        compiled_analysis = train_step_fn.memory_analysis()
                        to_log_later.update(print_compiled_memory_stats(compiled_analysis))
                        to_log_later.update(print_flops(train_step_fn.cost_analysis()))
                    batch = prepared_batch
                    first_step = False
            else:
                batch = enrich_batch_with_logps(
                    raw_batch,
                    compute_logps_fn(model, raw_batch, loop_rngs),
                    (
                        compute_logps_fn(reference_model, raw_batch, loop_rngs)
                        if reference_model is not None and float(config.grpo.beta) != 0.0
                        else None
                    ),
                )

            for _ in range(config.grpo.num_iterations):
                with (
                    jax.named_scope("grpo_train_step"),
                    jax.profiler.StepTraceAnnotation(f"grpo_train_step_{step}"),
                ):
                    model, aux = train_step_fn(model, batch, loop_rngs)

                accum_aux = add_aux(accum_aux, aux)
                global_aux = add_aux(global_aux, aux)

            emit = mini_step == (config.optimizer.grad_accum - 1)
            if emit:
                processed_aux = process_aux(accum_aux, "step")
                cum_processed_aux = process_aux(global_aux, "cum")
                if config.log.learning_rate:
                    processed_aux.update(find_learning_rate(model.opt_state))
                if config.log.grad_norm:
                    processed_aux.update(find_grad_norm(model.opt_state))
                if (
                    getattr(config.rollout, "sync_every_n_steps", 1) > 0
                    and step % config.rollout.sync_every_n_steps == 0
                ):
                    rollout_llm.sync_jaxformers_weights(model)
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
        program_wall_t0 = program_wall_t0 or time.monotonic()
        to_log_later["program_time"] = time.monotonic() - program_wall_t0
        to_log_later["compile_time"] = first_compile_time
        final_cum = process_aux(global_aux, "cum")
        to_log_later.update(final_cum)
        token_count = to_log_later.get("cum/token_count", 0)
        program_time = max(float(to_log_later.get("program_time", 1.0)), 1e-6)
        to_log_later["systems/tok_s"] = token_count / program_time

        if jax.process_index() == 0:
            logger.config.update(to_log_later)
            print(f"program_time: {to_log_later['program_time']:.3f}s")
            print(f"tok/s: {to_log_later['systems/tok_s']:.2f}")
        if pbar is not None:
            pbar.close()
        ckpt_manager.close()

    return model, to_log_later


def main(config: sws.FinalConfig):
    _preparse_absl_flags()
    logger = create_logger(config)
    try:
        rngs = jax.random.key(config.train_seed) if config.train_seed else None
        if config.random_init or config.resume or config.use_lora:
            raise NotImplementedError("Minimal GRPO path currently supports full-weight training only.")

        model = load_model(config, config.model_name)
        scheduler = load_scheduler(config, config.lr_scheduler_name)
        model = load_optimizer(config, model, config.optimizer_name, scheduler)

        reference_model = None
        if float(config.grpo.beta) != 0.0:
            reference_model = load_model(config, config.model_name)

        rollout_llm = create_rollout_llm(config)
        train_ds = load_rollout_dataset(config, rollout_llm, model)
        _, metrics = train(
            config,
            model,
            reference_model,
            rollout_llm,
            train_ds,
            logger,
            rngs,
        )
        return metrics
    finally:
        if jax.process_index() == 0:
            logger.finish()


if __name__ == "__main__":
    sws.run(main)
