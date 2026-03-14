#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen with simply-style dense prefill + decode."
    )
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--local-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=2048)
    parser.add_argument("--max-decode-steps", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--prefill-size", type=int, default=None)
    parser.add_argument("--min-prefill-size", type=int, default=256)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--attn-impl", type=str, default="eager")
    parser.add_argument(
        "--sequence-parallelism",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--non-uniform-prompt-lens",
        action="store_true",
        help="Sample prompt lengths uniformly from [1, input_len] per row.",
    )
    return parser.parse_args()


def _first_leaf(tree):
    import jax.tree_util as jtu

    leaves = jtu.tree_leaves(tree)
    if not leaves:
        raise ValueError("Expected a non-empty pytree")
    return leaves[0]


def _first_available_id(*values: int | None) -> int:
    for value in values:
        if value is not None:
            return int(value)
    return 0


def _resolve_special_ids(tokenizer) -> tuple[int, int, tuple[int, ...]]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    end_of_turn_id = None
    try:
        token_ids = tokenizer.encode("<end_of_turn>", add_special_tokens=False)
        if len(token_ids) == 1:
            end_of_turn_id = int(token_ids[0])
    except Exception:
        end_of_turn_id = None

    pad_id = _first_available_id(pad_token_id, eos_token_id, bos_token_id, 0)
    bos_id = _first_available_id(bos_token_id, eos_token_id, pad_id)

    stop_ids = {
        _first_available_id(eos_token_id, pad_token_id, pad_id),
        _first_available_id(getattr(tokenizer, "sep_token_id", None), eos_token_id, pad_id),
    }
    if end_of_turn_id is not None:
        stop_ids.add(end_of_turn_id)
    return pad_id, bos_id, tuple(sorted(stop_ids))


def _build_prompt_batch(
    *,
    rng: np.random.Generator,
    batch_size: int,
    input_len: int,
    total_length: int,
    pad_id: int,
    bos_id: int,
    non_uniform_prompt_lens: bool,
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    if non_uniform_prompt_lens:
        prompt_lengths = rng.integers(
            low=1,
            high=input_len + 1,
            size=(batch_size,),
            dtype=np.int32,
        )
    else:
        prompt_lengths = np.full((batch_size,), input_len, dtype=np.int32)

    prompt_token_ids: list[list[int]] = []
    for prompt_len in prompt_lengths.tolist():
        token_ids = np.empty((prompt_len,), dtype=np.int32)
        token_ids[0] = bos_id
        if prompt_len > 1:
            token_ids[1:] = rng.integers(
                low=8,
                high=8192,
                size=(prompt_len - 1,),
                dtype=np.int32,
            )
        prompt_token_ids.append(token_ids.tolist())

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from jaxformers.rollout_utils import pad_sequences

    prompt_tokens = pad_sequences(
        prompt_token_ids,
        pad_id=pad_id,
        length=total_length,
    )
    return prompt_token_ids, prompt_lengths, prompt_tokens


def bench_simply_generate(args: argparse.Namespace) -> dict[str, int | float]:
    import jax
    import jax.numpy as jnp

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from jaxformers.attention_utils import init_kv_cache
    from jaxformers.models.huggingface import qwen3
    from jaxformers.rollout.simple import RolloutParams
    from jaxformers.rollout_utils import (
        apply_top_k_top_p,
        make_attention_mask,
    )

    if args.input_len <= 0:
        raise ValueError("--input-len must be > 0")
    if args.max_decode_steps <= 0:
        raise ValueError("--max-decode-steps must be > 0")

    max_seq_len = args.max_seq_len
    if max_seq_len is None:
        max_seq_len = args.input_len + args.max_decode_steps
    if max_seq_len <= 0:
        raise ValueError("--max-seq-len must be > 0")

    rollout_params = RolloutParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_decode_steps=args.max_decode_steps,
        decode_chunk_size=args.max_decode_steps,
        max_seq_len=max_seq_len,
        max_input_len=args.input_len,
        min_prefill_size=args.min_prefill_size,
        prefill_size=args.prefill_size,
    )

    model = qwen3.load(
        model_id=args.model_id,
        parallel_dims={
            "dp_replicate": 1,
            "dp_shard": 1,
            "cp": 1,
            "tp": args.tp,
        },
        local_dir=args.local_dir,
        additional_config={
            "attn_implementation": args.attn_impl,
            "remat_layer": False,
            "sequence_parallelism": args.sequence_parallelism,
            "forward_impl": "loop",
        },
    )

    pad_id, bos_id, eos_ids = _resolve_special_ids(model.tokenizer)
    rng = np.random.default_rng(args.seed)
    prompt_token_ids, prompt_lengths_np, prompt_tokens_np = _build_prompt_batch(
        rng=rng,
        batch_size=args.batch_size,
        input_len=args.input_len,
        total_length=max_seq_len,
        pad_id=pad_id,
        bos_id=bos_id,
        non_uniform_prompt_lens=args.non_uniform_prompt_lens,
    )
    del prompt_token_ids

    schedule = rollout_params.get_decoding_schedule(
        min_input_length=int(prompt_lengths_np.min()),
        max_input_length=int(prompt_lengths_np.max()),
    )
    prefill_size = max(1, min(int(schedule.prefill_size), max_seq_len))
    begin_position = min(int(schedule.begin_position), max_seq_len - 1)
    end_position = min(int(schedule.end_position), max_seq_len - 1)

    prompt_tokens = jnp.asarray(prompt_tokens_np, dtype=jnp.int32)
    prompt_lengths = jnp.asarray(prompt_lengths_np, dtype=jnp.int32)
    visible_lens = np.minimum(prompt_lengths_np, prefill_size).astype(np.int32)
    prefill_attention_mask = jnp.asarray(
        make_attention_mask(visible_lens, prefill_size),
        dtype=jnp.bool_,
    )
    eos_ids_jax = jnp.asarray(eos_ids, dtype=jnp.int32)

    @jax.jit
    def run_prefill(token_matrix, attention_mask, kv_cache):
        hidden_states, kv_next = model.forward(
            model.weights,
            input_ids=token_matrix,
            attention_mask=attention_mask,
            dtype=jnp.bfloat16,
            kv=kv_cache,
            pos=0,
        )
        logits = model.unembed(
            model.weights,
            hidden_states,
            dtype=jnp.float32,
        )
        return logits, kv_next

    @jax.jit
    def run_decode(prng_key, prompt_matrix, prompt_lens, token_matrix, kv_cache):
        batch_size = token_matrix.shape[0]
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)

        def cond_fn(state):
            _key, _tokens, _output_lens, finished, position, _kv = state
            return (position < end_position) & ~jnp.all(finished)

        def body_fn(state):
            key, tokens, output_lens, finished, position, kv = state
            current_tokens = jax.lax.dynamic_slice_in_dim(
                tokens,
                position,
                1,
                axis=1,
            )
            hidden_states, kv = model.forward(
                model.weights,
                input_ids=current_tokens,
                dtype=jnp.bfloat16,
                kv=kv,
                pos=position,
            )
            logits = model.unembed(
                model.weights,
                hidden_states,
                dtype=jnp.float32,
            )
            next_logits = logits[:, 0, :]

            sample_key, next_key = jax.random.split(key)
            if rollout_params.temperature <= 0:
                scaled_logits = next_logits
            else:
                scaled_logits = next_logits / max(rollout_params.temperature, 1e-6)
            filtered_logits = apply_top_k_top_p(
                scaled_logits,
                top_k=rollout_params.top_k,
                top_p=rollout_params.top_p,
            )
            if rollout_params.temperature <= 0:
                sampled_ids = jnp.argmax(filtered_logits, axis=-1).astype(jnp.int32)
            else:
                sampled_ids = jax.random.categorical(
                    sample_key,
                    filtered_logits,
                    axis=-1,
                ).astype(jnp.int32)

            next_position = position + 1
            prompt_next_ids = jnp.squeeze(
                jax.lax.dynamic_slice_in_dim(
                    prompt_matrix,
                    next_position,
                    1,
                    axis=1,
                ),
                axis=1,
            )
            current_token_ids = jnp.squeeze(current_tokens, axis=1)
            still_prefilling_prompt = next_position < prompt_lens
            next_ids = jnp.select(
                [
                    finished,
                    still_prefilling_prompt,
                ],
                [
                    current_token_ids,
                    prompt_next_ids,
                ],
                default=sampled_ids,
            )

            tokens = tokens.at[batch_indices, next_position].set(next_ids)

            can_generate = (
                (~finished)
                & (~still_prefilling_prompt)
                & (output_lens < rollout_params.max_decode_steps)
            )
            new_output_lens = output_lens + can_generate.astype(jnp.int32)
            eos_reached = can_generate & jnp.any(
                next_ids[:, None] == eos_ids_jax[None, :],
                axis=1,
            )
            prompt_finished = next_position >= prompt_lens
            decode_budget_exhausted = (
                new_output_lens >= rollout_params.max_decode_steps
            )
            no_more_room = next_position >= (max_seq_len - 1)
            finished = (
                finished
                | eos_reached
                | no_more_room
                | (prompt_finished & decode_budget_exhausted)
            )

            return (
                next_key,
                tokens,
                new_output_lens,
                finished,
                next_position,
                kv,
            )

        initial_state = (
            prng_key,
            token_matrix,
            jnp.zeros((batch_size,), dtype=jnp.int32),
            jnp.zeros((batch_size,), dtype=jnp.bool_),
            jnp.asarray(begin_position, dtype=jnp.int32),
            kv_cache,
        )
        return jax.lax.while_loop(cond_fn, body_fn, initial_state)

    with jax.set_mesh(model.mesh):
        kv = init_kv_cache(
            model.config,
            batch_size=args.batch_size,
            cache_len=max_seq_len,
            dtype=jnp.bfloat16,
        )
        logits, _ = run_prefill(
            prompt_tokens[:, :prefill_size],
            prefill_attention_mask,
            kv,
        )
        jax.block_until_ready(logits)

        kv = init_kv_cache(
            model.config,
            batch_size=args.batch_size,
            cache_len=max_seq_len,
            dtype=jnp.bfloat16,
        )
        t0 = time.perf_counter()
        logits, kv_prefilled = run_prefill(
            prompt_tokens[:, :prefill_size],
            prefill_attention_mask,
            kv,
        )
        jax.block_until_ready(logits)
        _first_leaf(kv_prefilled).block_until_ready()
        t1 = time.perf_counter()

        kv = init_kv_cache(
            model.config,
            batch_size=args.batch_size,
            cache_len=max_seq_len,
            dtype=jnp.bfloat16,
        )
        _, kv_prefilled = run_prefill(
            prompt_tokens[:, :prefill_size],
            prefill_attention_mask,
            kv,
        )
        _first_leaf(kv_prefilled).block_until_ready()
        warmup_state = run_decode(
            jax.random.key(args.seed),
            prompt_tokens,
            prompt_lengths,
            prompt_tokens,
            kv_prefilled,
        )
        _first_leaf(warmup_state).block_until_ready()

        kv = init_kv_cache(
            model.config,
            batch_size=args.batch_size,
            cache_len=max_seq_len,
            dtype=jnp.bfloat16,
        )
        _, kv_prefilled = run_prefill(
            prompt_tokens[:, :prefill_size],
            prefill_attention_mask,
            kv,
        )
        _first_leaf(kv_prefilled).block_until_ready()
        t2 = time.perf_counter()
        final_state = run_decode(
            jax.random.key(args.seed + 1),
            prompt_tokens,
            prompt_lengths,
            prompt_tokens,
            kv_prefilled,
        )
        _first_leaf(final_state).block_until_ready()
        t3 = time.perf_counter()

    final_output_lens = np.asarray(jax.device_get(final_state[2]), dtype=np.int32)
    final_position = int(jax.device_get(final_state[4]))
    decode_steps = max(final_position - begin_position, 0)

    prefill_dense_tokens = args.batch_size * prefill_size
    prefill_visible_tokens = int(np.minimum(prompt_lengths_np, prefill_size).sum())
    decode_dense_tokens = args.batch_size * decode_steps
    generated_tokens = int(final_output_lens.sum())

    prefill_elapsed = t1 - t0
    decode_elapsed = t3 - t2

    result: dict[str, int | float] = {
        "prompt_len_min": int(prompt_lengths_np.min()),
        "prompt_len_max": int(prompt_lengths_np.max()),
        "prefill_size": prefill_size,
        "begin_position": begin_position,
        "end_position": end_position,
        "decode_steps": decode_steps,
        "prefill_dense_tokens": prefill_dense_tokens,
        "prefill_visible_tokens": prefill_visible_tokens,
        "generated_tokens": generated_tokens,
        "prefill_elapsed_s": prefill_elapsed,
        "prefill_dense_tok_s": prefill_dense_tokens / prefill_elapsed,
        "prefill_visible_tok_s": prefill_visible_tokens / prefill_elapsed,
        "decode_elapsed_s": decode_elapsed,
        "decode_dense_tok_s": decode_dense_tokens / decode_elapsed,
        "generated_tok_s": generated_tokens / decode_elapsed,
    }
    return result


def print_result(name: str, result: dict[str, int | float]) -> None:
    print(f"RESULT {name}")
    for key, value in result.items():
        if isinstance(value, float):
            if key.endswith("_tok_s"):
                print(f"{key} {value:.2f}")
            else:
                print(f"{key} {value:.6f}")
        else:
            print(f"{key} {value}")


def main() -> None:
    args = parse_args()
    print(
        f"Benchmarking mode=simply_generate model_id={args.model_id} "
        f"batch_size={args.batch_size} input_len={args.input_len} "
        f"max_decode_steps={args.max_decode_steps} tp={args.tp} "
        f"attn_impl={args.attn_impl} "
        f"sequence_parallelism={args.sequence_parallelism}"
    )
    print_result("jaxformers_simply_generate", bench_simply_generate(args))


if __name__ == "__main__":
    main()
