from __future__ import annotations

import dataclasses as dc
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from jaxformers.attention_utils import init_kv_cache
from jaxformers.modeling_utils import logical_to_physical, Model
from jaxformers.rollout_utils import (
    DecodingSchedule,
    gather_token_logprobs,
    left_truncate,
    make_attention_mask,
    pad_sequences,
    sample_from_logits,
)


@dc.dataclass(frozen=True)
class RolloutParams:
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    max_decode_steps: int = 256
    decode_chunk_size: int | None = None
    max_seq_len: int = 1025
    max_input_len: int | None = None
    num_samples_per_example: int = 1
    min_prefill_size: int = 256
    prefill_size: int | None = None
    stop_token_ids: tuple[int, ...] | list[int] = ()

    def get_decoding_schedule(
        self,
        *,
        min_input_length: int,
        max_input_length: int,
    ) -> DecodingSchedule:
        prefill_size = self.prefill_size
        if prefill_size is None:
            prefill_size = max(
                int(np.exp2(np.ceil(np.log2(max(min_input_length, 1))))),
                self.min_prefill_size,
            )

        begin_position = min(prefill_size, max(min_input_length - 1, 0))
        end_position = min(
            self.max_seq_len - 1,
            int(max_input_length) + self.max_decode_steps - 1,
        )
        chunk_size = self.decode_chunk_size
        if chunk_size is None or chunk_size <= 0:
            chunk_size = self.max_decode_steps

        return DecodingSchedule(
            prefill_size=prefill_size,
            begin_position=begin_position,
            end_position=end_position,
            chunk_size=chunk_size,
        )


@dc.dataclass(frozen=True)
class SimpleRolloutSample:
    prompt_index: int
    sample_index: int
    name: str
    prompt_text: str
    output_text: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    output_token_logprobs: list[float]
    is_truncated: bool


@dc.dataclass(frozen=True)
class SimpleRolloutBatch:
    token_ids: np.ndarray
    attention_mask: np.ndarray
    generation_mask: np.ndarray
    logprobs: np.ndarray
    prompt_lengths: np.ndarray
    output_lengths: np.ndarray
    prompt_indices: np.ndarray
    samples: list[SimpleRolloutSample]


def _first_available_id(*values: int | None) -> int:
    for value in values:
        if value is not None:
            return int(value)
    return 0


class SimpleRolloutEngine:
    """Dense prefill+decode rollout with a simply-style KV cache."""

    def __init__(
        self,
        *,
        model: Model,
        rollout_params: RolloutParams,
        params=None,
        forward_dtype=None,
    ) -> None:
        self.model = model
        self.rollout_params = rollout_params
        self.params = params
        self.forward_dtype = (
            forward_dtype
            if forward_dtype is not None
            else getattr(model.weights[model.lm_head_key], "dtype", jnp.bfloat16)
        )

        tokenizer = model.tokenizer
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        end_of_turn_id = None
        try:
            end_of_turn_ids = tokenizer.encode("<end_of_turn>", add_special_tokens=False)
            if len(end_of_turn_ids) == 1:
                end_of_turn_id = int(end_of_turn_ids[0])
        except Exception:
            end_of_turn_id = None

        self.pad_id = _first_available_id(pad_token_id, eos_token_id, bos_token_id, 0)
        self.bos_id = _first_available_id(bos_token_id, eos_token_id, self.pad_id)
        stop_ids = {
            _first_available_id(eos_token_id, pad_token_id, self.pad_id),
            _first_available_id(
                getattr(tokenizer, "sep_token_id", None),
                eos_token_id,
                self.pad_id,
            ),
        }
        if end_of_turn_id is not None:
            stop_ids.add(end_of_turn_id)
        stop_ids.update(int(token_id) for token_id in self.rollout_params.stop_token_ids)
        self.eos_ids = tuple(sorted(stop_ids))

        def prefill_fn(params, token_ids, attention_mask, kv):
            hidden_states, kv = self.model.forward(
                params,
                input_ids=token_ids,
                attention_mask=attention_mask,
                dtype=self.forward_dtype,
                kv=kv,
                pos=0,
            )
            logits = self.model.unembed(
                params,
                hidden_states,
                dtype=jnp.float32,
            )
            logits = jax.reshard(
                logits,
                logical_to_physical(
                    ("batch", "context", "none"),
                    self.model.config.sharding_rules,
                ),
            )
            return logits, kv

        self._prefill_fn = jax.jit(prefill_fn)

        def decode_step(
            params,
            prng_key,
            prompt_tokens,
            tokens,
            attention_mask,
            generation_mask,
            token_logprobs,
            prompt_lengths,
            output_lens,
            finished,
            position,
            kv,
        ):
            current_tokens = jax.lax.dynamic_slice_in_dim(
                tokens,
                position,
                1,
                axis=1,
            )
            hidden_states, kv = self.model.forward(
                params,
                input_ids=current_tokens,
                dtype=self.forward_dtype,
                kv=kv,
                pos=position,
            )
            logits = self.model.unembed(
                params,
                hidden_states,
                dtype=jnp.float32,
            )
            logits = jax.reshard(
                logits,
                logical_to_physical(
                    ("batch", "context", "none"),
                    self.model.config.sharding_rules,
                ),
            )

            batch_indices = jnp.arange(tokens.shape[0], dtype=jnp.int32)
            next_logits = logits[:, 0, :]

            sample_key, next_key = jax.random.split(prng_key)
            sampled_ids, sampled_logprobs = sample_from_logits(
                sample_key,
                next_logits,
                temperature=self.rollout_params.temperature,
                top_k=self.rollout_params.top_k,
                top_p=self.rollout_params.top_p,
            )

            next_position = position + 1
            safe_write_positions = jnp.full(
                (tokens.shape[0],),
                jnp.minimum(next_position, tokens.shape[1] - 1),
                dtype=jnp.int32,
            )
            prompt_next_ids = jnp.take_along_axis(
                prompt_tokens,
                safe_write_positions[:, None],
                axis=1,
            ).squeeze(-1)
            prompt_next_logprobs = gather_token_logprobs(next_logits, prompt_next_ids)
            current_token_ids = jnp.squeeze(current_tokens, axis=1)
            current_token_logprobs = gather_token_logprobs(
                next_logits,
                current_token_ids,
            )

            still_prefilling_prompt = next_position < prompt_lengths
            has_room = next_position < tokens.shape[1]
            can_generate = (
                (~still_prefilling_prompt)
                & (~finished)
                & has_room
                & (output_lens < self.rollout_params.max_decode_steps)
            )
            should_write = has_room

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
            next_logprobs = jnp.select(
                [
                    finished,
                    still_prefilling_prompt,
                ],
                [
                    current_token_logprobs,
                    prompt_next_logprobs,
                ],
                default=sampled_logprobs,
            )

            existing_tokens = tokens[batch_indices, safe_write_positions]
            tokens = tokens.at[batch_indices, safe_write_positions].set(
                jnp.where(should_write, next_ids, existing_tokens)
            )

            attention_mask = attention_mask.at[batch_indices, safe_write_positions].set(
                should_write
            )
            generation_mask = generation_mask.at[
                batch_indices, safe_write_positions
            ].set(can_generate)

            existing_logprobs = token_logprobs[batch_indices, safe_write_positions]
            token_logprobs = token_logprobs.at[
                batch_indices, safe_write_positions
            ].set(jnp.where(should_write, next_logprobs, existing_logprobs))

            new_output_lens = output_lens + can_generate.astype(jnp.int32)
            eos_reached = can_generate & jnp.any(
                next_ids[:, None] == jnp.asarray(self.eos_ids)[None, :],
                axis=1,
            )
            decode_budget_exhausted = new_output_lens >= self.rollout_params.max_decode_steps
            new_position = jnp.minimum(next_position, tokens.shape[1] - 1)
            no_more_room = new_position >= (tokens.shape[1] - 1)
            prompt_finished = next_position >= prompt_lengths
            finished = (
                finished
                | eos_reached
                | no_more_room
                | (prompt_finished & decode_budget_exhausted)
            )

            return (
                next_key,
                tokens,
                attention_mask,
                generation_mask,
                token_logprobs,
                new_output_lens,
                finished,
                new_position,
                kv,
            )

        self._decode_step = jax.jit(decode_step)

    def set_params(self, params) -> None:
        if params is None:
            self.params = None
            return

        def _reshard_like(param_leaf, template_leaf):
            sharding = getattr(template_leaf, "sharding", None)
            if sharding is None:
                return param_leaf
            try:
                return jax.device_put(param_leaf, sharding)
            except Exception:
                return param_leaf

        self.params = jtu.tree_map(_reshard_like, params, self.model.weights)

    def _normalize_input_ids(self, input_ids: Sequence[int]) -> list[int]:
        max_input_len = self.rollout_params.max_input_len
        if max_input_len is None:
            max_input_len = max(self.rollout_params.max_seq_len - 1, 1)
        else:
            max_input_len = min(max_input_len, max(self.rollout_params.max_seq_len - 1, 1))

        normalized = left_truncate(list(input_ids), max_input_len)
        if not normalized:
            normalized = [self.bos_id]
        return normalized

    def generate(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        prng_key: int | jax.Array | None = None,
        params=None,
    ) -> SimpleRolloutBatch:
        with jax.set_mesh(self.model.mesh) if self.model.mesh else nullcontext():
            return self._generate(examples, prng_key=prng_key, params=params)

    def _generate(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        prng_key: int | jax.Array | None = None,
        params=None,
    ) -> SimpleRolloutBatch:
        if params is None:
            params = self.params
        if params is None:
            raise ValueError("SimpleRolloutEngine.generate requires params.")

        if prng_key is None:
            prng_key = jax.random.key(seed=0)
        elif isinstance(prng_key, int):
            prng_key = jax.random.key(seed=prng_key)

        repeated_examples: list[Mapping[str, Any]] = []
        prompt_indices: list[int] = []
        for prompt_index, example in enumerate(examples):
            for _ in range(self.rollout_params.num_samples_per_example):
                repeated_examples.append(example)
                prompt_indices.append(prompt_index)

        if not repeated_examples:
            empty = np.zeros((0, self.rollout_params.max_seq_len), dtype=np.int32)
            empty_mask = np.zeros_like(empty, dtype=bool)
            empty_logprobs = np.zeros_like(empty, dtype=np.float32)
            empty_lengths = np.zeros((0,), dtype=np.int32)
            return SimpleRolloutBatch(
                token_ids=empty,
                attention_mask=empty_mask,
                generation_mask=empty_mask,
                logprobs=empty_logprobs,
                prompt_lengths=empty_lengths,
                output_lengths=empty_lengths,
                prompt_indices=empty_lengths,
                samples=[],
            )

        prompt_token_ids = [
            self._normalize_input_ids(example["input_ids"])
            for example in repeated_examples
        ]
        prompt_lengths = np.asarray(
            [len(token_ids) for token_ids in prompt_token_ids],
            dtype=np.int32,
        )

        schedule = self.rollout_params.get_decoding_schedule(
            min_input_length=int(prompt_lengths.min()),
            max_input_length=int(prompt_lengths.max()),
        )

        total_length = self.rollout_params.max_seq_len
        prefill_size = max(1, min(int(schedule.prefill_size), total_length))
        begin_position = min(int(schedule.begin_position), total_length - 1)
        end_position = min(int(schedule.end_position), total_length - 1)

        prompt_tokens = pad_sequences(
            prompt_token_ids,
            pad_id=self.pad_id,
            length=total_length,
        )
        visible_lens = np.minimum(prompt_lengths, prefill_size).astype(np.int32)

        tokens = jnp.asarray(prompt_tokens, dtype=jnp.int32)
        attention_mask = jnp.asarray(
            make_attention_mask(visible_lens, total_length),
            dtype=jnp.bool_,
        )
        generation_mask = jnp.zeros((len(repeated_examples), total_length), dtype=jnp.bool_)
        token_logprobs = jnp.zeros((len(repeated_examples), total_length), dtype=jnp.float32)
        kv = init_kv_cache(
            self.model.config,
            batch_size=len(repeated_examples),
            cache_len=total_length,
            dtype=self.forward_dtype,
        )

        if prefill_size > 0:
            prefill_logits, kv = self._prefill_fn(
                params,
                tokens[:, :prefill_size],
                attention_mask[:, :prefill_size].astype(jnp.int32),
                kv,
            )
            prefill_targets = tokens[:, 1:prefill_size]
            prefill_token_logprobs = gather_token_logprobs(
                prefill_logits[:, :-1, :],
                prefill_targets,
            )
            prefill_positions = jnp.arange(max(prefill_size - 1, 0))[None, :] + 1
            prefill_mask = prefill_positions < visible_lens[:, None]
            token_logprobs = token_logprobs.at[:, 1:prefill_size].set(
                jnp.where(prefill_mask, prefill_token_logprobs, 0.0)
            )

        output_lens = np.zeros_like(prompt_lengths)
        finished = np.zeros_like(prompt_lengths, dtype=bool)

        state = (
            jnp.asarray(prng_key),
            tokens,
            attention_mask,
            generation_mask,
            token_logprobs,
            jnp.asarray(output_lens, dtype=jnp.int32),
            jnp.asarray(finished, dtype=jnp.bool_),
            jnp.asarray(begin_position, dtype=jnp.int32),
            kv,
        )

        prompt_tokens_jax = jnp.asarray(prompt_tokens, dtype=jnp.int32)
        prompt_lengths_jax = jnp.asarray(prompt_lengths, dtype=jnp.int32)
        max_steps = max(end_position - begin_position, 0)
        for _ in range(max_steps):
            state = self._decode_step(
                params,
                state[0],
                prompt_tokens_jax,
                state[1],
                state[2],
                state[3],
                state[4],
                prompt_lengths_jax,
                state[5],
                state[6],
                state[7],
                state[8],
            )

        _, tokens, attention_mask, generation_mask, token_logprobs, output_lens, _, _, _ = state
        token_ids = np.asarray(jax.device_get(tokens), dtype=np.int32)
        attention_mask_np = np.asarray(jax.device_get(attention_mask), dtype=bool)
        generation_mask_np = np.asarray(jax.device_get(generation_mask), dtype=bool)
        token_logprobs_np = np.asarray(jax.device_get(token_logprobs), dtype=np.float32)
        output_lens_np = np.asarray(jax.device_get(output_lens), dtype=np.int32)
        full_lens_np = attention_mask_np.astype(np.int32).sum(axis=1)

        samples: list[SimpleRolloutSample] = []
        for sample_index, example in enumerate(repeated_examples):
            prompt_len = int(prompt_lengths[sample_index])
            full_len = int(full_lens_np[sample_index])
            output_ids = token_ids[sample_index, prompt_len:full_len].tolist()
            output_text = self.model.tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
            )
            output_token_logprobs = token_logprobs_np[
                sample_index, prompt_len:full_len
            ].tolist()
            is_truncated = bool(
                output_lens_np[sample_index] >= self.rollout_params.max_decode_steps
            )

            samples.append(
                SimpleRolloutSample(
                    prompt_index=prompt_indices[sample_index],
                    sample_index=sample_index,
                    name=str(example["name"]),
                    prompt_text=str(example["content"]),
                    output_text=output_text,
                    prompt_token_ids=prompt_token_ids[sample_index],
                    output_token_ids=output_ids,
                    output_token_logprobs=output_token_logprobs,
                    is_truncated=is_truncated,
                )
            )

        return SimpleRolloutBatch(
            token_ids=token_ids,
            attention_mask=attention_mask_np,
            generation_mask=generation_mask_np,
            logprobs=token_logprobs_np,
            prompt_lengths=prompt_lengths,
            output_lengths=output_lens_np,
            prompt_indices=np.asarray(prompt_indices, dtype=np.int32),
            samples=samples,
        )


def make(
    *,
    model: Model,
    rollout_config: dict,
    params=None,
    forward_dtype=None,
) -> SimpleRolloutEngine:
    return SimpleRolloutEngine(
        model=model,
        rollout_params=RolloutParams(**rollout_config),
        params=params,
        forward_dtype=forward_dtype,
    )
