from __future__ import annotations

import dataclasses as dc

import jax
import jax.numpy as jnp
import numpy as np


@dc.dataclass(frozen=True)
class DecodingSchedule:
    """Dense decoding schedule mirroring the simply prefill/decode split."""

    prefill_size: int
    begin_position: int
    end_position: int
    chunk_size: int

    def get_next_length(self, cur_position: int) -> int:
        if cur_position < self.begin_position:
            return self.begin_position
        step_multiple = ((cur_position - self.prefill_size) // self.chunk_size) + 1
        next_position = self.prefill_size + self.chunk_size * step_multiple
        return min(next_position, self.end_position)


def left_truncate(tokens: list[int], max_length: int | None) -> list[int]:
    if max_length is None or len(tokens) <= max_length:
        return tokens
    return tokens[-max_length:]


def pad_sequences(
    sequences: list[list[int]],
    *,
    pad_id: int,
    length: int | None = None,
    dtype: np.dtype = np.int32,
) -> np.ndarray:
    if not sequences:
        target_length = length or 0
        return np.zeros((0, target_length), dtype=dtype)

    target_length = length or max(len(seq) for seq in sequences)
    output = np.full((len(sequences), target_length), pad_id, dtype=dtype)
    for index, seq in enumerate(sequences):
        seq_len = min(len(seq), target_length)
        output[index, :seq_len] = np.asarray(seq[:seq_len], dtype=dtype)
    return output


def make_attention_mask(lengths: np.ndarray, total_length: int) -> np.ndarray:
    positions = np.arange(total_length, dtype=np.int32)[None, :]
    return positions < np.asarray(lengths, dtype=np.int32)[:, None]


def gather_token_logprobs(
    logits: jax.Array,
    token_ids: jax.Array,
) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    gathered = jnp.take_along_axis(log_probs, token_ids[..., None], axis=-1)
    return jnp.squeeze(gathered, axis=-1)


def apply_top_k_top_p(
    logits: jax.Array,
    *,
    top_k: int = -1,
    top_p: float = 1.0,
) -> jax.Array:
    filtered = logits

    if top_k > 0 and top_k < logits.shape[-1]:
        top_k_values = jax.lax.top_k(filtered, top_k)[0][..., -1, None]
        filtered = jnp.where(filtered < top_k_values, -jnp.inf, filtered)

    if top_p < 1.0:
        sorted_indices = jnp.argsort(filtered, axis=-1)[:, ::-1]
        sorted_logits = jnp.take_along_axis(filtered, sorted_indices, axis=-1)
        sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
        cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask = sorted_mask.at[:, 0].set(False)
        sorted_logits = jnp.where(sorted_mask, -jnp.inf, sorted_logits)

        batch_indices = jnp.arange(filtered.shape[0])[:, None]
        restored = jnp.full_like(filtered, -jnp.inf)
        filtered = restored.at[batch_indices, sorted_indices].set(sorted_logits)

    return filtered


def sample_from_logits(
    prng_key: jax.Array,
    logits: jax.Array,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[jax.Array, jax.Array]:
    if temperature <= 0:
        scaled_logits = logits
    else:
        scaled_logits = logits / max(temperature, 1e-6)

    filtered_logits = apply_top_k_top_p(
        scaled_logits,
        top_k=top_k,
        top_p=top_p,
    )

    if temperature <= 0:
        token_ids = jnp.argmax(filtered_logits, axis=-1).astype(jnp.int32)
    else:
        token_ids = jax.random.categorical(
            prng_key,
            filtered_logits,
            axis=-1,
        ).astype(jnp.int32)

    token_logprobs = gather_token_logprobs(filtered_logits, token_ids)
    return token_ids, token_logprobs.astype(jnp.float32)
