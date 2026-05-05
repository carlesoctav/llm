from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp


_SAMPLING_EPS = 1e-5
_NEG_INF = -1e12


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["temperature", "top_k", "top_p"],
    meta_fields=["do_sample"],
)
@dataclass
class SamplingMetadata:
    temperature: jax.Array
    top_k: jax.Array
    top_p: jax.Array
    do_sample: bool = False


def _apply_top_k_row(logits_row: jax.Array, top_k: jax.Array) -> jax.Array:
    vocab_size = logits_row.shape[-1]
    k = jnp.clip(top_k, 0, vocab_size)

    def _mask():
        kth = jnp.sort(logits_row)[-k]
        return jnp.where(logits_row < kth, _NEG_INF, logits_row)

    return jax.lax.cond(k > 0, _mask, lambda: logits_row)


def _apply_top_p_row(logits_row: jax.Array, top_p: jax.Array) -> jax.Array:
    p = jnp.clip(top_p, 0.0, 1.0)

    def _mask():
        sorted_idx = jnp.argsort(logits_row)[::-1]
        sorted_logits = logits_row[sorted_idx]
        sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
        cdf = jnp.cumsum(sorted_probs, axis=-1)
        drop_mask = cdf > p
        drop_mask = drop_mask.at[0].set(False)
        kept_sorted_logits = jnp.where(drop_mask, _NEG_INF, sorted_logits)
        out = jnp.full_like(logits_row, _NEG_INF)
        return out.at[sorted_idx].set(kept_sorted_logits)

    return jax.lax.cond(p < 1.0, _mask, lambda: logits_row)


def sample(
    rng: jax.Array,
    logits: jax.Array,
    sampling_metadata: SamplingMetadata,
) -> tuple[jax.Array, jax.Array]:
    # logits: [num_reqs, vocab_size]
    greedy_tokens = jnp.argmax(logits, axis=-1)
    if not sampling_metadata.do_sample:
        return greedy_tokens, rng

    logits = logits.astype(jnp.float32)
    logits = jax.vmap(_apply_top_k_row)(logits, sampling_metadata.top_k)
    logits = jax.vmap(_apply_top_p_row)(logits, sampling_metadata.top_p)

    temperatures = sampling_metadata.temperature.astype(logits.dtype)
    temperatures_exp = jnp.expand_dims(jnp.maximum(temperatures, _SAMPLING_EPS), axis=-1)
    logits = logits / temperatures_exp

    rng, sample_rng = jax.random.split(rng)
    sampled_tokens = jax.random.categorical(sample_rng, logits, axis=-1)
    next_tokens = jnp.where(temperatures < _SAMPLING_EPS, greedy_tokens, sampled_tokens)
    return next_tokens.astype(jnp.int32), rng
