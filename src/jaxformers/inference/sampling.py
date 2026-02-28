from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int
    temperature: float
    top_p: float = 1.0
    top_k: int = 0
    ignore_eos: bool = True


@dataclass(frozen=True)
class TPUSamplingMetadata:
    temperature: jax.Array
    top_p: jax.Array
    top_k: jax.Array
    do_sampling: bool

    @classmethod
    def from_input_batch(
        cls,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        padded_num_seqs: int,
        sharding,
    ) -> "TPUSamplingMetadata":
        temp = jnp.full((padded_num_seqs,), float(temperature), dtype=jnp.float32)
        top_p_arr = jnp.full((padded_num_seqs,), float(top_p), dtype=jnp.float32)
        top_k_arr = jnp.full((padded_num_seqs,), int(top_k), dtype=jnp.int32)

        temp = jax.device_put(temp, sharding)
        top_p_arr = jax.device_put(top_p_arr, sharding)
        top_k_arr = jax.device_put(top_k_arr, sharding)

        do_sampling = float(temperature) != 0.0
        return cls(
            temperature=temp,
            top_p=top_p_arr,
            top_k=top_k_arr,
            do_sampling=do_sampling,
        )

