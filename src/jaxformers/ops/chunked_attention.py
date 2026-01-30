from typing import Protocol

import jax
import jax.numpy as jnp
from absl.logging import log
from jax import P
from jaxtyping import Array, Bool, Float, PRNGKeyArray

def chunked_attention(
    query: Float[Array, "B T N H"],
    key: Float[Array, "B S K H"],
    value: Float[Array, "B S K H"],
    bias: Array | None = None,
    mask: Bool[Array, "T S"]
    | Bool[Array, "N T S"]
    | Bool[Array, "B N T S"]
    | Array
    | None = None,
    **kwargs,
):
    C = kwargs["query_chunk_size"]
    B, T, N, H = query.shape
    query = query.reshape(C, B, T//C, N, H)

    def scan_query(q_chunk, key, value):
        pass

    jax.lax.scan()
