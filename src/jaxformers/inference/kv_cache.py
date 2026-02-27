from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PagedKVCacheConfig:
    max_model_len: int
    max_num_seqs: int
    page_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int

    @property
    def pages_per_seq(self) -> int:
        return int(math.ceil(self.max_model_len / self.page_size))

    @property
    def num_pages(self) -> int:
        return self.max_num_seqs * self.pages_per_seq

    @property
    def num_combined_kv_heads(self) -> int:
        return self.num_kv_heads * 2


def make_page_indices(cfg: PagedKVCacheConfig) -> jax.Array:
    pages_per_seq = cfg.pages_per_seq
    base = jnp.arange(cfg.max_num_seqs, dtype=jnp.int32)[:, None] * pages_per_seq
    offsets = jnp.arange(pages_per_seq, dtype=jnp.int32)[None, :]
    return base + offsets


def make_kv_cache(cfg: PagedKVCacheConfig, *, dtype: jnp.dtype) -> tuple[jax.Array, ...]:
    # ragged_paged_attention expects kv_pages:
    #   [num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_shape = (cfg.num_pages, cfg.page_size, cfg.num_combined_kv_heads, cfg.head_dim)

    mesh = jax.sharding.get_mesh()
    kv_sharding = jax.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(None, None, "tp", None),
    )

    layers: list[jax.Array] = []
    for _ in range(cfg.num_layers):
        layers.append(jax.device_put(jnp.zeros(kv_shape, dtype=dtype), kv_sharding))
    return tuple(layers)
