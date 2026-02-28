from __future__ import annotations

from dataclasses import dataclass

import jax


@dataclass(frozen=True)
class AttentionMetadata:
    kv_lens: jax.Array
    page_indices: jax.Array
    cu_q_lens: jax.Array
    num_seqs: jax.Array

