import functools
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "input_positions",
        "token_req_indices",
        "block_tables",
        "seq_lens",
        "query_start_loc",
        "request_distribution",
    ],
    meta_fields=[],
    drop_fields=["query_start_loc_cpu", "seq_lens_cpu"],
)
@dataclass
class AttentionMetadata:
    # Flattened token positions for scheduled tokens.
    input_positions: jax.Array
    # Request index (0..num_reqs-1) for each scheduled token.
    token_req_indices: jax.Array
    # Flattened block table with shape [max_num_seqs * max_num_blocks_per_req].
    block_tables: jax.Array
    # Sequence lengths per request. Padded to max_num_seqs.
    seq_lens: jax.Array
    # Cumulative query lengths. Shape [max_num_seqs + 1].
    query_start_loc: jax.Array
    # [decode_end, prefill_end, total_num_reqs].
    request_distribution: jax.Array

    # CPU copies are convenient for scheduler bookkeeping and should stay out of traces.
    query_start_loc_cpu: Any = field(init=False, default=None)
    seq_lens_cpu: Any = field(init=False, default=None)

    @classmethod
    def empty(cls, max_num_seqs: int, max_num_blocks_per_req: int):
        seq_lens = jnp.zeros((max_num_seqs,), dtype=jnp.int32)
        query_start_loc = jnp.zeros((max_num_seqs + 1,), dtype=jnp.int32)
        block_tables = jnp.zeros(
            (max_num_seqs * max_num_blocks_per_req,), dtype=jnp.int32
        )
        md = cls(
            input_positions=jnp.zeros((0,), dtype=jnp.int32),
            token_req_indices=jnp.zeros((0,), dtype=jnp.int32),
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=jnp.zeros((3,), dtype=jnp.int32),
        )
        md.query_start_loc_cpu = query_start_loc
        md.seq_lens_cpu = seq_lens
        return md
