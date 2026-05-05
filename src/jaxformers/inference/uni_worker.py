from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard

from jaxformers.inference.attention_metadata import AttentionMetadata
from jaxformers.inference.bucketing import power_of_two_buckets
from jaxformers.inference.compilation_manager import get_compiled
from jaxformers.inference.input_batch import ScheduledBatch
from jaxformers.inference.ops.sample import SamplingMetadata, sample
from jaxformers.models.qwen3 import InferenceModel


@dataclass
class UniWorker:
    model: InferenceModel
    seed: int = 0
    compilation_cache: dict = field(default_factory=dict)

    def __post_init__(self):
        self.rng = jax.random.PRNGKey(self.seed)

    def _backbone(self, input_ids: jax.Array, weights, kv_cache, attn_metadata):
        hidden_states, kv_cache = self.model.forward(
            input_ids=input_ids,
            weights=weights,
            kv=kv_cache,
            attn_metadata=attn_metadata,
            return_hidden_states=True,
        )
        return hidden_states, kv_cache

    def _select_hidden_states(self, hidden_states: jax.Array, logits_indices: jax.Array):
        # hidden_states: [1, num_tokens, hidden_size]
        # Ensure the gather happens on a replicated tensor to avoid ambiguous sharding.
        hidden_states = reshard(hidden_states, P(None, None, None))
        return hidden_states[0, logits_indices]

    def _compute_logits(self, hidden_states: jax.Array, weights):
        # hidden_states: [num_reqs, hidden_size] -> logits: [num_reqs, vocab_size]
        logits = self.model.compute_logits(hidden_states[None, :, :], weights)
        return reshard(logits[0], P(None, None))

    def precompile(self, kv_cache):
        if jax.default_backend() != "tpu":
            return kv_cache

        additional_config = self.model.config["additional_config"]
        max_num_batched_token = additional_config["max_num_batched_token"]
        max_num_seqs = additional_config.get("max_num_seqs")
        if max_num_seqs is None:
            max_num_seqs = additional_config.get("max_num_request")
        if max_num_seqs is None:
            raise KeyError("Model additional_config must include `max_num_seqs`.")
        max_num_seqs = int(max_num_seqs)
        max_model_len = additional_config["max_model_len"]
        page_size = additional_config["page_size"]
        pages_per_req = (max_model_len + page_size - 1) // page_size

        weights = self.model.weights

        backbone_fn = get_compiled(
            self.compilation_cache,
            "backbone",
            self._backbone,
            donate_argnums=(2,),
        )
        select_hidden_fn = get_compiled(
            self.compilation_cache, "select_hidden_states", self._select_hidden_states
        )
        compute_logits_fn = get_compiled(
            self.compilation_cache, "compute_logits", self._compute_logits
        )
        sample_fn = get_compiled(self.compilation_cache, "sample", sample)

        token_buckets = power_of_two_buckets(max_num_batched_token)
        req_buckets = power_of_two_buckets(max_num_seqs)

        # Common (static-shape) metadata buffers.
        block_tables = jnp.arange(
            max_num_seqs * pages_per_req, dtype=jnp.int32
        )
        seq_lens = jnp.zeros((max_num_seqs,), dtype=jnp.int32).at[0].set(1)
        query_start_loc = jnp.zeros((max_num_seqs + 1,), dtype=jnp.int32).at[1:].set(1)
        request_distribution = jnp.array([0, 1, 1], dtype=jnp.int32)

        # 1) Backbone compilation over num_batched_tokens buckets.
        for num_tokens in token_buckets:
            dummy_input_ids = jnp.zeros((1, num_tokens), dtype=jnp.int32)
            attn_metadata = AttentionMetadata(
                input_positions=jnp.zeros((num_tokens,), dtype=jnp.int32),
                token_req_indices=jnp.zeros((num_tokens,), dtype=jnp.int32),
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
            )
            hidden_states, kv_cache = backbone_fn(
                dummy_input_ids, weights, kv_cache, attn_metadata
            )
            jax.tree.map(lambda x: x.block_until_ready(), (hidden_states, kv_cache))

        hidden_size = self.model.config["hidden_size"]

        # 2) Hidden-state selection compilation over (num_tokens, num_reqs) buckets.
        for num_tokens in token_buckets:
            dummy_hidden = jnp.zeros((1, num_tokens, hidden_size), dtype=jnp.bfloat16)
            for num_reqs in req_buckets:
                if num_reqs > num_tokens:
                    continue
                dummy_indices = jnp.zeros((num_reqs,), dtype=jnp.int32)
                selected = select_hidden_fn(dummy_hidden, dummy_indices)
                selected.block_until_ready()

        dummy_rng = jax.random.PRNGKey(0)

        # 3) Compute logits + sampling compilation over num_reqs buckets.
        for num_reqs in req_buckets:
            dummy_selected = jnp.zeros((num_reqs, hidden_size), dtype=jnp.bfloat16)
            logits = compute_logits_fn(dummy_selected, weights)
            logits.block_until_ready()

            greedy_md = SamplingMetadata(
                temperature=jnp.full((num_reqs,), -1.0, dtype=jnp.float32),
                top_k=jnp.zeros((num_reqs,), dtype=jnp.int32),
                top_p=jnp.ones((num_reqs,), dtype=jnp.float32),
                do_sample=False,
            )
            tokens, dummy_rng = sample_fn(dummy_rng, logits, greedy_md)
            jax.tree.map(lambda x: x.block_until_ready(), (tokens, dummy_rng))

            sample_md = SamplingMetadata(
                temperature=jnp.full((num_reqs,), 0.7, dtype=jnp.float32),
                top_k=jnp.full((num_reqs,), 20, dtype=jnp.int32),
                top_p=jnp.full((num_reqs,), 0.8, dtype=jnp.float32),
                do_sample=True,
            )
            tokens, dummy_rng = sample_fn(dummy_rng, logits, sample_md)
            jax.tree.map(lambda x: x.block_until_ready(), (tokens, dummy_rng))

        return kv_cache

    def run_backbone(self, batch: ScheduledBatch, kv_cache):
        weights = self.model.weights
        attn_impl = self.model.config["additional_config"]["attn_implementation"]
        if attn_impl == "ragged_paged_dot_product_attention" and jax.default_backend() != "tpu":
            backbone_fn = self._backbone
        else:
            backbone_fn = get_compiled(
                self.compilation_cache,
                "backbone",
                self._backbone,
                donate_argnums=(2,),
            )

        _, kv_cache = backbone_fn(batch.input_ids, weights, kv_cache, batch.attn_metadata)
        return kv_cache

    def run(self, batch: ScheduledBatch, kv_cache):
        weights = self.model.weights
        attn_impl = self.model.config["additional_config"]["attn_implementation"]
        if attn_impl == "ragged_paged_dot_product_attention" and jax.default_backend() != "tpu":
            # The reference ragged kernel uses Python control flow over dynamic metadata.
            # Keep this path eager until we move it to a fully JAX-looped implementation.
            backbone_fn = self._backbone
        else:
            backbone_fn = get_compiled(
                self.compilation_cache,
                "backbone",
                self._backbone,
                donate_argnums=(2,),
            )

        hidden_states, kv_cache = backbone_fn(
            batch.input_ids, weights, kv_cache, batch.attn_metadata
        )
        select_hidden_fn = get_compiled(
            self.compilation_cache, "select_hidden_states", self._select_hidden_states
        )
        hidden_2d = select_hidden_fn(hidden_states, batch.logits_indices)
        compute_logits_fn = get_compiled(
            self.compilation_cache, "compute_logits", self._compute_logits
        )
        logits_2d = compute_logits_fn(hidden_2d, weights)
        sample_fn = get_compiled(self.compilation_cache, "sample", sample)
        sampled_token_ids, self.rng = sample_fn(self.rng, logits_2d, batch.sampling_metadata)
        return logits_2d, sampled_token_ids, kv_cache
