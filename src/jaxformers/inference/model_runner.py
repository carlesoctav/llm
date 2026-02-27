from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from jaxformers.modeling_utils import Model
from jaxformers.models import qwen3


@dataclass(frozen=True)
class CompiledStep:
    max_tokens: int
    fn: callable


class JaxModelRunner:
    def __init__(
        self,
        *,
        model: Model,
        max_num_seqs: int,
        max_model_len: int,
        dtype: jnp.dtype,
    ) -> None:
        self.model = model
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.dtype = dtype

        self._compiled: dict[int, CompiledStep] = {}

        model_type = model.config.model_type
        if model_type != "qwen3":
            raise ValueError(f"Only model_type='qwen3' supported, got {model_type!r}")

        mesh = jax.sharding.get_mesh()
        replicated = jax.NamedSharding(mesh, P())
        rope_theta = qwen3.get_rope_theta(model.config)
        rope_sin, rope_cos = qwen3.make_rope_cache(
            max_model_len=max_model_len,
            head_dim=model.config.head_dim,
            theta=rope_theta,
            dtype=dtype,
        )
        self.rope_sin = jax.device_put(rope_sin, replicated)
        self.rope_cos = jax.device_put(rope_cos, replicated)

    def compile(self, *, max_num_batched_tokens: int, min_bucket_tokens: int = 16) -> None:
        buckets: list[int] = []
        n = min_bucket_tokens
        if n < 1:
            raise ValueError("min_bucket_tokens must be >= 1")
        while n < max_num_batched_tokens:
            buckets.append(n)
            n *= 2
        buckets.append(max_num_batched_tokens)

        for max_tokens in buckets:
            self._compiled[max_tokens] = CompiledStep(
                max_tokens=max_tokens,
                fn=self._compile_one(max_tokens=max_tokens),
            )

    def pick_bucket(self, total_q_tokens: int) -> int:
        if total_q_tokens < 0:
            raise ValueError("total_q_tokens must be >= 0")
        if total_q_tokens == 0:
            return min(self._compiled)
        want = 1 << int(math.ceil(math.log2(total_q_tokens)))
        if want in self._compiled:
            return want
        # Fallback: smallest bucket that fits.
        for k in sorted(self._compiled):
            if k >= total_q_tokens:
                return k
        raise ValueError("total_q_tokens exceeds compiled max_num_batched_tokens")

    def step(
        self,
        *,
        max_tokens: int,
        kv_cache: tuple[jax.Array, ...],
        token_ids: jax.Array,
        positions: jax.Array,
        page_ids: jax.Array,
        page_offsets: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        num_seqs: jax.Array,
        sample_mask: jax.Array,
        rng: jax.Array,
        temperature: float,
    ) -> tuple[tuple[jax.Array, ...], jax.Array, jax.Array]:
        compiled = self._compiled[max_tokens]
        return compiled.fn(
            kv_cache,
            token_ids,
            positions,
            page_ids,
            page_offsets,
            kv_lens,
            page_indices,
            cu_q_lens,
            num_seqs,
            sample_mask,
            rng,
            temperature,
        )

    def _compile_one(self, *, max_tokens: int):
        model = self.model
        max_num_seqs = self.max_num_seqs
        dtype = self.dtype
        mesh = jax.sharding.get_mesh()

        in_specs = (
            P(None, "tp", None),  # q
            P(None, None, "tp", None),  # kv_pages
            P(),  # kv_lens
            P(),  # page_indices
            P(),  # cu_q_lens
            P(),  # num_seqs
        )
        out_specs = P(None, "tp", None)
        sm_scale = 1.0 / math.sqrt(model.config.head_dim)

        def _ragged_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
            return qwen3.ragged_paged_attention(
                q,
                kv_pages,
                kv_lens,
                page_indices,
                cu_q_lens,
                num_seqs,
                sm_scale=sm_scale,
            )

        ragged_attention = jax.jit(
            jax.shard_map(
                _ragged_attention,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            )
        )

        def _step(
            kv_cache: tuple[jax.Array, ...],
            token_ids: jax.Array,  # [max_tokens]
            positions: jax.Array,  # [max_tokens]
            page_ids: jax.Array,  # [max_tokens]
            page_offsets: jax.Array,  # [max_tokens]
            kv_lens: jax.Array,  # [max_num_seqs]
            page_indices: jax.Array,  # [max_num_seqs, pages_per_seq]
            cu_q_lens: jax.Array,  # [max_num_seqs + 1]
            num_seqs: jax.Array,  # [1]
            sample_mask: jax.Array,  # [max_num_seqs]
            rng: jax.Array,
            temperature: float,
        ):
            hidden, new_kv_cache = qwen3.forward_inference_ragged_paged(
                model.config,
                model.weights,
                kv_cache,
                token_ids,
                positions,
                page_ids,
                page_offsets,
                kv_lens,
                page_indices,
                cu_q_lens,
                num_seqs,
                self.rope_sin,
                self.rope_cos,
                dtype=dtype,
                ragged_attention=ragged_attention,
            )

            q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
            has_q = q_lens > 0
            last_idx = jnp.where(has_q, cu_q_lens[1:] - 1, 0)
            last_hidden = hidden[last_idx]  # [max_num_seqs, hidden]

            logits = model.unembed(
                model.weights,
                last_hidden[:, None, :],
                dtype=jnp.float32,
                compute_dtype=dtype,
            )[:, 0, :]

            if temperature == 0.0:
                sampled = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            else:
                sampled = jax.random.categorical(
                    rng, logits / temperature, axis=-1
                ).astype(jnp.int32)

            next_token_ids = jnp.where(sample_mask, sampled, jnp.zeros_like(sampled))
            next_rng = jax.random.split(rng, 2)[1]
            return new_kv_cache, next_token_ids, next_rng

        return jax.jit(
            _step,
            donate_argnums=(0,),
            static_argnames=("temperature",),
        )
