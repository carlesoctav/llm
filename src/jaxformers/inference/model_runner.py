from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from jaxformers.modeling_utils import Model
from jaxformers.models import qwen3


@dataclass(frozen=True)
class CompiledBackbone:
    max_tokens: int
    fn: callable


@dataclass(frozen=True)
class CompiledSelect:
    padded_num_seqs: int
    fn: callable


@dataclass(frozen=True)
class CompiledLogits:
    padded_num_seqs: int
    fn: callable


@dataclass(frozen=True)
class CompiledSample:
    padded_num_seqs: int
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

        self._backbone: dict[int, CompiledBackbone] = {}
        self._select: dict[int, CompiledSelect] = {}
        self._logits: dict[int, CompiledLogits] = {}
        self._sample: dict[tuple[int, bool], CompiledSample] = {}

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

    def compile(
        self,
        *,
        max_num_batched_tokens: int,
        min_token_bucket: int = 16,
        min_seq_bucket: int = 8,
    ) -> None:
        token_buckets: list[int] = []
        n = int(min_token_bucket)
        if n < 1:
            raise ValueError("min_token_bucket must be >= 1")
        while n < max_num_batched_tokens:
            token_buckets.append(n)
            n *= 2
        token_buckets.append(int(max_num_batched_tokens))

        for max_tokens in token_buckets:
            self._backbone[max_tokens] = CompiledBackbone(
                max_tokens=max_tokens,
                fn=self._compile_backbone(max_tokens=max_tokens),
            )

        seq_buckets: list[int] = []
        m = int(min_seq_bucket)
        if m < 1:
            raise ValueError("min_seq_bucket must be >= 1")
        while m < self.max_num_seqs:
            seq_buckets.append(m)
            m *= 2
        seq_buckets.append(self.max_num_seqs)

        for padded_num_seqs in seq_buckets:
            self._select[padded_num_seqs] = CompiledSelect(
                padded_num_seqs=padded_num_seqs,
                fn=self._compile_select(padded_num_seqs=padded_num_seqs),
            )
            self._logits[padded_num_seqs] = CompiledLogits(
                padded_num_seqs=padded_num_seqs,
                fn=self._compile_logits(padded_num_seqs=padded_num_seqs),
            )
            for do_sampling in (False, True):
                self._sample[(padded_num_seqs, do_sampling)] = CompiledSample(
                    padded_num_seqs=padded_num_seqs,
                    fn=self._compile_sample(
                        padded_num_seqs=padded_num_seqs, do_sampling=do_sampling
                    ),
                )

    def pick_token_bucket(self, total_q_tokens: int) -> int:
        if total_q_tokens < 0:
            raise ValueError("total_q_tokens must be >= 0")
        if total_q_tokens == 0:
            return min(self._backbone)
        want = 1 << int(math.ceil(math.log2(total_q_tokens)))
        if want in self._backbone:
            return want
        for k in sorted(self._backbone):
            if k >= total_q_tokens:
                return k
        raise ValueError("total_q_tokens exceeds compiled max_num_batched_tokens")

    def pick_seq_bucket(self, num_sampled: int) -> int:
        if num_sampled < 1:
            raise ValueError("num_sampled must be >= 1")
        want = 1 << int(math.ceil(math.log2(num_sampled)))
        if want in self._select:
            return want
        for k in sorted(self._select):
            if k >= num_sampled:
                return k
        raise ValueError("num_sampled exceeds compiled max_num_seqs")

    def backbone(
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
    ) -> tuple[tuple[jax.Array, ...], jax.Array]:
        compiled = self._backbone[max_tokens]
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
        )

    def select(
        self,
        *,
        padded_num_seqs: int,
        last_hidden: jax.Array,
        indices: jax.Array,
    ) -> jax.Array:
        compiled = self._select[padded_num_seqs]
        return compiled.fn(last_hidden, indices)

    def compute_logits(
        self, *, padded_num_seqs: int, hidden: jax.Array
    ) -> jax.Array:
        compiled = self._logits[padded_num_seqs]
        return compiled.fn(hidden)

    def sample(
        self,
        *,
        padded_num_seqs: int,
        rng: jax.Array,
        logits: jax.Array,
        temperature: float,
        do_sampling: bool,
    ) -> jax.Array:
        compiled = self._sample[(padded_num_seqs, do_sampling)]
        return compiled.fn(rng, logits, temperature)

    def _compile_backbone(self, *, max_tokens: int):
        model = self.model
        dtype = self.dtype
        max_num_seqs = self.max_num_seqs
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

        def _backbone(
            kv_cache: tuple[jax.Array, ...],
            token_ids: jax.Array,  # [max_tokens]
            positions: jax.Array,  # [max_tokens]
            page_ids: jax.Array,  # [max_tokens]
            page_offsets: jax.Array,  # [max_tokens]
            kv_lens: jax.Array,  # [max_num_seqs]
            page_indices: jax.Array,  # [max_num_seqs, pages_per_seq]
            cu_q_lens: jax.Array,  # [max_num_seqs + 1]
            num_seqs: jax.Array,  # [1]
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
            return new_kv_cache, last_hidden

        return jax.jit(
            _backbone,
            donate_argnums=(0,),
        )

    def _compile_select(self, *, padded_num_seqs: int):
        def _select(last_hidden: jax.Array, indices: jax.Array) -> jax.Array:
            safe = jnp.where(indices >= 0, indices, 0)
            return last_hidden[safe]

        return jax.jit(_select)

    def _compile_logits(self, *, padded_num_seqs: int):
        model = self.model
        dtype = self.dtype

        def _logits(hidden: jax.Array) -> jax.Array:
            logits = model.unembed(
                model.weights,
                hidden[:, None, :],
                dtype=jnp.float32,
                compute_dtype=dtype,
            )[:, 0, :]
            return logits

        return jax.jit(_logits)

    def _compile_sample(self, *, padded_num_seqs: int, do_sampling: bool):
        def _sample(rng: jax.Array, logits: jax.Array, temperature: float) -> jax.Array:
            if do_sampling:
                sampled = jax.random.categorical(
                    rng, logits / temperature, axis=-1
                ).astype(jnp.int32)
                return sampled
            return jnp.argmax(logits, axis=-1).astype(jnp.int32)

        return jax.jit(_sample)
