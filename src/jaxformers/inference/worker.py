from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxformers.inference.kv_cache import PagedKVCacheConfig, make_kv_cache, make_page_indices
from jaxformers.inference.model_runner import JaxModelRunner
from jaxformers.inference.scheduler import Request, Scheduler, Sequence
from jaxformers.modeling_utils import Model


@dataclass(frozen=True)
class EngineConfig:
    max_num_batched_tokens: int
    max_num_seqs: int
    max_model_len: int
    page_size: int
    dtype: jnp.dtype
    seed: int


class JaxWorker:
    def __init__(self, *, model: Model, config: EngineConfig) -> None:
        self.model = model
        self.config = config

        self.scheduler = Scheduler(
            max_num_seqs=config.max_num_seqs,
            max_model_len=config.max_model_len,
            max_num_batched_tokens=config.max_num_batched_tokens,
        )

        cfg = model.config
        kv_cfg = PagedKVCacheConfig(
            max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs,
            page_size=config.page_size,
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
        )
        self.kv_cfg = kv_cfg

        self.page_indices = make_page_indices(kv_cfg)
        self.kv_cache = make_kv_cache(kv_cfg, dtype=config.dtype)

        self.runner = JaxModelRunner(
            model=model,
            max_num_seqs=config.max_num_seqs,
            dtype=config.dtype,
        )
        self.runner.compile(max_num_batched_tokens=config.max_num_batched_tokens)

        self.rng = jax.random.PRNGKey(config.seed)

    def add_request(self, request: Request) -> None:
        self.scheduler.add_request(request)

    def run(self, *, temperature: float) -> dict[int, Sequence]:
        finished: dict[int, Sequence] = {}
        while not self.scheduler.is_finished():
            step = self.scheduler.schedule()
            total_q_tokens = len(step.token_ids)
            if total_q_tokens == 0:
                seqs = self.scheduler.drain_finished()
                for seq in seqs:
                    finished[seq.request.request_id] = seq
                continue

            bucket_tokens = self.runner.pick_bucket(total_q_tokens)

            token_ids = jnp.zeros((bucket_tokens,), dtype=jnp.int32)
            positions = jnp.zeros((bucket_tokens,), dtype=jnp.int32)
            page_ids = jnp.full((bucket_tokens,), -1, dtype=jnp.int32)
            page_offsets = jnp.full((bucket_tokens,), -1, dtype=jnp.int32)

            packed_slots = [
                slot_id
                for slot_id in range(self.config.max_num_seqs)
                if step.q_lens[slot_id] > 0
            ]
            num_packed = len(packed_slots)
            if num_packed == 0:
                raise RuntimeError("total_q_tokens > 0 but no packed slots found")

            if total_q_tokens > 0:
                token_ids = token_ids.at[:total_q_tokens].set(
                    jnp.array(step.token_ids, dtype=jnp.int32)
                )
                positions = positions.at[:total_q_tokens].set(
                    jnp.array(step.positions, dtype=jnp.int32)
                )

                slot_ids: list[int] = []
                for slot_id in range(self.config.max_num_seqs):
                    slot_ids.extend([slot_id for _ in range(step.q_lens[slot_id])])
                if len(slot_ids) != total_q_tokens:
                    raise RuntimeError("slot_ids length mismatch")
                slot_ids_arr = jnp.array(slot_ids, dtype=jnp.int32)
                pos_arr = positions[:total_q_tokens]
                pages_per_seq = self.kv_cfg.pages_per_seq
                page_ids_live = slot_ids_arr * pages_per_seq + (pos_arr // self.config.page_size)
                page_offsets_live = pos_arr % self.config.page_size
                page_ids = page_ids.at[:total_q_tokens].set(page_ids_live)
                page_offsets = page_offsets.at[:total_q_tokens].set(page_offsets_live)

            slot_ids_packed = jnp.zeros((self.config.max_num_seqs,), dtype=jnp.int32)
            slot_ids_packed = slot_ids_packed.at[:num_packed].set(
                jnp.array(packed_slots, dtype=jnp.int32)
            )
            page_indices = self.page_indices[slot_ids_packed]

            kv_lens_list = [step.kv_lens[slot_id] for slot_id in packed_slots]
            kv_lens = jnp.zeros((self.config.max_num_seqs,), dtype=jnp.int32)
            kv_lens = kv_lens.at[:num_packed].set(jnp.array(kv_lens_list, dtype=jnp.int32))

            q_lens_list = [step.q_lens[slot_id] for slot_id in packed_slots]
            cu_q_lens_list = [0]
            for q_len in q_lens_list:
                cu_q_lens_list.append(cu_q_lens_list[-1] + int(q_len))
            while len(cu_q_lens_list) < self.config.max_num_seqs + 1:
                cu_q_lens_list.append(cu_q_lens_list[-1])
            cu_q_lens = jnp.array(cu_q_lens_list, dtype=jnp.int32)

            num_seqs = jnp.array([num_packed], dtype=jnp.int32)

            sample_mask_list = [step.sample_mask[slot_id] for slot_id in packed_slots]
            sample_mask = jnp.zeros((self.config.max_num_seqs,), dtype=jnp.bool_)
            if sample_mask_list:
                sample_mask = sample_mask.at[:num_packed].set(
                    jnp.array(sample_mask_list, dtype=jnp.bool_)
                )

            self.rng, step_rng = jax.random.split(self.rng, 2)
            self.kv_cache, next_token_ids, next_rng = self.runner.step(
                max_tokens=bucket_tokens,
                kv_cache=self.kv_cache,
                token_ids=token_ids,
                positions=positions,
                page_ids=page_ids,
                page_offsets=page_offsets,
                kv_lens=kv_lens,
                page_indices=page_indices,
                cu_q_lens=cu_q_lens,
                num_seqs=num_seqs,
                sample_mask=sample_mask,
                rng=step_rng,
                temperature=temperature,
            )
            self.rng = next_rng

            next_token_ids_host = list(jax.device_get(next_token_ids))
            next_token_ids_by_slot = [0 for _ in range(self.config.max_num_seqs)]
            for i in range(num_packed):
                slot_id = packed_slots[i]
                next_token_ids_by_slot[slot_id] = int(next_token_ids_host[i])
            self.scheduler.postprocess(next_token_ids_by_slot, step)

            seqs = self.scheduler.drain_finished()
            for seq in seqs:
                finished[seq.request.request_id] = seq

        return finished
