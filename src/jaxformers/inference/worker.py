from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

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
        self.page_indices_host = np.array(jax.device_get(self.page_indices))
        self.kv_cache = make_kv_cache(kv_cfg, dtype=config.dtype)

        self.runner = JaxModelRunner(
            model=model,
            max_num_seqs=config.max_num_seqs,
            max_model_len=config.max_model_len,
            dtype=config.dtype,
        )
        self.runner.compile(max_num_batched_tokens=config.max_num_batched_tokens)

        self.rng = jax.random.PRNGKey(config.seed)
        mesh = jax.sharding.get_mesh()
        self.replicated = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())

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

            packed_slots = [
                slot_id
                for slot_id in range(self.config.max_num_seqs)
                if step.q_lens[slot_id] > 0
            ]
            num_packed = len(packed_slots)
            if num_packed == 0:
                raise RuntimeError("total_q_tokens > 0 but no packed slots found")

            token_ids_host = np.zeros((bucket_tokens,), dtype=np.int32)
            positions_host = np.zeros((bucket_tokens,), dtype=np.int32)
            page_ids_host = np.full((bucket_tokens,), -1, dtype=np.int32)
            page_offsets_host = np.full((bucket_tokens,), -1, dtype=np.int32)

            token_ids_host[:total_q_tokens] = np.array(step.token_ids, dtype=np.int32)
            positions_host[:total_q_tokens] = np.array(step.positions, dtype=np.int32)

            slot_ids: list[int] = []
            for slot_id in range(self.config.max_num_seqs):
                slot_ids.extend([slot_id for _ in range(step.q_lens[slot_id])])
            if len(slot_ids) != total_q_tokens:
                raise RuntimeError("slot_ids length mismatch")

            slot_ids_arr = np.array(slot_ids, dtype=np.int32)
            pos_arr = positions_host[:total_q_tokens]
            pages_per_seq = self.kv_cfg.pages_per_seq
            page_ids_host[:total_q_tokens] = slot_ids_arr * pages_per_seq + (
                pos_arr // self.config.page_size
            )
            page_offsets_host[:total_q_tokens] = pos_arr % self.config.page_size

            kv_lens_host = np.zeros((self.config.max_num_seqs,), dtype=np.int32)
            kv_lens_host[:num_packed] = np.array(
                [step.kv_lens[slot_id] for slot_id in packed_slots],
                dtype=np.int32,
            )

            q_lens_list = [step.q_lens[slot_id] for slot_id in packed_slots]
            cu_q_lens_host = np.zeros((self.config.max_num_seqs + 1,), dtype=np.int32)
            for i, q_len in enumerate(q_lens_list):
                cu_q_lens_host[i + 1] = cu_q_lens_host[i] + int(q_len)
            if num_packed + 1 < self.config.max_num_seqs + 1:
                cu_q_lens_host[num_packed + 1 :] = cu_q_lens_host[num_packed]

            num_seqs_host = np.array([num_packed], dtype=np.int32)

            sample_mask_host = np.zeros((self.config.max_num_seqs,), dtype=np.bool_)
            sample_mask_host[:num_packed] = np.array(
                [step.sample_mask[slot_id] for slot_id in packed_slots],
                dtype=np.bool_,
            )

            page_indices_host = np.zeros(
                (self.config.max_num_seqs, self.kv_cfg.pages_per_seq), dtype=np.int32
            )
            page_indices_host[:num_packed] = self.page_indices_host[packed_slots]

            token_ids = jax.device_put(token_ids_host, self.replicated)
            positions = jax.device_put(positions_host, self.replicated)
            page_ids = jax.device_put(page_ids_host, self.replicated)
            page_offsets = jax.device_put(page_offsets_host, self.replicated)
            kv_lens = jax.device_put(kv_lens_host, self.replicated)
            page_indices = jax.device_put(page_indices_host, self.replicated)
            cu_q_lens = jax.device_put(cu_q_lens_host, self.replicated)
            num_seqs = jax.device_put(num_seqs_host, self.replicated)
            sample_mask = jax.device_put(sample_mask_host, self.replicated)

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
