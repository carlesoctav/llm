from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxformers.inference.input_batch import InputBatch
from jaxformers.inference.kv_cache import PagedKVCacheConfig, make_kv_cache
from jaxformers.inference.model_runner import JaxModelRunner
from jaxformers.inference.scheduler import Request, Scheduler, SchedulerOutput, Sequence
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
        self.kv_cache = make_kv_cache(kv_cfg, dtype=config.dtype)

        self.input_batch = InputBatch(
            max_num_seqs=config.max_num_seqs,
            max_model_len=config.max_model_len,
            max_num_batched_tokens=config.max_num_batched_tokens,
            page_size=config.page_size,
        )
        self.input_batch.init_page_indices(pages_per_seq=kv_cfg.pages_per_seq)

        self.runner = JaxModelRunner(
            model=model,
            max_num_seqs=config.max_num_seqs,
            max_model_len=config.max_model_len,
            dtype=config.dtype,
        )
        self.runner.compile(
            max_num_batched_tokens=config.max_num_batched_tokens,
            min_token_bucket=16,
            min_seq_bucket=8,
        )

        self.rng = jax.random.PRNGKey(config.seed)
        mesh = jax.sharding.get_mesh()
        self.replicated = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())

        self._precompile()

    def _precompile(self) -> None:
        pages_per_seq = self.kv_cfg.pages_per_seq

        max_num_seqs = self.config.max_num_seqs
        kv_lens = jnp.zeros((max_num_seqs,), dtype=jnp.int32).at[0].set(1)
        kv_lens = jax.device_put(kv_lens, self.replicated)
        page_indices = jax.device_put(
            jnp.zeros((max_num_seqs, pages_per_seq), dtype=jnp.int32),
            self.replicated,
        )
        cu_q_lens = jnp.zeros((max_num_seqs + 1,), dtype=jnp.int32).at[1:].set(1)
        cu_q_lens = jax.device_put(cu_q_lens, self.replicated)
        num_seqs = jax.device_put(jnp.array([1], dtype=jnp.int32), self.replicated)

        for max_tokens in sorted(self.runner._backbone):
            token_ids = jax.device_put(jnp.zeros((max_tokens,), dtype=jnp.int32), self.replicated)
            positions = jax.device_put(jnp.zeros((max_tokens,), dtype=jnp.int32), self.replicated)
            page_ids = jnp.full((max_tokens,), -1, dtype=jnp.int32).at[0].set(0)
            page_offsets = jnp.full((max_tokens,), -1, dtype=jnp.int32).at[0].set(0)
            page_ids = jax.device_put(page_ids, self.replicated)
            page_offsets = jax.device_put(page_offsets, self.replicated)
            self.kv_cache, last_hidden = self.runner.backbone(
                max_tokens=max_tokens,
                kv_cache=self.kv_cache,
                token_ids=token_ids,
                positions=positions,
                page_ids=page_ids,
                page_offsets=page_offsets,
                kv_lens=kv_lens,
                page_indices=page_indices,
                cu_q_lens=cu_q_lens,
                num_seqs=num_seqs,
            )
            last_hidden.block_until_ready()

        dummy_hidden = jax.device_put(
            jnp.zeros((max_num_seqs, self.model.config.hidden_size), dtype=self.config.dtype),
            self.replicated,
        )

        for padded_num_seqs in sorted(self.runner._select):
            indices = jax.device_put(jnp.zeros((padded_num_seqs,), dtype=jnp.int32), self.replicated)
            selected = self.runner.select(
                padded_num_seqs=padded_num_seqs,
                last_hidden=dummy_hidden,
                indices=indices,
            )
            logits = self.runner.compute_logits(padded_num_seqs=padded_num_seqs, hidden=selected)
            logits.block_until_ready()

            self.rng, step_rng = jax.random.split(self.rng, 2)
            sampled = self.runner.sample(
                padded_num_seqs=padded_num_seqs,
                rng=step_rng,
                logits=logits,
                temperature=1.0,
                do_sampling=True,
            )
            sampled.block_until_ready()
            greedy = self.runner.sample(
                padded_num_seqs=padded_num_seqs,
                rng=step_rng,
                logits=logits,
                temperature=0.0,
                do_sampling=False,
            )
            greedy.block_until_ready()

    def add_request(self, request: Request) -> None:
        self.scheduler.add_request(request)

    def run(self, *, temperature: float) -> dict[int, tuple[list[int], list[bool]]]:
        finished: dict[int, tuple[list[int], list[bool]]] = {}
        pages_per_seq = self.kv_cfg.pages_per_seq

        while not self.scheduler.is_finished():
            for slot_id in range(self.config.max_num_seqs):
                seq = self.scheduler.slots[slot_id]
                if seq is None:
                    continue
                if seq.initialized:
                    continue
                self.input_batch.init_sequence(
                    slot_id=slot_id,
                    prompt_token_ids=seq.request.prompt_token_ids,
                )
                seq.initialized = True

            step = self.scheduler.schedule()
            total_q_tokens = int(step.total_num_scheduled_tokens)
            if total_q_tokens == 0:
                seqs = self.scheduler.drain_finished()
                for seq in seqs:
                    token_ids = self.input_batch.token_ids_cpu[
                        seq.slot_id, : seq.prompt_len + seq.num_generated
                    ].tolist()
                    generation_mask = seq.request.prompt_generation_mask + [
                        True for _ in range(seq.num_generated)
                    ]
                    finished[seq.request.request_id] = (token_ids, generation_mask)
                continue

            bucket_tokens = self.runner.pick_token_bucket(total_q_tokens)

            prepared = self.input_batch.prepare_step_inputs(
                step=step,
                slots=self.scheduler.slots,
                bucket_tokens=bucket_tokens,
                pages_per_seq=pages_per_seq,
            )

            token_ids = jax.device_put(prepared.input_ids, self.replicated)
            positions = jax.device_put(prepared.positions, self.replicated)
            page_ids = jax.device_put(prepared.page_ids, self.replicated)
            page_offsets = jax.device_put(prepared.page_offsets, self.replicated)
            kv_lens = jax.device_put(prepared.kv_lens, self.replicated)
            page_indices = jax.device_put(prepared.page_indices, self.replicated)
            cu_q_lens = jax.device_put(prepared.cu_q_lens, self.replicated)
            num_seqs = jax.device_put(prepared.num_seqs, self.replicated)

            self.kv_cache, last_hidden = self.runner.backbone(
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
            )

            next_token_ids_by_slot = [0 for _ in range(self.config.max_num_seqs)]
            if prepared.num_sampled > 0:
                padded_num_sampled = prepared.padded_num_sampled
                sample_indices = jax.device_put(prepared.sample_packed_indices, self.replicated)

                selected_hidden = self.runner.select(
                    padded_num_seqs=padded_num_sampled,
                    last_hidden=last_hidden,
                    indices=sample_indices,
                )
                logits = self.runner.compute_logits(
                    padded_num_seqs=padded_num_sampled, hidden=selected_hidden
                )

                do_sampling = float(temperature) != 0.0
                self.rng, step_rng = jax.random.split(self.rng, 2)
                sampled = self.runner.sample(
                    padded_num_seqs=padded_num_sampled,
                    rng=step_rng,
                    logits=logits,
                    temperature=float(temperature) if do_sampling else 1.0,
                    do_sampling=do_sampling,
                )

                sampled_host = list(jax.device_get(sampled))[: prepared.num_sampled]
                packed_indices_host = prepared.sample_packed_indices[: prepared.num_sampled]

                for j in range(prepared.num_sampled):
                    packed_idx = int(packed_indices_host[j])
                    slot_id = prepared.packed_slots[packed_idx]
                    token_id = int(sampled_host[j])
                    write_pos = int(step.kv_lens[slot_id])
                    self.input_batch.write_token(
                        slot_id=slot_id,
                        position=write_pos,
                        token_id=token_id,
                    )
                    next_token_ids_by_slot[slot_id] = token_id

            self.scheduler.postprocess(next_token_ids_by_slot, step)

            seqs = self.scheduler.drain_finished()
            for seq in seqs:
                token_ids = self.input_batch.token_ids_cpu[
                    seq.slot_id, : seq.prompt_len + seq.num_generated
                ].tolist()
                generation_mask = seq.request.prompt_generation_mask + [
                    True for _ in range(seq.num_generated)
                ]
                finished[seq.request.request_id] = (token_ids, generation_mask)

        return finished
