from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import jax
import jax.numpy as jnp
import numpy as np

from jaxformers.inference.attention_metadata import AttentionMetadata
from jaxformers.inference.bucketing import round_up_to_power_of_two
from jaxformers.inference.ops.sample import SamplingMetadata


@dataclass
class SamplingParams:
    temperature: float = -1.0
    top_k: int = 0
    top_p: float = 1.0
    max_new_tokens: int = 128
    ignore_eos: bool = False
    stop_token_ids: tuple[int, ...] = ()


@dataclass
class RequestState:
    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    output_token_ids: list[int] = field(default_factory=list)
    slot_id: int | None = None
    num_cached_tokens: int = 0

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


@dataclass
class ScheduledBatch:
    input_ids: jax.Array
    attn_metadata: AttentionMetadata
    req_ids: list[str]
    q_lens: list[int]
    logits_indices: jax.Array
    sampling_metadata: SamplingMetadata
    is_prefill: bool
    should_sample: bool
    num_reqs: int
    num_tokens: int
    padded_num_reqs: int
    padded_num_tokens: int


class InputBatch:
    def __init__(
        self,
        max_model_len: int,
        max_num_batched_token: int,
        page_size: int,
        num_pages: int,
        *,
        max_num_seqs: int | None = None,
        max_num_request: int | None = None,
    ):
        if max_num_seqs is None:
            max_num_seqs = max_num_request
        if max_num_seqs is None:
            raise TypeError("InputBatch requires `max_num_seqs`.")
        if max_num_request is not None and max_num_request != max_num_seqs:
            raise ValueError(
                "`max_num_seqs` and legacy `max_num_request` must match when both are provided."
            )
        self.max_num_seqs = int(max_num_seqs)
        self.max_model_len = max_model_len
        self.max_num_batched_token = max_num_batched_token
        self.page_size = page_size
        self.num_pages = num_pages
        self.pages_per_req = (max_model_len + page_size - 1) // page_size
        if self.pages_per_req * self.max_num_seqs > self.num_pages:
            raise ValueError(
                "Insufficient `num_pages` for static slot mapping. "
                f"Need at least {self.pages_per_req * self.max_num_seqs}, got {self.num_pages}."
            )

        self.waiting: Deque[RequestState] = deque()
        self.running: dict[str, RequestState] = {}
        self.running_order: list[str] = []
        self.finished: Deque[tuple[str, list[int]]] = deque()
        self._free_slots: Deque[int] = deque(range(self.max_num_seqs))
        self._req_counter = 0

    def add_request(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        req_id: str | None = None,
    ) -> str:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if req_id is None:
            req_id = f"req-{self._req_counter}"
            self._req_counter += 1
        req = RequestState(
            req_id=req_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
        )
        self.waiting.append(req)
        return req_id

    def is_finished(self) -> bool:
        return len(self.waiting) == 0 and len(self.running_order) == 0

    def pop_finished(self) -> list[tuple[str, list[int]]]:
        out = list(self.finished)
        self.finished.clear()
        return out

    def _slot_block_ids(self, slot_id: int) -> np.ndarray:
        start = slot_id * self.pages_per_req
        return np.arange(start, start + self.pages_per_req, dtype=np.int32)

    def _build_scheduled_batch(
        self,
        reqs: list[RequestState],
        is_prefill: bool,
        q_lens_override: list[int] | None = None,
    ) -> ScheduledBatch:
        num_reqs = len(reqs)
        if num_reqs == 0:
            raise ValueError("Cannot build a batch with zero requests.")
        if q_lens_override is not None and len(q_lens_override) != num_reqs:
            raise ValueError("`q_lens_override` must match `reqs` length.")

        block_tables = np.zeros((self.max_num_seqs, self.pages_per_req), dtype=np.int32)
        seq_lens = np.zeros((self.max_num_seqs,), dtype=np.int32)
        query_start_loc = np.zeros((self.max_num_seqs + 1,), dtype=np.int32)
        flat_tokens: list[int] = []
        input_positions: list[int] = []
        token_req_indices: list[int] = []
        q_lens: list[int] = []
        req_ids: list[str] = []
        temperature = np.full((num_reqs,), -1.0, dtype=np.float32)
        top_k = np.zeros((num_reqs,), dtype=np.int32)
        top_p = np.ones((num_reqs,), dtype=np.float32)
        should_sample = not is_prefill

        for i, req in enumerate(reqs):
            token_ids = req.token_ids
            if is_prefill:
                remaining = req.prompt_len - req.num_cached_tokens
                q_len = remaining
                if q_lens_override is not None:
                    q_len = min(int(q_lens_override[i]), remaining)
                q_tokens = token_ids[req.num_cached_tokens : req.num_cached_tokens + q_len]
            else:
                q_tokens = [token_ids[-1]]

            q_len = len(q_tokens)
            if q_len <= 0:
                raise ValueError(f"Request {req.req_id} has no tokens to schedule.")

            seq_len = req.num_cached_tokens + q_len
            if seq_len > self.max_model_len:
                raise ValueError(
                    f"Request {req.req_id} exceeds max_model_len={self.max_model_len}."
                )
            if is_prefill and seq_len == req.prompt_len:
                should_sample = True

            query_start_loc[i + 1] = query_start_loc[i] + q_len
            seq_lens[i] = seq_len
            q_lens.append(q_len)
            req_ids.append(req.req_id)
            flat_tokens.extend(q_tokens)
            input_positions.extend(range(req.num_cached_tokens, seq_len))
            token_req_indices.extend([i] * q_len)
            block_tables[i, :] = self._slot_block_ids(req.slot_id)

            sp = req.sampling_params
            temperature[i] = np.float32(sp.temperature)
            top_k[i] = np.int32(sp.top_k)
            top_p[i] = np.float32(sp.top_p)

        num_tokens = int(query_start_loc[num_reqs])
        padded_num_tokens = round_up_to_power_of_two(
            num_tokens, max_value=self.max_num_batched_token
        )
        padded_num_reqs = round_up_to_power_of_two(
            num_reqs, max_value=self.max_num_seqs
        )

        flat_tokens_padded = np.zeros((padded_num_tokens,), dtype=np.int32)
        flat_tokens_padded[:num_tokens] = np.asarray(flat_tokens, dtype=np.int32)
        input_positions_padded = np.zeros((padded_num_tokens,), dtype=np.int32)
        input_positions_padded[:num_tokens] = np.asarray(input_positions, dtype=np.int32)
        token_req_indices_padded = np.zeros((padded_num_tokens,), dtype=np.int32)
        token_req_indices_padded[:num_tokens] = np.asarray(token_req_indices, dtype=np.int32)

        # Pad request-level sampling tensors to a compiled bucket.
        temperature_padded = np.full((padded_num_reqs,), -1.0, dtype=np.float32)
        top_k_padded = np.zeros((padded_num_reqs,), dtype=np.int32)
        top_p_padded = np.ones((padded_num_reqs,), dtype=np.float32)
        temperature_padded[:num_reqs] = temperature
        top_k_padded[:num_reqs] = top_k
        top_p_padded[:num_reqs] = top_p

        request_distribution = np.array(
            [0, num_reqs, num_reqs] if is_prefill else [num_reqs, num_reqs, num_reqs],
            dtype=np.int32,
        )
        query_start_loc_cpu = query_start_loc.copy()
        seq_lens_cpu = seq_lens.copy()
        # Requests beyond `num_reqs` are treated as empty.
        if num_reqs < self.max_num_seqs:
            query_start_loc[num_reqs + 1 :] = query_start_loc[num_reqs]

        attn_metadata = AttentionMetadata(
            input_positions=jnp.asarray(input_positions_padded, dtype=jnp.int32),
            token_req_indices=jnp.asarray(token_req_indices_padded, dtype=jnp.int32),
            block_tables=jnp.asarray(block_tables.reshape(-1), dtype=jnp.int32),
            seq_lens=jnp.asarray(seq_lens, dtype=jnp.int32),
            query_start_loc=jnp.asarray(query_start_loc, dtype=jnp.int32),
            request_distribution=jnp.asarray(request_distribution, dtype=jnp.int32),
        )
        attn_metadata.query_start_loc_cpu = query_start_loc_cpu
        attn_metadata.seq_lens_cpu = seq_lens_cpu

        logits_indices = query_start_loc[:num_reqs] + np.array(q_lens, dtype=np.int32) - 1
        logits_indices_padded = np.zeros((padded_num_reqs,), dtype=np.int32)
        logits_indices_padded[:num_reqs] = logits_indices

        do_sample = bool(np.any(temperature_padded > 0.0))
        sampling_metadata = SamplingMetadata(
            temperature=jnp.asarray(temperature_padded, dtype=jnp.float32),
            top_k=jnp.asarray(top_k_padded, dtype=jnp.int32),
            top_p=jnp.asarray(top_p_padded, dtype=jnp.float32),
            do_sample=do_sample,
        )

        return ScheduledBatch(
            input_ids=jnp.asarray(flat_tokens_padded, dtype=jnp.int32)[None, :],
            attn_metadata=attn_metadata,
            req_ids=req_ids,
            q_lens=q_lens,
            logits_indices=jnp.asarray(logits_indices_padded, dtype=jnp.int32),
            sampling_metadata=sampling_metadata,
            is_prefill=is_prefill,
            should_sample=should_sample,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            padded_num_reqs=padded_num_reqs,
            padded_num_tokens=padded_num_tokens,
        )

    def schedule(self) -> ScheduledBatch | None:
        # 1) Continue prefill for running requests that still have prompt tokens left.
        remaining_budget = self.max_num_batched_token
        prefill_reqs: list[RequestState] = []
        prefill_q_lens: list[int] = []
        for req_id in self.running_order:
            if remaining_budget <= 0:
                break
            req = self.running[req_id]
            remaining = req.prompt_len - req.num_cached_tokens
            if remaining <= 0:
                continue
            q_len = min(remaining, remaining_budget)
            prefill_reqs.append(req)
            prefill_q_lens.append(q_len)
            remaining_budget -= q_len

        if prefill_reqs:
            return self._build_scheduled_batch(
                prefill_reqs, is_prefill=True, q_lens_override=prefill_q_lens
            )

        # 2) Start prefill for new waiting requests if there is free capacity.
        remaining_budget = self.max_num_batched_token
        while self.waiting and self._free_slots and remaining_budget > 0:
            req = self.waiting[0]
            remaining = req.prompt_len - req.num_cached_tokens
            if remaining <= 0:
                self.waiting.popleft()
                continue

            q_len = min(remaining, remaining_budget)
            if q_len <= 0:
                break

            self.waiting.popleft()
            req.slot_id = self._free_slots.popleft()
            self.running[req.req_id] = req
            self.running_order.append(req.req_id)
            prefill_reqs.append(req)
            prefill_q_lens.append(q_len)
            remaining_budget -= q_len

        if prefill_reqs:
            return self._build_scheduled_batch(
                prefill_reqs, is_prefill=True, q_lens_override=prefill_q_lens
            )

        if not self.running_order:
            return None

        num_decode_reqs = min(len(self.running_order), self.max_num_batched_token)
        decode_reqs = [self.running[req_id] for req_id in self.running_order[:num_decode_reqs]]
        return self._build_scheduled_batch(decode_reqs, is_prefill=False)

    def commit_model_step(self, scheduled_batch: ScheduledBatch) -> None:
        for req_id, q_len in zip(scheduled_batch.req_ids, scheduled_batch.q_lens):
            req = self.running[req_id]
            req.num_cached_tokens += q_len

    def commit_sampled_tokens(
        self,
        scheduled_batch: ScheduledBatch,
        sampled_token_ids: list[int],
        eos_token_id: int | None,
    ) -> None:
        to_remove: list[str] = []
        for req_id, token_id in zip(scheduled_batch.req_ids, sampled_token_ids):
            req = self.running[req_id]
            # Chunked prefill: only accept a sample once the full prompt is cached.
            if scheduled_batch.is_prefill and req.num_cached_tokens < req.prompt_len:
                continue
            req.output_token_ids.append(int(token_id))

            sp = req.sampling_params
            reached_max_tokens = len(req.output_token_ids) >= sp.max_new_tokens
            reached_model_len = req.num_tokens >= self.max_model_len
            reached_stop_token = int(token_id) in sp.stop_token_ids
            reached_eos = (
                (not sp.ignore_eos)
                and eos_token_id is not None
                and int(token_id) == int(eos_token_id)
            )
            if reached_max_tokens or reached_model_len or reached_stop_token or reached_eos:
                to_remove.append(req_id)

        for req_id in to_remove:
            req = self.running.pop(req_id)
            self.running_order.remove(req_id)
            self.finished.append((req_id, list(req.output_token_ids)))
            if req.slot_id is not None:
                self._free_slots.append(req.slot_id)
                req.slot_id = None
