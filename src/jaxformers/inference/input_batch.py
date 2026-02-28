from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jaxformers.inference.scheduler import SchedulerOutput, Sequence


@dataclass(frozen=True)
class PreparedStepInputs:
    bucket_tokens: int
    total_q_tokens: int
    packed_slots: list[int]
    num_packed: int

    input_ids: np.ndarray
    positions: np.ndarray
    page_ids: np.ndarray
    page_offsets: np.ndarray

    kv_lens: np.ndarray
    cu_q_lens: np.ndarray
    num_seqs: np.ndarray
    page_indices: np.ndarray

    num_sampled: int
    padded_num_sampled: int
    sample_packed_indices: np.ndarray


class InputBatch:
    def __init__(
        self,
        *,
        max_num_seqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        page_size: int,
    ) -> None:
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.page_size = page_size

        self.token_ids_cpu = np.zeros((max_num_seqs, max_model_len), dtype=np.int32)

        self.input_ids_cpu = np.zeros((max_num_batched_tokens,), dtype=np.int32)
        self.positions_cpu = np.zeros((max_num_batched_tokens,), dtype=np.int32)
        self.page_ids_cpu = np.full((max_num_batched_tokens,), -1, dtype=np.int32)
        self.page_offsets_cpu = np.full((max_num_batched_tokens,), -1, dtype=np.int32)

        self.kv_lens_cpu = np.zeros((max_num_seqs,), dtype=np.int32)
        self.cu_q_lens_cpu = np.zeros((max_num_seqs + 1,), dtype=np.int32)
        self.num_seqs_cpu = np.zeros((1,), dtype=np.int32)

        self.sample_packed_indices_cpu = np.full((max_num_seqs,), -1, dtype=np.int32)

        self.seq_buckets: list[int] = []
        n = 8
        while n < max_num_seqs:
            self.seq_buckets.append(n)
            n *= 2
        self.seq_buckets.append(max_num_seqs)

        self.page_indices_static_cpu = None
        self.page_indices_cpu = None

    def init_page_indices(self, *, pages_per_seq: int) -> None:
        base = np.arange(self.max_num_seqs, dtype=np.int32)[:, None] * int(pages_per_seq)
        offsets = np.arange(int(pages_per_seq), dtype=np.int32)[None, :]
        self.page_indices_static_cpu = base + offsets
        self.page_indices_cpu = np.zeros((self.max_num_seqs, pages_per_seq), dtype=np.int32)

    def init_sequence(self, *, slot_id: int, prompt_token_ids: list[int]) -> None:
        if slot_id < 0 or slot_id >= self.max_num_seqs:
            raise ValueError("slot_id out of range")
        prompt_len = len(prompt_token_ids)
        if prompt_len > self.max_model_len:
            raise ValueError("prompt_len exceeds max_model_len")
        self.token_ids_cpu[slot_id, :] = 0
        self.token_ids_cpu[slot_id, :prompt_len] = np.array(prompt_token_ids, dtype=np.int32)

    def write_token(self, *, slot_id: int, position: int, token_id: int) -> None:
        if position < 0 or position >= self.max_model_len:
            raise ValueError("position out of range")
        self.token_ids_cpu[slot_id, position] = int(token_id)

    def pick_seq_bucket(self, num_sampled: int) -> int:
        if num_sampled < 0:
            raise ValueError("num_sampled must be >= 0")
        if num_sampled == 0:
            return self.seq_buckets[0]
        for b in self.seq_buckets:
            if b >= num_sampled:
                return b
        raise ValueError("num_sampled exceeds max_num_seqs")

    def prepare_step_inputs(
        self,
        *,
        step: SchedulerOutput,
        slots: list[Sequence | None],
        bucket_tokens: int,
        pages_per_seq: int,
    ) -> PreparedStepInputs:
        if self.page_indices_static_cpu is None:
            raise RuntimeError("init_page_indices must be called first")
        if self.page_indices_cpu is None:
            raise RuntimeError("init_page_indices must be called first")

        packed_slots: list[int] = []
        for slot_id in range(self.max_num_seqs):
            if step.q_lens[slot_id] > 0:
                packed_slots.append(slot_id)
        num_packed = len(packed_slots)
        if num_packed == 0:
            raise RuntimeError("No packed slots but total_q_tokens > 0")

        total_q_tokens = int(step.total_num_scheduled_tokens)
        if total_q_tokens <= 0:
            raise ValueError("total_q_tokens must be > 0")
        if total_q_tokens > bucket_tokens:
            raise ValueError("total_q_tokens must fit in bucket_tokens")

        cur = 0
        for slot_id in packed_slots:
            q_len = int(step.q_lens[slot_id])
            seq = slots[slot_id]
            if seq is None:
                raise RuntimeError("Scheduled q_len>0 for empty slot")

            cached_len = int(seq.cached_len)
            if cached_len < int(seq.prompt_len):
                start = cached_len
                end = cached_len + q_len
                self.input_ids_cpu[cur:cur + q_len] = self.token_ids_cpu[slot_id, start:end]
                self.positions_cpu[cur:cur + q_len] = np.arange(start, end, dtype=np.int32)
            else:
                if q_len != 1:
                    raise RuntimeError("Decode must schedule exactly 1 token")
                self.input_ids_cpu[cur] = self.token_ids_cpu[slot_id, cached_len]
                self.positions_cpu[cur] = np.int32(cached_len)

            pos_slice = self.positions_cpu[cur:cur + q_len]
            base = int(slot_id) * int(pages_per_seq)
            self.page_ids_cpu[cur:cur + q_len] = base + (pos_slice // int(self.page_size))
            self.page_offsets_cpu[cur:cur + q_len] = pos_slice % int(self.page_size)

            cur += q_len

        if cur != total_q_tokens:
            raise RuntimeError("Packed token count mismatch")

        self.input_ids_cpu[total_q_tokens:bucket_tokens] = 0
        self.positions_cpu[total_q_tokens:bucket_tokens] = 0
        self.page_ids_cpu[total_q_tokens:bucket_tokens] = -1
        self.page_offsets_cpu[total_q_tokens:bucket_tokens] = -1

        self.kv_lens_cpu[:] = 0
        for i, slot_id in enumerate(packed_slots):
            self.kv_lens_cpu[i] = np.int32(step.kv_lens[slot_id])

        self.cu_q_lens_cpu[:] = 0
        for i, slot_id in enumerate(packed_slots):
            self.cu_q_lens_cpu[i + 1] = self.cu_q_lens_cpu[i] + np.int32(step.q_lens[slot_id])
        if num_packed + 1 < self.max_num_seqs + 1:
            self.cu_q_lens_cpu[num_packed + 1:] = self.cu_q_lens_cpu[num_packed]

        self.num_seqs_cpu[0] = np.int32(num_packed)

        self.page_indices_cpu[:] = 0
        self.page_indices_cpu[:num_packed] = self.page_indices_static_cpu[packed_slots]

        sample_packed: list[int] = []
        for i, slot_id in enumerate(packed_slots):
            if step.sample_mask[slot_id]:
                sample_packed.append(i)
        num_sampled = len(sample_packed)

        padded_num_sampled = 0
        if num_sampled > 0:
            padded_num_sampled = self.pick_seq_bucket(num_sampled)
            self.sample_packed_indices_cpu[:padded_num_sampled] = -1
            for j, packed_idx in enumerate(sample_packed):
                self.sample_packed_indices_cpu[j] = np.int32(packed_idx)
        else:
            self.sample_packed_indices_cpu[0] = -1
        return PreparedStepInputs(
            bucket_tokens=bucket_tokens,
            total_q_tokens=total_q_tokens,
            packed_slots=packed_slots,
            num_packed=num_packed,
            input_ids=self.input_ids_cpu[:bucket_tokens],
            positions=self.positions_cpu[:bucket_tokens],
            page_ids=self.page_ids_cpu[:bucket_tokens],
            page_offsets=self.page_offsets_cpu[:bucket_tokens],
            kv_lens=self.kv_lens_cpu,
            cu_q_lens=self.cu_q_lens_cpu,
            num_seqs=self.num_seqs_cpu,
            page_indices=self.page_indices_cpu,
            num_sampled=num_sampled,
            padded_num_sampled=padded_num_sampled,
            sample_packed_indices=self.sample_packed_indices_cpu[:max(1, padded_num_sampled)],
        )
