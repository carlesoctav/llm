from __future__ import annotations

from dataclasses import dataclass

from jaxformers.inference.sampling import SamplingParams


@dataclass(frozen=True)
class Request:
    request_id: int
    prompt_token_ids: list[int]
    prompt_generation_mask: list[bool]
    eos_token_id: int
    sampling_params: SamplingParams


@dataclass
class Sequence:
    request: Request
    slot_id: int
    cached_len: int
    num_generated: int
    finished: bool
    initialized: bool

    @property
    def prompt_len(self) -> int:
        return len(self.request.prompt_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.num_generated

    def should_stop(self, token_id: int) -> bool:
        if self.num_generated >= self.request.sampling_params.max_tokens:
            return True
        if (
            (not self.request.sampling_params.ignore_eos)
            and token_id == self.request.eos_token_id
        ):
            return True
        return False


@dataclass(frozen=True)
class SchedulerOutput:
    num_seqs: int
    q_lens: list[int]
    kv_lens: list[int]
    cu_q_lens: list[int]
    sample_mask: list[bool]
    total_num_scheduled_tokens: int


class Scheduler:
    def __init__(
        self,
        *,
        max_num_seqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        prefill_chunk_size: int | None = None,
    ) -> None:
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        if prefill_chunk_size is not None:
            if prefill_chunk_size < 1:
                raise ValueError("prefill_chunk_size must be >= 1")
            if prefill_chunk_size > max_num_batched_tokens:
                raise ValueError(
                    "prefill_chunk_size must be <= max_num_batched_tokens "
                    f"({prefill_chunk_size} > {max_num_batched_tokens})"
                )
        self.prefill_chunk_size = prefill_chunk_size

        self.pending: list[Request] = []
        self.slots: list[Sequence | None] = [None for _ in range(max_num_seqs)]

    def add_request(self, request: Request) -> None:
        prompt_len = len(request.prompt_token_ids)
        if prompt_len > self.max_model_len:
            raise ValueError(
                f"prompt_len ({prompt_len}) must be <= max_model_len ({self.max_model_len})"
            )
        if prompt_len + request.sampling_params.max_tokens > self.max_model_len:
            raise ValueError(
                "prompt_len + max_tokens must be <= max_model_len "
                f"({prompt_len} + {request.sampling_params.max_tokens} > {self.max_model_len})"
            )
        if len(request.prompt_generation_mask) != prompt_len:
            raise ValueError("prompt_generation_mask must align with prompt_token_ids")
        self.pending.append(request)

    def is_finished(self) -> bool:
        if self.pending:
            return False
        for seq in self.slots:
            if seq is not None:
                return False
        return True

    def _fill_slots(self) -> None:
        if not self.pending:
            return
        for slot_id in range(self.max_num_seqs):
            if not self.pending:
                return
            if self.slots[slot_id] is not None:
                continue
            req = self.pending.pop(0)
            if req.sampling_params.max_tokens == 0:
                self.slots[slot_id] = Sequence(
                    request=req,
                    slot_id=slot_id,
                    cached_len=0,
                    num_generated=0,
                    finished=True,
                    initialized=False,
                )
                continue
            self.slots[slot_id] = Sequence(
                request=req,
                slot_id=slot_id,
                cached_len=0,
                num_generated=0,
                finished=False,
                initialized=False,
            )

    def schedule(self) -> SchedulerOutput:
        self._fill_slots()

        # Define num_seqs as the last non-empty slot + 1 (can include holes).
        max_slot = -1
        for i, seq in enumerate(self.slots):
            if seq is not None and not seq.finished:
                max_slot = i
        num_seqs = max_slot + 1

        q_lens = [0 for _ in range(self.max_num_seqs)]
        kv_lens = [0 for _ in range(self.max_num_seqs)]
        sample_mask = [False for _ in range(self.max_num_seqs)]

        # Start from current cached lens.
        for slot_id in range(self.max_num_seqs):
            seq = self.slots[slot_id]
            if seq is None:
                continue
            kv_lens[slot_id] = seq.cached_len

        token_budget = self.max_num_batched_tokens

        # 1) Decode: 1 token per seq (if available), prioritized.
        for slot_id in range(num_seqs):
            if token_budget <= 0:
                break
            seq = self.slots[slot_id]
            if seq is None or seq.finished:
                continue
            if seq.cached_len < seq.prompt_len:
                continue
            if seq.cached_len >= seq.total_len:
                continue
            q_lens[slot_id] = 1
            kv_lens[slot_id] = seq.cached_len + 1
            sample_mask[slot_id] = True
            token_budget -= 1

        # 2) Prefill: chunk prompt tokens into remaining budget.
        for slot_id in range(num_seqs):
            if token_budget <= 0:
                break
            if q_lens[slot_id] != 0:
                continue
            seq = self.slots[slot_id]
            if seq is None or seq.finished:
                continue
            if seq.cached_len >= seq.prompt_len:
                continue
            remaining = seq.prompt_len - seq.cached_len
            cap = remaining
            if self.prefill_chunk_size is not None:
                cap = min(cap, int(self.prefill_chunk_size))
            chunk = min(cap, token_budget)
            q_lens[slot_id] = chunk
            kv_lens[slot_id] = seq.cached_len + chunk
            # Only sample when we finish the prompt in this step.
            if seq.cached_len + chunk == seq.prompt_len:
                sample_mask[slot_id] = True
            token_budget -= chunk

        cu_q_lens = [0 for _ in range(self.max_num_seqs + 1)]
        for slot_id in range(self.max_num_seqs):
            cu_q_lens[slot_id + 1] = cu_q_lens[slot_id] + q_lens[slot_id]

        total_num_scheduled_tokens = cu_q_lens[self.max_num_seqs]
        return SchedulerOutput(
            num_seqs=num_seqs,
            q_lens=q_lens,
            kv_lens=kv_lens,
            cu_q_lens=cu_q_lens,
            sample_mask=sample_mask,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
        )

    def postprocess(self, next_token_ids_by_slot: list[int], step: SchedulerOutput) -> None:
        # Update state for scheduled sequences.
        for slot_id in range(step.num_seqs):
            q_len = step.q_lens[slot_id]
            if q_len == 0:
                continue
            seq = self.slots[slot_id]
            if seq is None:
                raise RuntimeError("Scheduled q_len>0 for an empty slot")
            if seq.finished:
                continue

            seq.cached_len = step.kv_lens[slot_id]

            if step.sample_mask[slot_id]:
                token_id = next_token_ids_by_slot[slot_id]
                seq.num_generated += 1
                if seq.should_stop(int(token_id)):
                    seq.finished = True

        # Free finished slots and fill with pending (for continuous batching).
        for slot_id in range(self.max_num_seqs):
            seq = self.slots[slot_id]
            if seq is None:
                continue
            if seq.finished:
                # Keep the sequence object for result collection until the caller reads it.
                # The caller can explicitly drain finished results, then we can reuse the slot.
                continue
        self._fill_slots()

    def drain_finished(self) -> list[Sequence]:
        finished: list[Sequence] = []
        for slot_id in range(self.max_num_seqs):
            seq = self.slots[slot_id]
            if seq is None:
                continue
            if not seq.finished:
                continue
            finished.append(seq)
            self.slots[slot_id] = None
        return finished
