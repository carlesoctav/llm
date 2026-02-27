from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    request_id: int
    prompt_token_ids: list[int]
    prompt_generation_mask: list[bool]
    max_tokens: int
    eos_token_id: int
    ignore_eos: bool


@dataclass
class Sequence:
    request: Request
    cached_len: int
    generated_token_ids: list[int]
    last_token_to_feed: int | None
    finished: bool

    @property
    def prompt_len(self) -> int:
        return len(self.request.prompt_token_ids)

    @property
    def num_generated(self) -> int:
        return len(self.generated_token_ids)

    def should_stop(self, token_id: int) -> bool:
        if self.num_generated >= self.request.max_tokens:
            return True
        if (not self.request.ignore_eos) and token_id == self.request.eos_token_id:
            return True
        return False


@dataclass(frozen=True)
class StepBatch:
    # Dynamic sequence count used by ragged paged attention.
    num_seqs: int

    # Concatenated q tokens, ordered by slot id.
    token_ids: list[int]
    positions: list[int]

    # Per-slot q lengths for this step.
    q_lens: list[int]

    # KV lens after updating cache with q tokens.
    kv_lens: list[int]

    # Cumulative q lengths (len == max_num_seqs + 1).
    cu_q_lens: list[int]

    # Per-slot flag: whether to sample next token from this step's logits.
    sample_mask: list[bool]


class Scheduler:
    def __init__(
        self,
        *,
        max_num_seqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
    ) -> None:
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens

        self.pending: list[Request] = []
        self.slots: list[Sequence | None] = [None for _ in range(max_num_seqs)]

    def add_request(self, request: Request) -> None:
        prompt_len = len(request.prompt_token_ids)
        if prompt_len > self.max_model_len:
            raise ValueError(
                f"prompt_len ({prompt_len}) must be <= max_model_len ({self.max_model_len})"
            )
        if prompt_len + request.max_tokens > self.max_model_len:
            raise ValueError(
                "prompt_len + max_tokens must be <= max_model_len "
                f"({prompt_len} + {request.max_tokens} > {self.max_model_len})"
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
            if req.max_tokens == 0:
                self.slots[slot_id] = Sequence(
                    request=req,
                    cached_len=0,
                    generated_token_ids=[],
                    last_token_to_feed=None,
                    finished=True,
                )
                continue
            self.slots[slot_id] = Sequence(
                request=req,
                cached_len=0,
                generated_token_ids=[],
                last_token_to_feed=None,
                finished=False,
            )

    def schedule(self) -> StepBatch:
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
            if seq.last_token_to_feed is None:
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
            chunk = remaining if remaining <= token_budget else token_budget
            q_lens[slot_id] = chunk
            kv_lens[slot_id] = seq.cached_len + chunk
            # Only sample when we finish the prompt in this step.
            if seq.cached_len + chunk == seq.prompt_len:
                sample_mask[slot_id] = True
            token_budget -= chunk

        # Build concatenated (token_ids, positions) and cu_q_lens.
        token_ids: list[int] = []
        positions: list[int] = []
        cu_q_lens = [0 for _ in range(self.max_num_seqs + 1)]
        for slot_id in range(self.max_num_seqs):
            cu_q_lens[slot_id + 1] = cu_q_lens[slot_id] + q_lens[slot_id]

            q_len = q_lens[slot_id]
            if q_len == 0:
                continue
            seq = self.slots[slot_id]
            if seq is None:
                raise RuntimeError("Scheduled q_len>0 for an empty slot")

            if seq.cached_len < seq.prompt_len:
                start = seq.cached_len
                end = seq.cached_len + q_len
                token_ids.extend(seq.request.prompt_token_ids[start:end])
                positions.extend(list(range(start, end)))
            else:
                if q_len != 1:
                    raise RuntimeError("Decode must schedule exactly 1 token")
                if seq.last_token_to_feed is None:
                    raise RuntimeError("Decode scheduled without last_token_to_feed")
                token_ids.append(seq.last_token_to_feed)
                positions.append(seq.cached_len)

        if cu_q_lens[self.max_num_seqs] != len(token_ids):
            raise RuntimeError("cu_q_lens does not match concatenated token count")

        return StepBatch(
            num_seqs=num_seqs,
            token_ids=token_ids,
            positions=positions,
            q_lens=q_lens,
            kv_lens=kv_lens,
            cu_q_lens=cu_q_lens,
            sample_mask=sample_mask,
        )

    def postprocess(self, next_token_ids_by_slot: list[int], step: StepBatch) -> None:
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
                seq.generated_token_ids.append(int(token_id))
                if seq.should_stop(int(token_id)):
                    seq.finished = True
                    seq.last_token_to_feed = None
                else:
                    seq.last_token_to_feed = int(token_id)

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
