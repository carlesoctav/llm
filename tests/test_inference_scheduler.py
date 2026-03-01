from __future__ import annotations

import pytest

from jaxformers.inference.sampling import SamplingParams
from jaxformers.inference.scheduler import Request, Scheduler, Sequence


def _make_request(*, request_id: int, prompt_len: int, max_tokens: int) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=[1 for _ in range(prompt_len)],
        prompt_generation_mask=[False for _ in range(prompt_len)],
        eos_token_id=-1,
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            ignore_eos=True,
        ),
    )


def test_prefill_chunk_size_validation() -> None:
    with pytest.raises(ValueError, match="prefill_chunk_size must be >= 1"):
        Scheduler(
            max_num_seqs=1,
            max_model_len=16,
            max_num_batched_tokens=8,
            prefill_chunk_size=0,
        )

    with pytest.raises(ValueError, match="prefill_chunk_size must be <= max_num_batched_tokens"):
        Scheduler(
            max_num_seqs=1,
            max_model_len=16,
            max_num_batched_tokens=8,
            prefill_chunk_size=9,
        )


def test_prefill_is_chunked_across_steps() -> None:
    scheduler = Scheduler(
        max_num_seqs=1,
        max_model_len=32,
        max_num_batched_tokens=8,
        prefill_chunk_size=3,
    )
    scheduler.add_request(_make_request(request_id=0, prompt_len=8, max_tokens=1))

    step1 = scheduler.schedule()
    assert step1.q_lens[0] == 3
    assert step1.kv_lens[0] == 3
    assert step1.sample_mask[0] is False
    assert step1.total_num_scheduled_tokens == 3
    scheduler.postprocess([0], step1)

    step2 = scheduler.schedule()
    assert step2.q_lens[0] == 3
    assert step2.kv_lens[0] == 6
    assert step2.sample_mask[0] is False
    assert step2.total_num_scheduled_tokens == 3
    scheduler.postprocess([0], step2)

    step3 = scheduler.schedule()
    assert step3.q_lens[0] == 2
    assert step3.kv_lens[0] == 8
    assert step3.sample_mask[0] is True
    assert step3.total_num_scheduled_tokens == 2

    scheduler.postprocess([123], step3)
    finished = scheduler.drain_finished()
    assert len(finished) == 1
    assert finished[0].num_generated == 1


def test_decode_priority_with_prefill_chunking() -> None:
    scheduler = Scheduler(
        max_num_seqs=2,
        max_model_len=32,
        max_num_batched_tokens=4,
        prefill_chunk_size=2,
    )

    decode_req = _make_request(request_id=0, prompt_len=2, max_tokens=3)
    prefill_req = _make_request(request_id=1, prompt_len=10, max_tokens=1)
    scheduler.slots[0] = Sequence(
        request=decode_req,
        slot_id=0,
        cached_len=2,
        num_generated=1,
        finished=False,
        initialized=True,
    )
    scheduler.slots[1] = Sequence(
        request=prefill_req,
        slot_id=1,
        cached_len=0,
        num_generated=0,
        finished=False,
        initialized=True,
    )

    step = scheduler.schedule()
    assert step.q_lens[0] == 1
    assert step.sample_mask[0] is True
    assert step.q_lens[1] == 2
    assert step.sample_mask[1] is False
    assert step.total_num_scheduled_tokens == 3
