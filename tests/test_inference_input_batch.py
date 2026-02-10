import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from jaxformers.inference.input_batch import InputBatch, SamplingParams


def test_input_batch_prefill_then_decode():
    batch = InputBatch(
        max_num_seqs=4,
        max_model_len=16,
        max_num_batched_token=8,
        page_size=4,
        num_pages=16,
    )
    sp = SamplingParams(max_new_tokens=2)
    batch.add_request([1, 2, 3], sp, req_id="a")
    batch.add_request([4, 5], sp, req_id="b")

    prefill = batch.schedule()
    assert prefill is not None
    assert prefill.is_prefill
    assert prefill.input_ids.shape == (1, 8)
    assert prefill.num_tokens == 5
    assert prefill.padded_num_tokens == 8
    np.testing.assert_array_equal(
        np.asarray(prefill.attn_metadata.query_start_loc)[:3],
        np.asarray([0, 3, 5], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(prefill.attn_metadata.seq_lens)[:2],
        np.asarray([3, 2], dtype=np.int32),
    )

    batch.commit_model_step(prefill)
    batch.commit_sampled_tokens(prefill, [10, 11], eos_token_id=None)
    assert not batch.pop_finished()

    decode = batch.schedule()
    assert decode is not None
    assert not decode.is_prefill
    assert decode.input_ids.shape == (1, 2)
    assert decode.num_tokens == 2
    assert decode.padded_num_tokens == 2
    np.testing.assert_array_equal(
        np.asarray(decode.attn_metadata.seq_lens)[:2],
        np.asarray([4, 3], dtype=np.int32),
    )

    batch.commit_model_step(decode)
    batch.commit_sampled_tokens(decode, [12, 13], eos_token_id=None)
    finished = dict(batch.pop_finished())
    assert set(finished.keys()) == {"a", "b"}
    assert finished["a"] == [10, 12]
    assert finished["b"] == [11, 13]


def test_input_batch_chunked_prefill_delays_sampling_until_prompt_complete():
    batch = InputBatch(
        max_num_seqs=2,
        max_model_len=16,
        max_num_batched_token=4,
        page_size=4,
        num_pages=16,
    )
    sp = SamplingParams(max_new_tokens=2)
    batch.add_request([1, 2, 3, 4, 5, 6], sp, req_id="a")

    prefill_0 = batch.schedule()
    assert prefill_0 is not None
    assert prefill_0.is_prefill
    assert not prefill_0.should_sample
    assert prefill_0.q_lens == [4]
    batch.commit_model_step(prefill_0)
    batch.commit_sampled_tokens(prefill_0, [10], eos_token_id=None)
    assert not batch.pop_finished()

    prefill_1 = batch.schedule()
    assert prefill_1 is not None
    assert prefill_1.is_prefill
    assert prefill_1.should_sample
    assert prefill_1.q_lens == [2]
    batch.commit_model_step(prefill_1)
    batch.commit_sampled_tokens(prefill_1, [11], eos_token_id=None)
    assert not batch.pop_finished()

    decode = batch.schedule()
    assert decode is not None
    assert not decode.is_prefill
    assert decode.should_sample
    batch.commit_model_step(decode)
    batch.commit_sampled_tokens(decode, [12], eos_token_id=None)
    finished = dict(batch.pop_finished())
    assert finished["a"] == [11, 12]
