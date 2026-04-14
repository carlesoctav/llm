import jax.numpy as jnp
import numpy as np

from jaxformers.attention_utils import (
    encode_tpu_flash_attention_code,
    prepare_attention_kwargs,
)
from jaxformers.masking_utils import (
    make_causal_mask,
    make_sliding_window_causal_mask,
)


def test_prepare_attention_kwargs_eager_matches_causal_mask():
    input_embeds = jnp.zeros((2, 4, 8), dtype=jnp.float32)
    attention_mask = jnp.array([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=jnp.bool_)

    kwargs = prepare_attention_kwargs(
        "eager",
        input_embeds,
        attention_mask=attention_mask,
        is_causal=True,
        is_sliding=False,
        window_size=2,
    )

    expected = make_causal_mask(
        "eager",
        input_embeds,
        attention_mask=attention_mask,
    )
    np.testing.assert_array_equal(np.asarray(kwargs["mask"]), np.asarray(expected))


def test_prepare_attention_kwargs_eager_matches_sliding_causal_mask():
    input_embeds = jnp.zeros((2, 6, 8), dtype=jnp.float32)
    attention_mask = jnp.array(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]],
        dtype=jnp.bool_,
    )

    kwargs = prepare_attention_kwargs(
        "sdpa",
        input_embeds,
        attention_mask=attention_mask,
        is_causal=True,
        is_sliding=True,
        window_size=3,
    )

    expected = make_sliding_window_causal_mask(
        "eager",
        input_embeds,
        3,
        attention_mask=attention_mask,
    )
    np.testing.assert_array_equal(np.asarray(kwargs["mask"]), np.asarray(expected))


def test_prepare_attention_kwargs_tpu_flash_zeroes_segment_padding():
    input_embeds = jnp.zeros((1, 3, 8), dtype=jnp.float32)
    attention_mask = jnp.array([[1, 1, 0]], dtype=jnp.bool_)
    segment_ids = jnp.array([[5, 5, 9]], dtype=jnp.int32)

    kwargs = prepare_attention_kwargs(
        "tpu_flash",
        input_embeds,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
        is_causal=True,
        is_sliding=True,
        is_mqa=True,
        window_size=4,
    )

    assert kwargs["mask"].shape == (1,)
    np.testing.assert_array_equal(
        np.asarray(kwargs["segment_ids"]),
        np.array([[5, 5, 0]], dtype=np.int32),
    )
    assert int(np.asarray(kwargs["attention_code"])) == 0b111
    assert kwargs["window_size"] == 4


def test_encode_tpu_flash_attention_code_uses_expected_bits():
    assert int(np.asarray(encode_tpu_flash_attention_code(
        is_sliding=False,
        is_causal=False,
        is_mqa=False,
    ))) == 0b000
    assert int(np.asarray(encode_tpu_flash_attention_code(
        is_sliding=True,
        is_causal=False,
        is_mqa=False,
    ))) == 0b001
    assert int(np.asarray(encode_tpu_flash_attention_code(
        is_sliding=False,
        is_causal=True,
        is_mqa=False,
    ))) == 0b010
    assert int(np.asarray(encode_tpu_flash_attention_code(
        is_sliding=False,
        is_causal=False,
        is_mqa=True,
    ))) == 0b100
