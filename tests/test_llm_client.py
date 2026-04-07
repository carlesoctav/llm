import numpy as np
import jax.numpy as jnp
from transformers.tokenization_utils_base import BatchEncoding

from jaxformers.inference.llm_client import _coerce_prompt_ids, _sync_dict_state
from jaxformers.module_utils import VllmMapping, VllmWeightLeaf, VllmWeightState


def test_sync_dict_state_updates_target_key():
    key = "model.layers.0.self_attn.qkv_proj.weight"
    target_state = {
        key: jnp.zeros((6, 4), dtype=jnp.float32),
    }
    mapping = VllmMapping(
        state=VllmWeightState(
            leaves=[
                (
                    tuple(key.split(".")),
                    VllmWeightLeaf(value=jnp.ones((6, 4), dtype=jnp.bfloat16)),
                ),
            ],
        ),
        mappings={key: (key, None)},
        transpose_keys={},
    )

    updated_state = _sync_dict_state(target_state, mapping)

    np.testing.assert_allclose(
        np.asarray(updated_state[key]),
        np.ones((6, 4), dtype=np.float32),
    )
    assert updated_state[key].dtype == target_state[key].dtype


def test_sync_dict_state_applies_transpose():
    source_key = "model.layers.0.mlp.down_proj.weight"
    target_state = {
        source_key: jnp.zeros((4, 6), dtype=jnp.float32),
    }
    mapping = VllmMapping(
        state=VllmWeightState(
            leaves=[
                (
                    tuple(source_key.split(".")),
                    VllmWeightLeaf(
                        value=jnp.arange(24, dtype=jnp.float32).reshape(6, 4),
                    ),
                ),
            ],
        ),
        mappings={source_key: (source_key, None)},
        transpose_keys={source_key: (1, 0)},
    )

    updated_state = _sync_dict_state(target_state, mapping)

    np.testing.assert_allclose(
        np.asarray(updated_state[source_key]),
        np.arange(24, dtype=np.float32).reshape(6, 4).T,
    )


def test_sync_dict_state_resolves_vllm_runner_prefix():
    source_key = "model.embed_tokens.weight"
    target_key = f"vllm_model.{source_key}"
    target_state = {
        target_key: jnp.zeros((8, 4), dtype=jnp.float32),
    }
    mapping = VllmMapping(
        state=VllmWeightState(
            leaves=[
                (
                    tuple(source_key.split(".")),
                    VllmWeightLeaf(value=jnp.ones((8, 4), dtype=jnp.float32)),
                ),
            ],
        ),
        mappings={source_key: (source_key, None)},
        transpose_keys={},
    )

    updated_state = _sync_dict_state(target_state, mapping)

    np.testing.assert_allclose(
        np.asarray(updated_state[target_key]),
        np.ones((8, 4), dtype=np.float32),
    )


def test_coerce_prompt_ids_from_batch_encoding():
    prompt_ids = BatchEncoding({"input_ids": [1, 2, 3]})

    assert _coerce_prompt_ids(prompt_ids) == [1, 2, 3]
