import numpy as np

from jaxformers.inference.vllm_sync import (
    build_gemma3_vllm_sync_payload,
    pack_gemma3_for_vllm_sync,
    reshard_like_vllm_state,
    SyncState,
    sync_vllm_state_in_place,
)


def _dense_weights(num_layers: int = 2):
    weights = {
        "model.embed_tokens.weight": np.arange(12, dtype=np.float32).reshape(4, 3),
        "model.norm.weight": np.arange(3, dtype=np.float32),
    }
    for layer in range(num_layers):
        base = 100 * (layer + 1)
        prefix = f"model.layers.{layer}."
        weights[f"{prefix}input_layernorm.weight"] = np.full((3,), base + 1, dtype=np.float32)
        weights[f"{prefix}post_attention_layernorm.weight"] = np.full((3,), base + 2, dtype=np.float32)
        weights[f"{prefix}pre_feedforward_layernorm.weight"] = np.full((3,), base + 3, dtype=np.float32)
        weights[f"{prefix}post_feedforward_layernorm.weight"] = np.full((3,), base + 4, dtype=np.float32)
        weights[f"{prefix}self_attn.q_proj.weight"] = np.full((4, 3), base + 10, dtype=np.float32)
        weights[f"{prefix}self_attn.k_proj.weight"] = np.full((2, 3), base + 20, dtype=np.float32)
        weights[f"{prefix}self_attn.v_proj.weight"] = np.full((2, 3), base + 30, dtype=np.float32)
        weights[f"{prefix}self_attn.o_proj.weight"] = np.full((3, 4), base + 40, dtype=np.float32)
        weights[f"{prefix}self_attn.q_norm.weight"] = np.full((2,), base + 50, dtype=np.float32)
        weights[f"{prefix}self_attn.k_norm.weight"] = np.full((2,), base + 60, dtype=np.float32)
        weights[f"{prefix}mlp.gate_proj.weight"] = np.full((5, 3), base + 70, dtype=np.float32)
        weights[f"{prefix}mlp.up_proj.weight"] = np.full((5, 3), base + 80, dtype=np.float32)
        weights[f"{prefix}mlp.down_proj.weight"] = np.full((3, 5), base + 90, dtype=np.float32)
    return weights


def _scan_weights(num_layers: int = 2):
    weights = {
        "model.embed_tokens.weight": np.arange(12, dtype=np.float32).reshape(4, 3),
        "model.norm.weight": np.arange(3, dtype=np.float32),
        "input_layernorm.weight": np.stack(
            [np.full((3,), 10 + layer, dtype=np.float32) for layer in range(num_layers)]
        ),
        "post_attention_layernorm.weight": np.stack(
            [np.full((3,), 20 + layer, dtype=np.float32) for layer in range(num_layers)]
        ),
        "pre_feedforward_layernorm.weight": np.stack(
            [np.full((3,), 30 + layer, dtype=np.float32) for layer in range(num_layers)]
        ),
        "post_feedforward_layernorm.weight": np.stack(
            [np.full((3,), 40 + layer, dtype=np.float32) for layer in range(num_layers)]
        ),
        "self_attn.q_proj.weight": np.stack(
            [np.full((4, 3), 100 + layer, dtype=np.float32) for layer in range(num_layers)]
        ),
        "self_attn.k_proj.weight": np.stack(
            [np.full((2, 3), 200 + layer, dtype=np.float32) for layer in range(num_layers)]
        ),
        "self_attn.v_proj.weight": np.stack(
            [np.full((2, 3), 300 + layer, dtype=np.float32) for layer in range(num_layers)]
        ),
        "self_attn.o_proj.weight": np.stack(
            [np.full((3, 4), 400 + layer, dtype=np.float32) for layer in range(num_layers)]
        ),
        "self_attn.q_norm.weight": np.stack(
            [np.full((2,), 500 + layer, dtype=np.float32) for layer in range(num_layers)]
        ),
        "self_attn.k_norm.weight": np.stack(
            [np.full((2,), 600 + layer, dtype=np.float32) for layer in range(num_layers)]
        ),
        "mlp.gate_proj.weight": np.stack(
            [np.full((5, 3), 700 + layer, dtype=np.float32) for layer in range(num_layers)]
        ),
        "mlp.up_proj.weight": np.stack(
            [np.full((5, 3), 800 + layer, dtype=np.float32) for layer in range(num_layers)]
        ),
        "mlp.down_proj.weight": np.stack(
            [np.full((3, 5), 900 + layer, dtype=np.float32) for layer in range(num_layers)]
        ),
    }
    return weights


def test_pack_dense_gemma3_for_vllm_sync_packs_qkv_and_gate_up():
    weights = _dense_weights()
    packed = pack_gemma3_for_vllm_sync(weights)

    key = "vllm_model.model.layers.1.self_attn.qkv_proj.weight"
    expected_qkv = np.concatenate(
        [
            weights["model.layers.1.self_attn.q_proj.weight"],
            weights["model.layers.1.self_attn.k_proj.weight"],
            weights["model.layers.1.self_attn.v_proj.weight"],
        ],
        axis=0,
    )
    np.testing.assert_array_equal(packed[key], expected_qkv)

    gate_key = "vllm_model.model.layers.1.mlp.gate_up_proj.weight"
    expected_gate = np.concatenate(
        [
            weights["model.layers.1.mlp.gate_proj.weight"],
            weights["model.layers.1.mlp.up_proj.weight"],
        ],
        axis=0,
    )
    np.testing.assert_array_equal(packed[gate_key], expected_gate)


def test_pack_dense_gemma3_for_vllm_sync_interleaves_tp_shards():
    weights = _dense_weights(num_layers=1)
    packed = pack_gemma3_for_vllm_sync(weights, tensor_parallel_size=2)

    qkv_key = "vllm_model.model.layers.0.self_attn.qkv_proj.weight"
    expected_qkv = np.concatenate(
        [
            weights["model.layers.0.self_attn.q_proj.weight"][:2],
            weights["model.layers.0.self_attn.k_proj.weight"][:1],
            weights["model.layers.0.self_attn.v_proj.weight"][:1],
            weights["model.layers.0.self_attn.q_proj.weight"][2:],
            weights["model.layers.0.self_attn.k_proj.weight"][1:],
            weights["model.layers.0.self_attn.v_proj.weight"][1:],
        ],
        axis=0,
    )
    np.testing.assert_array_equal(packed[qkv_key], expected_qkv)

    gate_key = "vllm_model.model.layers.0.mlp.gate_up_proj.weight"
    expected_gate = np.concatenate(
        [
            weights["model.layers.0.mlp.gate_proj.weight"][:3],
            weights["model.layers.0.mlp.up_proj.weight"][:3],
            weights["model.layers.0.mlp.gate_proj.weight"][3:],
            weights["model.layers.0.mlp.up_proj.weight"][3:],
        ],
        axis=0,
    )
    np.testing.assert_array_equal(packed[gate_key], expected_gate)


def test_pack_scan_gemma3_for_vllm_sync_expands_layers():
    weights = _scan_weights()
    payload = build_gemma3_vllm_sync_payload(weights)

    key = "vllm_model.model.layers.0.self_attn.qkv_proj.weight"
    expected = np.concatenate(
        [
            weights["self_attn.q_proj.weight"][0],
            weights["self_attn.k_proj.weight"][0],
            weights["self_attn.v_proj.weight"][0],
        ],
        axis=0,
    )
    np.testing.assert_array_equal(payload.updated_weights.tensors[key], expected)
    assert payload.mappings[key] == (key, ())
    assert payload.transpose_keys == {}


def test_pack_scan_gemma3_for_vllm_sync_interleaves_tp_shards():
    weights = _scan_weights(num_layers=1)
    payload = build_gemma3_vllm_sync_payload(weights, tensor_parallel_size=2)

    qkv_key = "vllm_model.model.layers.0.self_attn.qkv_proj.weight"
    expected_qkv = np.concatenate(
        [
            weights["self_attn.q_proj.weight"][0][:2],
            weights["self_attn.k_proj.weight"][0][:1],
            weights["self_attn.v_proj.weight"][0][:1],
            weights["self_attn.q_proj.weight"][0][2:],
            weights["self_attn.k_proj.weight"][0][1:],
            weights["self_attn.v_proj.weight"][0][1:],
        ],
        axis=0,
    )
    np.testing.assert_array_equal(payload.updated_weights.tensors[qkv_key], expected_qkv)

    gate_key = "vllm_model.model.layers.0.mlp.gate_up_proj.weight"
    expected_gate = np.concatenate(
        [
            weights["mlp.gate_proj.weight"][0][:3],
            weights["mlp.up_proj.weight"][0][:3],
            weights["mlp.gate_proj.weight"][0][3:],
            weights["mlp.up_proj.weight"][0][3:],
        ],
        axis=0,
    )
    np.testing.assert_array_equal(payload.updated_weights.tensors[gate_key], expected_gate)


def test_reshard_like_vllm_state_casts_dtype_to_target():
    payload = build_gemma3_vllm_sync_payload(_dense_weights(num_layers=1))
    target = SyncState(
        {
            key: np.zeros_like(value, dtype=np.float16)
            for key, value in payload.updated_weights.tensors.items()
        }
    )

    resharded = reshard_like_vllm_state(payload.updated_weights, target)

    assert set(resharded.tensors) == set(payload.updated_weights.tensors)
    assert all(value.dtype == np.dtype(np.float16) for value in resharded.tensors.values())


def test_sync_vllm_state_in_place_updates_flat_dict_targets():
    payload = build_gemma3_vllm_sync_payload(_dense_weights(num_layers=1))
    target_state = {
        key: np.zeros_like(value, dtype=np.float16)
        for key, value in payload.updated_weights.tensors.items()
    }

    synced = sync_vllm_state_in_place(
        payload.updated_weights,
        target_state,
        payload.mappings,
        payload.transpose_keys,
    )

    assert synced is target_state
    for key, value in payload.updated_weights.tensors.items():
        np.testing.assert_array_equal(target_state[key], value.astype(np.float16))
