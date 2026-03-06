from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from jaxformers.modeling_utils import Model


_DENSE_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_SCAN_LAYER_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
_OPTIONAL_SCAN_LAYER_SUFFIXES = (
    "self_attn.q_proj.bias",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.bias",
    "self_attn.o_proj.bias",
    "mlp.gate_proj.bias",
    "mlp.up_proj.bias",
    "mlp.down_proj.bias",
)


@dataclass
class _SyncLeaf:
    value: Any


@dataclass
class SyncState:
    tensors: dict[str, Any]

    def flat_state(self):
        return [((key,), _SyncLeaf(value)) for key, value in sorted(self.tensors.items())]

    def from_flat_path(self, flat_state):
        tensors = {
            ".".join(str(part) for part in path): leaf.value
            for path, leaf in flat_state
        }
        return SyncState(tensors=tensors)


@dataclass(frozen=True)
class VllmSyncPayload:
    updated_weights: SyncState
    mappings: dict[str, tuple[str, tuple[str, ...]]]
    transpose_keys: dict[str, tuple[int, ...]]


def _host_array(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _concat(values: list[Any], axis: int = 0) -> np.ndarray:
    return np.concatenate([_host_array(value) for value in values], axis=axis)


def _concat_tp_interleaved(
    values: list[Any],
    *,
    tensor_parallel_size: int,
    axis: int = 0,
) -> np.ndarray:
    arrays = [_host_array(value) for value in values]
    if tensor_parallel_size <= 1:
        return np.concatenate(arrays, axis=axis)

    split_arrays = [np.array_split(array, tensor_parallel_size, axis=axis) for array in arrays]
    interleaved = []
    for shard_index in range(tensor_parallel_size):
        for array_shards in split_arrays:
            interleaved.append(array_shards[shard_index])
    return np.concatenate(interleaved, axis=axis)


def _is_scan_layout(weights: dict[str, Any]) -> bool:
    return "input_layernorm.weight" in weights and not any(
        key.startswith("model.layers.") for key in weights
    )


def _infer_scan_num_layers(weights: dict[str, Any]) -> int:
    for key in _SCAN_LAYER_SUFFIXES:
        value = weights.get(key)
        if value is not None:
            return int(_host_array(value).shape[0])
    raise ValueError("Unable to infer Gemma3 scan layer count from weights.")


def _dense_layer_ids(weights: dict[str, Any]) -> list[int]:
    layer_ids = {
        int(match.group(1))
        for key in weights
        if (match := _DENSE_LAYER_PATTERN.match(key)) is not None
    }
    if not layer_ids:
        raise ValueError("No Gemma3 dense layer weights were found.")
    return sorted(layer_ids)


def _pack_dense_gemma3_weights(
    weights: dict[str, Any],
    *,
    target_prefix: str,
    tensor_parallel_size: int,
) -> dict[str, np.ndarray]:
    packed: dict[str, np.ndarray] = {
        f"{target_prefix}.model.embed_tokens.weight": _host_array(
            weights["model.embed_tokens.weight"]
        ),
        f"{target_prefix}.model.norm.weight": _host_array(weights["model.norm.weight"]),
    }
    if "lm_head.weight" in weights:
        packed[f"{target_prefix}.lm_head.weight"] = _host_array(weights["lm_head.weight"])

    for layer_id in _dense_layer_ids(weights):
        src = f"model.layers.{layer_id}."
        dst = f"{target_prefix}.model.layers.{layer_id}."
        packed[f"{dst}input_layernorm.weight"] = _host_array(
            weights[f"{src}input_layernorm.weight"]
        )
        packed[f"{dst}post_attention_layernorm.weight"] = _host_array(
            weights[f"{src}post_attention_layernorm.weight"]
        )
        packed[f"{dst}pre_feedforward_layernorm.weight"] = _host_array(
            weights[f"{src}pre_feedforward_layernorm.weight"]
        )
        packed[f"{dst}post_feedforward_layernorm.weight"] = _host_array(
            weights[f"{src}post_feedforward_layernorm.weight"]
        )
        packed[f"{dst}self_attn.q_norm.weight"] = _host_array(
            weights[f"{src}self_attn.q_norm.weight"]
        )
        packed[f"{dst}self_attn.k_norm.weight"] = _host_array(
            weights[f"{src}self_attn.k_norm.weight"]
        )
        packed[f"{dst}self_attn.o_proj.weight"] = _host_array(
            weights[f"{src}self_attn.o_proj.weight"]
        )
        packed[f"{dst}mlp.down_proj.weight"] = _host_array(
            weights[f"{src}mlp.down_proj.weight"]
        )
        packed[f"{dst}self_attn.qkv_proj.weight"] = _concat_tp_interleaved(
            [
                weights[f"{src}self_attn.q_proj.weight"],
                weights[f"{src}self_attn.k_proj.weight"],
                weights[f"{src}self_attn.v_proj.weight"],
            ],
            tensor_parallel_size=tensor_parallel_size,
        )
        packed[f"{dst}mlp.gate_up_proj.weight"] = _concat_tp_interleaved(
            [weights[f"{src}mlp.gate_proj.weight"], weights[f"{src}mlp.up_proj.weight"]],
            tensor_parallel_size=tensor_parallel_size,
        )

        if f"{src}self_attn.o_proj.bias" in weights:
            packed[f"{dst}self_attn.o_proj.bias"] = _host_array(
                weights[f"{src}self_attn.o_proj.bias"]
            )
        if f"{src}mlp.down_proj.bias" in weights:
            packed[f"{dst}mlp.down_proj.bias"] = _host_array(
                weights[f"{src}mlp.down_proj.bias"]
            )
        if f"{src}self_attn.q_proj.bias" in weights:
            packed[f"{dst}self_attn.qkv_proj.bias"] = _concat_tp_interleaved(
                [
                    weights[f"{src}self_attn.q_proj.bias"],
                    weights[f"{src}self_attn.k_proj.bias"],
                    weights[f"{src}self_attn.v_proj.bias"],
                ],
                tensor_parallel_size=tensor_parallel_size,
            )
        if f"{src}mlp.gate_proj.bias" in weights:
            packed[f"{dst}mlp.gate_up_proj.bias"] = _concat_tp_interleaved(
                [weights[f"{src}mlp.gate_proj.bias"], weights[f"{src}mlp.up_proj.bias"]],
                tensor_parallel_size=tensor_parallel_size,
            )

    return packed


def _pack_scan_gemma3_weights(
    weights: dict[str, Any],
    *,
    target_prefix: str,
    tensor_parallel_size: int,
) -> dict[str, np.ndarray]:
    num_layers = _infer_scan_num_layers(weights)
    packed: dict[str, np.ndarray] = {
        f"{target_prefix}.model.embed_tokens.weight": _host_array(
            weights["model.embed_tokens.weight"]
        ),
        f"{target_prefix}.model.norm.weight": _host_array(weights["model.norm.weight"]),
    }
    if "lm_head.weight" in weights:
        packed[f"{target_prefix}.lm_head.weight"] = _host_array(weights["lm_head.weight"])

    for layer_id in range(num_layers):
        dst = f"{target_prefix}.model.layers.{layer_id}."
        packed[f"{dst}input_layernorm.weight"] = _host_array(
            weights["input_layernorm.weight"][layer_id]
        )
        packed[f"{dst}post_attention_layernorm.weight"] = _host_array(
            weights["post_attention_layernorm.weight"][layer_id]
        )
        packed[f"{dst}pre_feedforward_layernorm.weight"] = _host_array(
            weights["pre_feedforward_layernorm.weight"][layer_id]
        )
        packed[f"{dst}post_feedforward_layernorm.weight"] = _host_array(
            weights["post_feedforward_layernorm.weight"][layer_id]
        )
        packed[f"{dst}self_attn.q_norm.weight"] = _host_array(
            weights["self_attn.q_norm.weight"][layer_id]
        )
        packed[f"{dst}self_attn.k_norm.weight"] = _host_array(
            weights["self_attn.k_norm.weight"][layer_id]
        )
        packed[f"{dst}self_attn.o_proj.weight"] = _host_array(
            weights["self_attn.o_proj.weight"][layer_id]
        )
        packed[f"{dst}mlp.down_proj.weight"] = _host_array(
            weights["mlp.down_proj.weight"][layer_id]
        )
        packed[f"{dst}self_attn.qkv_proj.weight"] = _concat_tp_interleaved(
            [
                weights["self_attn.q_proj.weight"][layer_id],
                weights["self_attn.k_proj.weight"][layer_id],
                weights["self_attn.v_proj.weight"][layer_id],
            ],
            tensor_parallel_size=tensor_parallel_size,
        )
        packed[f"{dst}mlp.gate_up_proj.weight"] = _concat_tp_interleaved(
            [
                weights["mlp.gate_proj.weight"][layer_id],
                weights["mlp.up_proj.weight"][layer_id],
            ],
            tensor_parallel_size=tensor_parallel_size,
        )

        if "self_attn.o_proj.bias" in weights:
            packed[f"{dst}self_attn.o_proj.bias"] = _host_array(
                weights["self_attn.o_proj.bias"][layer_id]
            )
        if "mlp.down_proj.bias" in weights:
            packed[f"{dst}mlp.down_proj.bias"] = _host_array(
                weights["mlp.down_proj.bias"][layer_id]
            )
        if "self_attn.q_proj.bias" in weights:
            packed[f"{dst}self_attn.qkv_proj.bias"] = _concat_tp_interleaved(
                [
                    weights["self_attn.q_proj.bias"][layer_id],
                    weights["self_attn.k_proj.bias"][layer_id],
                    weights["self_attn.v_proj.bias"][layer_id],
                ],
                tensor_parallel_size=tensor_parallel_size,
            )
        if "mlp.gate_proj.bias" in weights:
            packed[f"{dst}mlp.gate_up_proj.bias"] = _concat_tp_interleaved(
                [
                    weights["mlp.gate_proj.bias"][layer_id],
                    weights["mlp.up_proj.bias"][layer_id],
                ],
                tensor_parallel_size=tensor_parallel_size,
            )

    return packed


def pack_gemma3_for_vllm_sync(
    weights_or_model: Model | dict[str, Any],
    *,
    target_prefix: str = "vllm_model",
    tensor_parallel_size: int = 1,
) -> dict[str, np.ndarray]:
    weights = (
        weights_or_model.weights
        if isinstance(weights_or_model, Model)
        else weights_or_model
    )
    if _is_scan_layout(weights):
        return _pack_scan_gemma3_weights(
            weights,
            target_prefix=target_prefix,
            tensor_parallel_size=tensor_parallel_size,
        )
    return _pack_dense_gemma3_weights(
        weights,
        target_prefix=target_prefix,
        tensor_parallel_size=tensor_parallel_size,
    )


def _flatten_target_state(target_state: Any) -> dict[str, Any]:
    if hasattr(target_state, "flat_state"):
        return {
            ".".join(str(part) for part in path): leaf.value
            for path, leaf in target_state.flat_state()
        }
    if isinstance(target_state, dict):
        if all(isinstance(key, str) for key in target_state):
            return target_state
    raise TypeError(
        "Expected a vLLM target state with `flat_state()` or a flat `dict[str, tensor]`, "
        f"got {type(target_state)!r}."
    )


def _assign_target_value(target_state: Any, key: str, value: Any) -> None:
    if isinstance(target_state, dict) and key in target_state:
        target_state[key] = value
        return
    raise TypeError(
        "Only flat `dict[str, tensor]` target states are supported for in-process sync, "
        f"got {type(target_state)!r}."
    )


def reshard_like_vllm_state(updated_weights: SyncState, target_state: Any) -> SyncState:
    target_flat = _flatten_target_state(target_state)
    resharded: dict[str, Any] = {}
    for key, value in updated_weights.tensors.items():
        target_value = target_flat.get(key)
        if target_value is None:
            continue
        array = np.asarray(value, dtype=np.asarray(target_value).dtype)
        sharding = getattr(target_value, "sharding", None)
        if sharding is not None:
            resharded[key] = jax.device_put(array, sharding)
        else:
            resharded[key] = array
    return SyncState(tensors=resharded)


def sync_vllm_state_in_place(
    updated_weights: SyncState,
    target_state: Any,
    mappings: dict[str, tuple[str, tuple[str, ...]]],
    transpose_keys: dict[str, tuple[int, ...]],
) -> Any:
    target_flat = _flatten_target_state(target_state)
    for source_key, value in updated_weights.tensors.items():
        mapping = mappings.get(source_key)
        if mapping is None:
            continue
        target_key = mapping[0]
        target_value = target_flat.get(target_key)
        if target_value is None:
            continue

        array = np.asarray(value)
        leaf_name = source_key.rsplit(".", 1)[-1]
        if leaf_name in transpose_keys:
            array = np.transpose(array, transpose_keys[leaf_name])

        expected_shape = np.asarray(target_value).shape
        if expected_shape != array.shape:
            raise ValueError(
                f"Shape mismatch for {source_key}: expected {expected_shape}, got {array.shape}."
            )

        target_dtype = np.asarray(target_value).dtype
        if array.dtype != target_dtype:
            array = array.astype(target_dtype)

        sharding = getattr(target_value, "sharding", None)
        synced_value = jax.device_put(array, sharding) if sharding is not None else array
        _assign_target_value(target_state, target_key, synced_value)
    return target_state


def build_gemma3_vllm_sync_payload(
    weights_or_model: Model | dict[str, Any],
    *,
    target_prefix: str = "vllm_model",
    tensor_parallel_size: int = 1,
) -> VllmSyncPayload:
    packed = pack_gemma3_for_vllm_sync(
        weights_or_model,
        target_prefix=target_prefix,
        tensor_parallel_size=tensor_parallel_size,
    )
    mappings = {key: (key, ()) for key in packed}
    return VllmSyncPayload(
        updated_weights=SyncState(tensors=packed),
        mappings=mappings,
        transpose_keys={},
    )
