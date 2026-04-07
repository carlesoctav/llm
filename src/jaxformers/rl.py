from __future__ import annotations

import numpy as np
import sws
from functools import partial

from jaxformers.data.source import make_source
from jaxformers.data.transforms import make_transforms
from jaxformers.data.transforms.base import transform_ds


def _pad_1d(array: np.ndarray, target_len: int, pad_value) -> np.ndarray:
    pad_width = target_len - len(array)
    if pad_width == 0:
        return array
    return np.pad(array, (0, pad_width), constant_values=pad_value)


def _round_up_multiple(value: int, multiple: int) -> int:
    remainder = value % multiple
    if remainder == 0:
        return value
    return value + multiple - remainder


def batch_rl_data(values, target_len: int):
    for value in values:
        value_len = len(value["labels"])
        if value_len > target_len:
            raise ValueError(
                f"RL sequence length {value_len} exceeds fixed batch length {target_len}."
            )
    return {
        "inputs": {
            "input_ids": np.stack(
                [
                    _pad_1d(value["inputs"]["input_ids"], target_len, 0)
                    for value in values
                ]
            ),
            "attention_mask": np.stack(
                [
                    _pad_1d(value["inputs"]["attention_mask"], target_len, 0)
                    for value in values
                ]
            ),
        },
        "labels": np.stack(
            [_pad_1d(value["labels"], target_len, 0) for value in values]
        ),
        "loss_mask": np.stack(
            [_pad_1d(value["loss_mask"], target_len, 0.0) for value in values]
        ),
        "advantages": np.stack(
            [_pad_1d(value["advantages"], target_len, 0.0) for value in values]
        ),
        "behavior_logprobs": np.stack(
            [
                _pad_1d(value["behavior_logprobs"], target_len, 0.0)
                for value in values
            ]
        ),
        "reward": np.asarray([value["reward"] for value in values], dtype=np.float32),
        "example_id": np.asarray(
            [value["example_id"] for value in values],
            dtype=np.int32,
        ),
    }


def make_rl_data(config: sws.FinalConfig, llm_client):
    source_config = config.data.source.to_dict()
    source_config["client"] = llm_client
    source = make_source(config.data.source_name, source_config)[0]
    transforms = make_transforms("rl", {})
    dataset = transform_ds(source, *transforms)
    loader_config = config.data.loader.to_dict()
    target_len = _round_up_multiple(config.vllm.max_num_batched_tokens, 128)
    dataset = dataset.batch(
        loader_config["batch_size"],
        batch_fn=partial(batch_rl_data, target_len=target_len),
    )
    return dataset
