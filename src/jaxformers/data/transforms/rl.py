import dataclasses as dc
from typing import Any

import numpy as np
from grain import transforms as grain_transforms


def _flatten_trajectory_tokens(trajectory: list[dict[str, Any]]):
    if not trajectory:
        raise ValueError("trajectory is empty")

    full_ids: list[int] = []
    predicted_positions: list[tuple[int, float, int]] = []
    first_step = True

    for step in trajectory:
        tokens = step["tokens"]
        if tokens is None:
            raise ValueError("trajectory step is missing tokens")

        prompt_ids = list(tokens["prompt_ids"])
        completion_ids = list(tokens["completion_ids"])
        completion_logprobs = list(tokens["completion_logprobs"])
        completion_mask = (
            list(tokens["completion_mask"])
            if "completion_mask" in tokens
            else [1] * len(completion_ids)
        )
        if len(completion_logprobs) != len(completion_ids):
            raise ValueError("completion_logprobs must match completion_ids length")
        if len(completion_mask) != len(completion_ids):
            raise ValueError("completion_mask must match completion_ids length")

        if first_step:
            full_ids.extend(prompt_ids)
            first_step = False
        else:
            if prompt_ids[: len(full_ids)] != full_ids:
                raise ValueError(
                    "trajectory prompt tokens do not extend the prior prefix"
                )
            full_ids.extend(prompt_ids[len(full_ids) :])

        for completion_logprob, completion_id, should_train in zip(
            completion_logprobs,
            completion_ids,
            completion_mask,
        ):
            predicted_positions.append(
                (
                    len(full_ids) - 1,
                    float(completion_logprob),
                    int(should_train),
                )
            )
            full_ids.append(int(completion_id))

    if len(full_ids) < 2:
        raise ValueError("trajectory must contain at least two tokens")

    return full_ids, predicted_positions


@dc.dataclass
class ToRLDataTransform(grain_transforms.Map):
    def map(self, features: dict[str, Any]) -> dict[str, Any]:
        full_ids, predicted_positions = _flatten_trajectory_tokens(
            features["trajectory"]
        )
        seq_len = len(full_ids) - 1

        input_ids = np.asarray(full_ids[:-1], dtype=np.int32)
        labels = np.asarray(full_ids[1:], dtype=np.int32)
        attention_mask = np.ones((seq_len,), dtype=np.int32)
        loss_mask = np.zeros((seq_len,), dtype=np.float32)
        behavior_logprobs = np.zeros((seq_len,), dtype=np.float32)
        advantages = np.zeros((seq_len,), dtype=np.float32)
        advantage = np.float32(features["advantage"])

        for position, logprob, should_train in predicted_positions:
            behavior_logprobs[position] = np.float32(logprob)
            if should_train:
                loss_mask[position] = 1.0
                advantages[position] = advantage

        if not np.any(loss_mask):
            raise ValueError("trajectory has no trainable completion tokens")

        return {
            "inputs": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            "labels": labels,
            "loss_mask": loss_mask,
            "advantages": advantages,
            "behavior_logprobs": behavior_logprobs,
            "reward": np.float32(features["reward"]),
            "example_id": np.int32(features["example_id"]),
        }


def make():
    return [ToRLDataTransform()]
