import numpy as np
from datasets import Dataset

from jaxformers.data.source.verifiers import make as make_verifiers_source
from jaxformers.data.transforms.rl import ToRLDataTransform
from jaxformers.rl import batch_rl_data


class FakeEnv:
    def __init__(self):
        self.dataset = Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": "hi"}]],
                "example_id": [0],
                "task": ["fake"],
            }
        )

    def build_dataset(self):
        return self.dataset

    async def run_group(
        self,
        group_inputs,
        client,
        model,
        sampling_args,
        max_retries=0,
        state_columns=None,
    ):
        del client, model, sampling_args, max_retries, state_columns
        outputs = []
        for idx, rollout_input in enumerate(group_inputs):
            outputs.append(
                {
                    "trajectory": [
                        {
                            "tokens": {
                                "prompt_ids": [1, 2],
                                "prompt_mask": [0, 0],
                                "completion_ids": [3, 4],
                                "completion_mask": [1, 1],
                                "completion_logprobs": [-0.1, -0.2],
                            }
                        }
                    ],
                    "reward": float(idx),
                    "example_id": rollout_input["example_id"],
                    "task": rollout_input["task"],
                    "is_truncated": False,
                }
            )
        return outputs


class FakeClient:
    model_name = "fake-model"


def test_verifiers_source_group_normalizes_advantages():
    dataset = make_verifiers_source(
        envs=[FakeEnv()],
        client=FakeClient(),
        rollouts_per_example=2,
    )[0]

    iterator = iter(dataset)
    first = next(iterator)
    second = next(iterator)

    assert np.isclose(first["reward"], 0.0)
    assert np.isclose(second["reward"], 1.0)
    assert first["advantage"] < 0
    assert second["advantage"] > 0


def test_rl_transform_reconstructs_multiturn_sequence():
    features = {
        "trajectory": [
            {
                "tokens": {
                    "prompt_ids": [1, 2],
                    "prompt_mask": [0, 0],
                    "completion_ids": [3],
                    "completion_mask": [1],
                    "completion_logprobs": [-0.1],
                }
            },
            {
                "tokens": {
                    "prompt_ids": [1, 2, 3, 5],
                    "prompt_mask": [0, 0, 0, 0],
                    "completion_ids": [6],
                    "completion_mask": [1],
                    "completion_logprobs": [-0.2],
                }
            },
        ],
        "advantage": 1.5,
        "reward": 2.0,
        "example_id": 7,
    }

    transformed = ToRLDataTransform().map(features)

    assert transformed["inputs"]["input_ids"].tolist() == [1, 2, 3, 5]
    assert transformed["labels"].tolist() == [2, 3, 5, 6]
    assert transformed["loss_mask"].tolist() == [0.0, 1.0, 0.0, 1.0]
    np.testing.assert_allclose(
        transformed["behavior_logprobs"],
        np.asarray([0.0, -0.1, 0.0, -0.2], dtype=np.float32),
    )
    np.testing.assert_allclose(
        transformed["advantages"],
        np.asarray([0.0, 1.5, 0.0, 1.5], dtype=np.float32),
    )


def test_rl_transform_respects_completion_mask():
    features = {
        "trajectory": [
            {
                "tokens": {
                    "prompt_ids": [1, 2],
                    "prompt_mask": [0, 0],
                    "completion_ids": [3, 4],
                    "completion_mask": [1, 0],
                    "completion_logprobs": [-0.1, -0.2],
                }
            }
        ],
        "advantage": 2.0,
        "reward": 1.0,
        "example_id": 9,
    }

    transformed = ToRLDataTransform().map(features)

    assert transformed["loss_mask"].tolist() == [0.0, 1.0, 0.0]
    np.testing.assert_allclose(
        transformed["advantages"],
        np.asarray([0.0, 2.0, 0.0], dtype=np.float32),
    )


def test_batch_rl_data_pads_to_tpu_block_size():
    batch = batch_rl_data(
        [
            {
                "inputs": {
                    "input_ids": np.arange(3, dtype=np.int32),
                    "attention_mask": np.ones(3, dtype=np.int32),
                },
                "labels": np.arange(3, dtype=np.int32),
                "loss_mask": np.ones(3, dtype=np.float32),
                "advantages": np.ones(3, dtype=np.float32),
                "behavior_logprobs": np.zeros(3, dtype=np.float32),
                "reward": np.float32(1.0),
                "example_id": np.int32(0),
            }
        ],
        target_len=128,
    )

    assert batch["labels"].shape == (1, 128)
    assert batch["inputs"]["attention_mask"][0, 3:].sum() == 0
