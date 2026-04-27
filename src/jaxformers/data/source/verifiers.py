from __future__ import annotations

import asyncio
import copy
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import grain
import numpy as np


def _get_env_dataset(env):
    if hasattr(env, "build_dataset"):
        dataset = env.build_dataset()
        if dataset is not None:
            return dataset
    if hasattr(env, "get_dataset"):
        return env.get_dataset()
    dataset = getattr(env, "dataset", None)
    if dataset is None:
        raise ValueError(
            "verifiers env must expose build_dataset(), get_dataset(), or dataset"
        )
    return dataset


def _compute_advantages(rewards: list[float]) -> list[float]:
    rewards_array = np.asarray(rewards, dtype=np.float32)
    if rewards_array.size == 0:
        return []
    reward_std = float(rewards_array.std())
    if reward_std == 0.0:
        centered = rewards_array - rewards_array.mean()
        return centered.astype(np.float32).tolist()
    advantages = (rewards_array - rewards_array.mean()) / (reward_std + 1e-6)
    return advantages.astype(np.float32).tolist()


def _normalize_env_weights(env_weights: list[float] | None):
    if env_weights is None:
        return None
    weights = np.asarray(env_weights, dtype=np.float64)
    if np.any(weights < 0):
        raise ValueError("env_weights must be non-negative.")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("env_weights must sum to a positive value.")
    return (weights / total).tolist()


class VerifiersIterator(grain.DatasetIterator):
    def __init__(
        self,
        envs: list[Any],
        env_names: list[str],
        client,
        rollouts_per_example: int,
        sampling_args: dict[str, Any],
        env_weights: list[float] | None,
        seed: int,
        max_retries: int,
    ):
        super().__init__()
        self._envs = envs
        self._env_names = env_names
        self._datasets = [_get_env_dataset(env) for env in envs]
        self._client = client
        self._rollouts_per_example = rollouts_per_example
        self._sampling_args = sampling_args
        self._env_weights = env_weights
        self._rng = np.random.default_rng(seed)
        self._max_retries = max_retries
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._buffer: list[dict[str, Any]] = []
        self._future = self._submit_next_group()
        self._max_inflight_requests = 10

    def _submit_next_group(self) -> Future[list[dict[str, Any]]]:
        return self._executor.submit(self._fetch_next_group)

    def _sample_group_inputs(self):
        env_index = int(
            self._rng.choice(len(self._envs), p=self._env_weights)
            if self._env_weights is not None
            else self._rng.integers(len(self._envs))
        )
        dataset = self._datasets[env_index]
        if len(dataset) == 0:
            raise ValueError(
                f"Dataset for env {self._env_names[env_index]!r} is empty."
            )

        row_index = int(self._rng.integers(len(dataset)))
        example = copy.deepcopy(dataset[row_index])
        if "task" not in example:
            example["task"] = self._env_names[env_index]
        if "example_id" not in example:
            example["example_id"] = row_index
        return self._envs[env_index], example

    def _fetch_next_group(self) -> list[dict[str, Any]]:
        env, example = self._sample_group_inputs()
        group_inputs = [
            copy.deepcopy(example) for _ in range(self._rollouts_per_example)
        ]
        states = asyncio.run(
            env.run_group(
                group_inputs,
                self._client,
                self._client.model_name,
                self._sampling_args,
                max_retries=self._max_retries,
                state_columns=["trajectory"],
            )
        )
        rewards = [float(state["reward"] or 0.0) for state in states]
        advantages = _compute_advantages(rewards)
        outputs = []
        for state, advantage in zip(states, advantages):
            outputs.append(
                {
                    "trajectory": state["trajectory"],
                    "reward": np.float32(state["reward"] or 0.0),
                    "advantage": np.float32(advantage),
                    "example_id": state["example_id"],
                    "task": state["task"],
                    "is_truncated": state["is_truncated"],
                    "stop_condition": state.get("stop_condition"),
                }
            )
        # print("outputs",outputs)
        return outputs

    def __next__(self):
        if not self._buffer:
            self._buffer = self._future.result()
            self._future = self._submit_next_group()
        return self._buffer.pop(0)

    def get_state(self):
        return {
            "bit_generator_state": copy.deepcopy(self._rng.bit_generator.state),
            "buffer": copy.deepcopy(self._buffer),
        }

    def set_state(self, state):
        self._rng.bit_generator.state = state["bit_generator_state"]
        self._buffer = copy.deepcopy(state["buffer"])
        self._future.cancel()
        self._future = self._submit_next_group()

    def __del__(self):
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


class VerifiersDataset(grain.IterDataset):
    def __init__(
        self,
        envs: list[Any],
        env_names: list[str],
        client,
        rollouts_per_example: int,
        sampling_args: dict[str, Any],
        env_weights: list[float] | None,
        seed: int,
        max_retries: int,
    ):
        super().__init__()
        self._envs = envs
        self._env_names = env_names
        self._client = client
        self._rollouts_per_example = rollouts_per_example
        self._sampling_args = sampling_args
        self._env_weights = env_weights
        self._seed = seed
        self._max_retries = max_retries

    def __iter__(self):
        return VerifiersIterator(
            envs=self._envs,
            env_names=self._env_names,
            client=self._client,
            rollouts_per_example=self._rollouts_per_example,
            sampling_args=self._sampling_args,
            env_weights=self._env_weights,
            seed=self._seed,
            max_retries=self._max_retries,
        )

    def __str__(self) -> str:
        return "VerifiersDataset"


VerifiersSourceIterator = VerifiersIterator
VerifiersSourceIterDataset = VerifiersDataset


def make(
    envs: list[Any],
    client,
    env_names: list[str] | None = None,
    rollouts_per_example: int = 4,
    sampling_args: dict[str, Any] | None = None,
    env_weights: list[float] | None = None,
    seed: int = 0,
    max_retries: int = 0,
):
    resolved_env_names = env_names or [f"env_{idx}" for idx in range(len(envs))]
    if len(resolved_env_names) != len(envs):
        raise ValueError("env_names length must match envs length.")
    if env_weights is not None and len(env_weights) != len(envs):
        raise ValueError("env_weights length must match envs length.")
    if rollouts_per_example <= 0:
        raise ValueError("rollouts_per_example must be positive.")
    return [
        VerifiersSourceIterDataset(
            envs=envs,
            env_names=resolved_env_names,
            client=client,
            rollouts_per_example=rollouts_per_example,
            sampling_args=sampling_args or {},
            env_weights=_normalize_env_weights(env_weights),
            seed=seed,
            max_retries=max_retries,
        )
    ]
