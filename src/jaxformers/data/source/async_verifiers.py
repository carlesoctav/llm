from __future__ import annotations

import asyncio
import copy
import threading
from collections import deque
from concurrent.futures import Future as ConFuture, wait as con_wait
from typing import Any, TypeVar

import datasets as hf_datasets
import grain
import numpy as np
import verifiers as vf


VfArgs = TypeVar("VfArgs")


class AsyncLoopThread:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="rollout_async_loop"
        )
        self._thread.start()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro) -> ConFuture:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def close(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout = 1)
        self._loop.close()


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


class VerifiersIterator(grain.DatasetIterator):
    def __init__(
        self,
        envs: dict[str, vf.Environment],
        datasets: dict[str, hf_datasets.Dataset],
        client: vf.Client,
        rollouts_per_example: int,
        sampling_args: dict[str, Any],
        seed: int = 42,
        env_weights: list[float] | None = None,
        max_retries: int = 3,
    ):
        super().__init__()
        self._envs = envs
        self._datasets = datasets
        self._client = client
        self._rollouts_per_example = rollouts_per_example
        self._sampling_args = sampling_args
        self._env_weights = env_weights
        self._env_names = list(self._envs.keys())

        self._rng = np.random.default_rng(seed)
        self._max_retries = max_retries
        self._buffer = deque()
        self._max_inflight_requests = 10
        self._executor = AsyncLoopThread()
        self._futures: set[ConFuture] = set()

        self._rollout_counter = 0

    def fill_inlfight_queue(self):
        diff = max(self._max_inflight_requests - len(self._futures), 0)
        for _ in range(diff):
            fut = self._executor.submit(self._fetch_next_group())
            self._futures.add(fut)

    def _sample_group_inputs(self) -> tuple[vf.Environment, dict]:
        selected = self._rng.choice(self._env_names, p=self._env_weights)
        dataset, env = self._datasets[selected], self._envs[selected]
        row_index = int(self._rng.integers(len(dataset)))
        example = dataset[row_index]

        if "task" not in example:
            example["task"] = selected
        if "example_id" not in example:
            example["example_id"] = row_index

        return env, example

    async def _fetch_next_group(self) -> list[dict[str, Any]]:
        env, example = self._sample_group_inputs()
        group_inputs = [
            copy.deepcopy(example) for _ in range(self._rollouts_per_example)
        ]
        states = await env.run_group(
            group_inputs,
            self._client,
            self._client.model_name,
            self._sampling_args,
            max_retries=self._max_retries,
            state_columns=["trajectory"],
        )
        rewards = [state["reward"] for state in states]
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
        return outputs

    def __next__(self):
        self.fill_inlfight_queue()

        while not self._buffer:
            done, pending = con_wait(self._futures, return_when="FIRST_COMPLETED")
            self._futures = pending
            for fut in done:
                self._buffer.extend(fut.result())

        self._rollout_counter+=1
        return self._buffer.popleft()

    def get_state(self):
        return {
            # "bit_generator_state": copy.deepcopy(self._rng.bit_generator.state),
            # "buffer": copy.deepcopy(self._buffer),
        }

    def set_state(self, state):
        pass
        # self._rng.bit_generator.state = state["bit_generator_state"]
        # self._buffer = copy.deepcopy(state["buffer"])
        # self._future.cancel()
        # self._future = self._submit_next_group()

    def close(self):
        for future in self._futures:
            future.cancel()
        self._futures.clear()
        self._executor.close()
        super().close()

    def __del__(self):
        self.close()


class VerifiersSourceIterDataset(grain.IterDataset):
    def __init__(
        self,
        envs: dict[str, VfArgs],
        client: vf.Client,
        rollouts_per_example: int,
        sampling_args: dict[str, Any],
        env_weights: list[float] | None = None,
        seed: int = 42,
        max_retries: int = 3,
    ):
        super().__init__()
        self._envs = {}
        self._datasets = {}
        for k, v in envs.items():
            env = vf.load_environment(**v)
            self._envs[k] = env
            self._datasets[k] = env.get_dataset()

        self._client = client
        self._rollouts_per_example = rollouts_per_example
        self._sampling_args = sampling_args
        self._env_weights = env_weights
        self._seed = seed
        self._max_retries = max_retries

    def __iter__(self):
        return VerifiersIterator(
            envs=self._envs,
            datasets=self._datasets,
            client=self._client,
            rollouts_per_example=self._rollouts_per_example,
            sampling_args=self._sampling_args,
            env_weights=self._env_weights,
            seed=self._seed,
            max_retries=self._max_retries,
        )

    def __str__(self) -> str:
        return "VerifiersDataset"


def make(
    envs: dict[str, VfArgs],
    client,
    rollouts_per_example: int = 4,
    sampling_args: dict[str, Any] | None = None,
    env_weights: list[float] | None = None,
    seed: int = 0,
    max_retries: int = 0,
):
    if env_weights is not None and len(env_weights) != len(envs):
        raise ValueError("env_weights length must match envs length.")
    if rollouts_per_example <= 0:
        raise ValueError("rollouts_per_example must be positive.")
    return [
        VerifiersSourceIterDataset(
            envs=envs,
            client=client,
            rollouts_per_example=rollouts_per_example,
            sampling_args=sampling_args or {},
            env_weights=env_weights,
            seed=seed,
            max_retries=max_retries,
        )
    ]
