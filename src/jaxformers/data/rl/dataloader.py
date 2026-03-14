from __future__ import annotations

import dataclasses as dc
from collections.abc import Iterator, Sequence
from typing import Any

import grain
import jax
import numpy as np
from grain import IterDataset, MapDataset, transforms as grain_transforms

from jaxformers.data.rl.reward_postprocess import make_reward_postprocessor
from jaxformers.data.rl.tasks.base import DatasetWithReward
from jaxformers.data.rl_tokenize_transforms import (
    make_tokenize_rl_input,
    TokenizeRLInput,
)


@dc.dataclass
class _AttachTaskName(grain_transforms.Map):
    task_name: str

    def map(self, features: dict[str, Any]) -> dict[str, Any]:
        output = dict(features)
        output["__task_name__"] = self.task_name
        return output


def _coerce_reward_result(result: float | dict[str, Any]) -> tuple[float, bool]:
    if isinstance(result, dict):
        reward = float(result.get("reward", 0.0))
        correct = bool(result.get("correct", reward > 0))
        return reward, correct
    reward = float(result)
    return reward, reward > 0


def _prepare_dataset(
    dataset: IterDataset | MapDataset,
    *,
    task_name: str,
    seed: int,
    shuffle: bool,
    shuffle_buffer_size: int,
    num_epochs: int | None,
    dataloading_host_index: int,
    dataloading_host_count: int,
    is_not_sharded: bool,
    read_num_threads: int,
    read_prefetch_buffer_size: int,
) -> IterDataset:
    if dataloading_host_count > 1 and is_not_sharded:
        dataset = dataset.shard(
            num_shards=dataloading_host_count,
            index=dataloading_host_index,
            contiguous=True,
        )

    if shuffle:
        if isinstance(dataset, MapDataset):
            dataset = dataset.shuffle(seed=seed + dataloading_host_index)
        else:
            dataset = dataset.shuffle(
                seed=seed + dataloading_host_index,
                buffer_size=shuffle_buffer_size,
            )

    if num_epochs is not None:
        dataset = dataset.repeat(num_epochs)

    if isinstance(dataset, MapDataset):
        dataset = dataset.to_iter_dataset(
            grain.ReadOptions(
                num_threads=read_num_threads,
                prefetch_buffer_size=read_prefetch_buffer_size,
            )
        )

    if not isinstance(dataset, IterDataset):
        raise TypeError(f"Unsupported dataset type: {type(dataset)!r}")

    return dataset.map(_AttachTaskName(task_name))


class RLDataLoader(Iterator[tuple[dict[str, np.ndarray], dict[str, float]]]):
    def __init__(
        self,
        dataset: IterDataset,
        *,
        task_map: dict[str, DatasetWithReward],
        rollout_engine,
        tokenizer_transform: TokenizeRLInput,
        reward_name: str | None,
        reward_config: dict | None,
        global_batch_size: int,
        seed: int = 0,
        drop_remainder: bool = True,
    ) -> None:
        self._dataset = dataset
        self._iterator = iter(dataset)
        self._task_map = task_map
        self._rollout_engine = rollout_engine
        self._tokenizer_transform = tokenizer_transform
        self._reward_postprocessor = make_reward_postprocessor(
            reward_name,
            reward_config,
        )
        self._drop_remainder = drop_remainder
        self._rng = np.random.default_rng(seed)

        num_samples = max(self._rollout_engine.rollout_params.num_samples_per_example, 1)
        if global_batch_size % num_samples != 0:
            raise ValueError(
                "global_batch_size must be divisible by "
                "rollout.num_samples_per_example"
            )
        self._prompt_batch_size = global_batch_size // num_samples
        self._global_batch_size = global_batch_size

    def set_params(self, params) -> None:
        self._rollout_engine.set_params(params)

    def __iter__(self) -> RLDataLoader:
        return self

    def __next__(self) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        if self._rollout_engine.params is None:
            raise RuntimeError("RLDataLoader requires rollout params to be set first.")

        raw_examples: list[dict[str, Any]] = []
        while len(raw_examples) < self._prompt_batch_size:
            try:
                raw_examples.append(next(self._iterator))
            except StopIteration:
                if self._drop_remainder or not raw_examples:
                    raise
                break

        formatted_examples: list[dict[str, Any]] = []
        for raw_example in raw_examples:
            task_name = raw_example["__task_name__"]
            task = self._task_map[task_name]
            formatted = self._tokenizer_transform.map(task.format_example(raw_example))
            formatted["__task_name__"] = task_name
            formatted_examples.append(formatted)

        rollout_key = int(
            self._rng.integers(
                0,
                np.iinfo(np.uint32).max,
                dtype=np.uint32,
            )
        )
        rollout_batch = self._rollout_engine.generate(
            formatted_examples,
            prng_key=rollout_key,
        )

        rewards = np.zeros((len(rollout_batch.samples),), dtype=np.float32)
        correct = np.zeros((len(rollout_batch.samples),), dtype=np.float32)
        for sample_index, sample in enumerate(rollout_batch.samples):
            raw_example = raw_examples[sample.prompt_index]
            task = self._task_map[str(sample.name)]
            reward_value, is_correct = _coerce_reward_result(task.reward(raw_example, sample))
            rewards[sample_index] = reward_value
            correct[sample_index] = float(is_correct)

        processed_rewards = self._reward_postprocessor(
            rewards,
            group_ids=rollout_batch.prompt_indices,
        ).astype(np.float32, copy=False)

        batch = {
            "token_ids": rollout_batch.token_ids.astype(np.int32, copy=False),
            "attention_mask": rollout_batch.attention_mask.astype(bool, copy=False),
            "generation_mask": rollout_batch.generation_mask.astype(bool, copy=False),
            "logprobs": rollout_batch.logprobs.astype(np.float32, copy=False),
            "rewards": processed_rewards,
        }
        metrics = {
            "rollout/reward_mean": float(rewards.mean()) if len(rewards) else 0.0,
            "rollout/reward_std": float(rewards.std()) if len(rewards) else 0.0,
            "rollout/correct_mean": float(correct.mean()) if len(correct) else 0.0,
            "rollout/output_len_mean": float(rollout_batch.output_lengths.mean())
            if len(rollout_batch.output_lengths)
            else 0.0,
        }
        return batch, metrics


def make_rl_data_loader(
    *,
    tasks: Sequence[DatasetWithReward],
    rollout_engine,
    reward_name: str | None = None,
    reward_config: dict | None = None,
    loader_config: dict | None = None,
    tokenizer_transform: TokenizeRLInput | None = None,
) -> RLDataLoader:
    if not tasks:
        raise ValueError("make_rl_data_loader requires at least one task.")

    loader_config = loader_config or {}
    tokenizer_transform = tokenizer_transform or make_tokenize_rl_input(
        tokenizer=rollout_engine.model.tokenizer,
    )

    dataloading_host_index = loader_config.get(
        "dataloading_host_index",
        jax.process_index(),
    )
    dataloading_host_count = loader_config.get(
        "dataloading_host_count",
        jax.process_count(),
    )

    prepared = [
        _prepare_dataset(
            task.dataset,
            task_name=task.name,
            seed=loader_config.get("seed", 0),
            shuffle=loader_config.get("shuffle", True),
            shuffle_buffer_size=loader_config.get("shuffle_buffer_size", 1000),
            num_epochs=loader_config.get("num_epochs"),
            dataloading_host_index=dataloading_host_index,
            dataloading_host_count=dataloading_host_count,
            is_not_sharded=loader_config.get("is_not_sharded", True),
            read_num_threads=loader_config.get("read_num_threads", 0),
            read_prefetch_buffer_size=loader_config.get(
                "read_prefetch_buffer_size",
                0,
            ),
        )
        for task in tasks
    ]

    if len(prepared) == 1:
        mixed = prepared[0]
    else:
        mixed = grain.IterDataset.mix(
            prepared,
            weights=loader_config.get("dataset_weights"),
        )

    return RLDataLoader(
        mixed,
        task_map={task.name: task for task in tasks},
        rollout_engine=rollout_engine,
        tokenizer_transform=tokenizer_transform,
        reward_name=reward_name,
        reward_config=reward_config,
        global_batch_size=loader_config["global_batch_size"],
        seed=loader_config.get("seed", 0),
        drop_remainder=loader_config.get("drop_remainder", True),
    )
