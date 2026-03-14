from __future__ import annotations

import dataclasses as dc
from collections.abc import Callable
from typing import Any

from datasets import Dataset, IterableDataset, load_dataset
from grain import IterDataset, MapDataset

from jaxformers.data.huggingface import (
    HuggingFaceSourceIterDataset,
    HuggingFaceSourceMapDataset,
)


@dc.dataclass(frozen=True)
class DatasetWithReward:
    name: str
    dataset: IterDataset | MapDataset
    format_example_fn: Callable[[dict[str, Any]], dict[str, Any]]
    reward_fn: Callable[[dict[str, Any], Any], float | dict[str, Any]]

    def format_example(self, example: dict[str, Any]) -> dict[str, Any]:
        return self.format_example_fn(example)

    def reward(self, example: dict[str, Any], rollout_sample: Any) -> float | dict[str, Any]:
        return self.reward_fn(example, rollout_sample)


def load_huggingface_dataset(load_kwargs: dict[str, Any]) -> IterDataset | MapDataset:
    dataset = load_dataset(**load_kwargs)
    if isinstance(dataset, IterableDataset):
        return HuggingFaceSourceIterDataset(dataset)
    if isinstance(dataset, Dataset):
        return HuggingFaceSourceMapDataset(dataset)
    raise TypeError(
        "Expected a HuggingFace Dataset or IterableDataset, "
        f"got {type(dataset)!r}."
    )
