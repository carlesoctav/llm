import typing as tp
import warnings
from typing import Any

import grain
from datasets import Dataset, IterableDataset, load_dataset


Batch = tp.Any
_T = tp.TypeVar("_T")
_S = tp.TypeVar("_S")


class _HuggingFaceSourceIterator(grain.DatasetIterator):
    def __init__(self, dataset: IterableDataset):
        super().__init__()
        self._dataset = dataset
        self._iterator = iter(self._dataset)

    def __next__(self):
        return next(self._iterator)

    def get_state(self):
        return self._dataset.state_dict()

    def set_state(self, state):
        self._dataset.load_state_dict(state)
        self._iterator = iter(self._dataset)


class HuggingFaceSourceIterDataset(grain.IterDataset):
    def __init__(
        self,
        source: IterableDataset,
    ):
        super().__init__()
        self._source = source

    def __iter__(self) -> grain.DatasetIterator:
        return _HuggingFaceSourceIterator(self._source)

    def __str__(self) -> str:
        return "HuggingFaceIterableDataset"

    def repeat(self, num_epochs):
        return HuggingFaceSourceIterDataset(self._source.repeat(num_epochs))

    def shard(
        self,
        num_shards: int,
        index: int,
        contiguous: bool = True,
    ):
        return HuggingFaceSourceIterDataset(
            self._source.shard(num_shards, index, contiguous)
        )

    def set_slice(self, sl: slice, sequential_slice: bool = True) -> None:

        # sl.step is num of worker
        # (1, 2, 4)
        if sl.step is None or sl.step <= 0:
            raise ValueError("slice.step (num_workers) must be a positive integer.")
        worker_index = 0 if sl.start is None else sl.start
        contiguous = bool(sequential_slice)

        if self._source.num_shards < sl.step:
            warnings.warn(
                "The number of shards in the HuggingFace dataset is smaller than the number of workers. Some workers will not receive any data."
            )

        self._source = self._source.shard(
            num_shards=sl.step,
            index=worker_index,
            contiguous=contiguous,
        )

    def shuffle(
        self,
        seed: int | None = None,
        buffer_size: int | None = 1000,
    ) -> "HuggingFaceSourceIterDataset":
        return HuggingFaceSourceIterDataset(
            self._source.shuffle(seed=seed, buffer_size=buffer_size)
        )


class HuggingFaceSourceMapDataset(grain.MapDataset):
    def __init__(self, source: Dataset):
        super().__init__()
        self._source = source

    def __len__(self) -> int:
        return len(self._source)

    def __str__(self) -> str:
        return "HuggingFaceMapDataset"

    def repeat(self, num_epochs):
        return HuggingFaceSourceMapDataset(self._source.repeat(num_epochs))

    def slice(self, sl: slice) -> "HuggingFaceSourceMapDataset":
        start, stop, step = sl.indices(len(self._source))
        if step == 1:  # [ start: end]
            return HuggingFaceSourceMapDataset(self._source.select(range(start, stop)))
        if stop == len(self._source) and start < step:
            return HuggingFaceSourceMapDataset(
                self._source.shard(num_shards=step, index=start, contiguous=False)
            )

        return HuggingFaceSourceMapDataset(
            self._source.select(range(start, stop, step))
        )

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.slice(index)
        return self._source[index % len(self)]

    def shard(
        self,
        num_shards: int,
        index: int,
        contiguous: bool = True,
    ) -> "HuggingFaceSourceMapDataset":
        return HuggingFaceSourceMapDataset(
            self._source.shard(
                num_shards=num_shards, index=index, contiguous=contiguous
            )
        )

    def set_slice(
        self,
        sl: slice,
        sequential_slice: bool = True,
    ) -> None:
        if sl.step is None or sl.step <= 0:
            raise ValueError("slice.step (num_workers) must be a positive integer.")
        worker_index = 0 if sl.start is None else sl.start
        self._source = self._source.shard(
            num_shards=sl.step,
            index=worker_index,
            contiguous=sequential_slice,
        )


def load(load_kwargs: list[dict[str, Any]]):
    datasets = []
    for load_kwarg in load_kwargs:
        dataset = load_dataset(**load_kwarg)
        if isinstance(dataset, IterableDataset):
            datasets.append(HuggingFaceSourceIterDataset(dataset))
        elif isinstance(dataset, Dataset):
            datasets.append(HuggingFaceSourceMapDataset(dataset))
        else:
            raise ValueError(
                f"dataset must be IterableDataset or Dataset, got {type(dataset)}"
            )
    return datasets
