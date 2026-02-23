import logging
import time
import typing as tp
import warnings
from collections.abc import Sequence

import grain
import jax
import jax.tree_util as jtu
from datasets import Dataset, IterableDataset
from grain import (
    DatasetIterator,
    IterDataset,
    MapDataset,
    transforms as grain_transforms,
)
from jax.sharding import Mesh, PartitionSpec

from jaxformers.data.transforms import DatasetTransforms

from .huggingface import (
    HuggingFaceSourceIterDataset,
    HuggingFaceSourceMapDataset,
)


Batch = tp.Any
_T = tp.TypeVar("_T")
_S = tp.TypeVar("_S")


class _DatasetIteratorWithInputSpec(DatasetIterator[_T]):
    _SLEEP_SECONDS = 2.0
    _MAX_ATTEMPTS = 5

    def __init__(
        self,
        parent: DatasetIterator[_S],
        pspec: PartitionSpec,
        mesh: Mesh,
    ):
        super().__init__(parent)
        self._pspec = pspec
        self._mesh = mesh
        self._logger = logging.getLogger(__name__)

    def __next__(self) -> _T:
        attempts = 0
        last_error: Exception | None = None
        while True:
            try:
                local_values = next(self._parent)
                break
            except StopIteration:
                raise
            except Exception as err:  # pylint: disable=broad-except
                attempts += 1
                last_error = err
                if attempts >= self._MAX_ATTEMPTS:
                    break
                if self._logger.isEnabledFor(logging.WARNING):
                    self._logger.warning(
                        "Failed to fetch next batch (attempt %d/%d): %s. Retrying in %.1fs.",
                        attempts,
                        self._MAX_ATTEMPTS,
                        err,
                        self._SLEEP_SECONDS,
                    )
                time.sleep(self._SLEEP_SECONDS)

        if last_error is not None and attempts >= self._MAX_ATTEMPTS:
            raise last_error
        with self._stats.record_self_time():
            return self._stats.record_output_spec(
                jtu.tree_map(self.array_from_local_process, local_values)
            )

    def array_from_local_process(self, local_values: _T) -> _T:
        return jax.make_array_from_process_local_data(
            sharding=jax.NamedSharding(self._mesh, self._pspec),
            local_data=local_values,
        )

    def get_state(self):
        return self._parent.get_state()

    def set_state(self, state):
        self._parent.set_state(state)


class IterDatasetWithInputSpec(IterDataset[_T]):
    def __init__(
        self,
        parent: IterDataset[_S],
        pspec: PartitionSpec | None = None,
        mesh: Mesh | None = None,
    ):
        super().__init__(parent)

        self._pspec = pspec or PartitionSpec()
        self._mesh = mesh

    def __iter__(self) -> "IterDatasetWithInputSpec":
        parent_iter = self._parent.__iter__()
        return _DatasetIteratorWithInputSpec(
            parent_iter, pspec=self._pspec, mesh=self._mesh
        )


class _ShardedIterDataset(IterDataset[_T]):
    """Shard a Grain IterDataset by taking every Nth element.

    This is mainly intended for simple multi-host dataloading when a dataset does
    not provide a native `.shard(...)` method.
    """

    def __init__(self, parent: IterDataset[_T], *, num_shards: int, index: int):
        super().__init__(parent)
        if num_shards <= 0:
            raise ValueError("num_shards must be positive")
        if index < 0 or index >= num_shards:
            raise ValueError("index must be in [0, num_shards)")
        self._num_shards = num_shards
        self._index = index

    def __iter__(self) -> DatasetIterator[_T]:
        parent_iter = self._parent.__iter__()
        shard_index = self._index
        num_shards = self._num_shards

        class _Iterator(DatasetIterator[_T]):
            def __init__(self, parent_it: DatasetIterator[_T]):
                super().__init__(parent_it)
                self._counter = 0

            def __next__(self) -> _T:
                while True:
                    item = next(self._parent)
                    counter = self._counter
                    self._counter = counter + 1
                    if counter % num_shards == shard_index:
                        return item

            def get_state(self):
                return {
                    "parent_state": self._parent.get_state(),
                    "counter": self._counter,
                }

            def set_state(self, state):
                self._parent.set_state(state["parent_state"])
                self._counter = state["counter"]

        return _Iterator(parent_iter)


def make_dataloader(
    datasets: Sequence[IterDataset | MapDataset],
    transforms: Sequence[
        grain_transforms.Map | grain_transforms.RandomMap | DatasetTransforms
    ]
    | None,
    global_batch_size: int,
    pspec: PartitionSpec | None = None,
    mesh: Mesh | None = None,
    num_epochs: int | None = None,
    dataset_weights: Sequence[float] | None = None,
    dataloading_host_index: int | None = None,
    dataloading_host_count: int | None = None,
    is_not_sharded: bool = True,
    read_num_threads: int = 0,
    read_prefetch_buffer_size: int = 0,
    shuffle: bool = True,
    shuffle_buffer_size: int = 1000,
    seed: int = 0,
    worker_count: int = 0,
    worker_buffer_size: int = 0,
    drop_remainder: bool = True,
) -> IterDatasetWithInputSpec:
    if dataloading_host_index is None:
        dataloading_host_index = jax.process_index()
    if dataloading_host_count is None:
        dataloading_host_count = jax.process_count()

    transforms = tuple(transforms or ())

    if dataloading_host_count <= 0:
        raise ValueError("dataloading_host_count must be positive")
    if global_batch_size % dataloading_host_count != 0:
        raise ValueError(
            "global_batch_size must be divisible by dataloading_host_count"
        )

    prepared: list[grain.IterDataset] = []
    if isinstance(datasets, (IterableDataset, Dataset)) or not isinstance(
        datasets, Sequence
    ):
        datasets = (datasets,)
    else:
        datasets = tuple(datasets)

    read_options = grain.ReadOptions(
        num_threads=read_num_threads, prefetch_buffer_size=read_prefetch_buffer_size
    )

    for ds in datasets:
        if isinstance(ds, IterableDataset):
            ds = HuggingFaceSourceIterDataset(ds)
        elif isinstance(ds, Dataset):
            ds = HuggingFaceSourceMapDataset(ds)

        if dataloading_host_count > 1 and is_not_sharded:
            if hasattr(ds, "shard"):
                ds = ds.shard(
                    num_shards=dataloading_host_count,
                    index=dataloading_host_index,
                    contiguous=True,
                )
            elif isinstance(ds, grain.MapDataset):
                length = len(ds)
                start = (length * dataloading_host_index) // dataloading_host_count
                end = (length * (dataloading_host_index + 1)) // dataloading_host_count
                ds = ds.slice(slice(start, end))
            elif isinstance(ds, grain.IterDataset):
                ds = _ShardedIterDataset(
                    ds,
                    num_shards=dataloading_host_count,
                    index=dataloading_host_index,
                )
            else:
                raise TypeError(f"Dataset sharding unsupported for type {type(ds)}")

        if shuffle:
            if hasattr(ds, "shuffle"):
                try:
                    ds = ds.shuffle(
                        seed=seed + dataloading_host_index,
                        buffer_size=shuffle_buffer_size,
                    )
                except TypeError:
                    ds = ds.shuffle(seed=seed + dataloading_host_index)
                if isinstance(ds, HuggingFaceSourceMapDataset):
                    warnings.warn(
                        "Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. "
                        "If shuffling is important for your workflow, please pre-shuffle the dataset."
                    )
            elif isinstance(ds, grain.IterDataset):
                ds = grain.experimental.WindowShuffleIterDataset(
                    ds,
                    window_size=shuffle_buffer_size,
                    seed=seed + dataloading_host_index,
                )
            else:
                raise TypeError(
                    f"Shuffle requested but unsupported for dataset type {type(ds)}"
                )

        if num_epochs is not None:
            if hasattr(ds, "repeat"):
                ds = ds.repeat(num_epochs)
            elif isinstance(ds, grain.IterDataset):
                ds = grain.experimental.RepeatIterDataset(ds, num_epochs=num_epochs)
            else:
                raise TypeError(
                    f"Repeat requested but unsupported for dataset type {type(ds)}"
                )

        if isinstance(ds, grain.MapDataset):
            ds = ds.to_iter_dataset(read_options)
        elif not isinstance(ds, grain.IterDataset):
            raise TypeError(
                "Dataset transform pipeline must return a Grain MapDataset or IterDataset"
            )

        for op in transforms:
            # NOTES: pretty much all transformation just wrapping the dataset by another dataset class with new __iter__ and __next__ (the iterator part)
            # so by this we shouldnt differentiate between BaseDatasetTransform and grain transforms
            # need to think more about makeing single interface for all transformation
            if isinstance(op, DatasetTransforms):
                ds = op(ds)
            elif isinstance(op, grain_transforms.RandomMap):
                ds = ds.random_map(op)
            elif isinstance(op, grain_transforms.Map):
                ds = ds.map(op)
            else:
                raise TypeError(f"Unsupported operation type: {type(op)}")

        prepared.append(ds)

    mixed = grain.IterDataset.mix(prepared, weights=dataset_weights)
    local_process_batch_size = global_batch_size // dataloading_host_count

    mixed = mixed.batch(
        batch_size=local_process_batch_size, drop_remainder=drop_remainder
    )

    mp_options = grain.MultiprocessingOptions(
        num_workers=worker_count,
        per_worker_buffer_size=worker_buffer_size,
    )
    mixed = mixed.mp_prefetch(mp_options)
    if mesh:
        return IterDatasetWithInputSpec(mixed, pspec=pspec, mesh=mesh)
    return mixed
