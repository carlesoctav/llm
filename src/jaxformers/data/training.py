import logging
import time
import typing as tp
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

    for dataset in datasets:
        if not transforms:
            raise ValueError("No operations provided for dataset preparation")

        if isinstance(dataset, Dataset):
            ds = HuggingFaceSourceMapDataset(dataset)
        else:
            ds = HuggingFaceSourceIterDataset(dataset)

        if dataloading_host_count > 1 and is_not_sharded:
            ds = ds.shard(
                num_shards=dataloading_host_count,
                index=dataloading_host_index,
                contiguous=True,
            )

        if shuffle:
            if isinstance(ds, grain.MapDataset):
                ds = ds.shuffle(seed=seed + dataloading_host_index)
            elif isinstance(ds, HuggingFaceSourceIterDataset):
                ds = ds.shuffle(
                    seed=seed + dataloading_host_index,
                    buffer_size=shuffle_buffer_size,
                )
            else:
                raise TypeError(
                    f"Shuffle requested but unsupported for dataset type {type(ds)}"
                )

        if num_epochs is not None:
            if isinstance(ds, HuggingFaceSourceMapDataset):
                ds = ds.repeat(num_epochs)
            elif isinstance(ds, HuggingFaceSourceIterDataset):
                ds = ds.repeat(num_epochs)
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
