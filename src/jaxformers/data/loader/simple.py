from __future__ import annotations

import logging
import time
import typing as tp
import warnings
from collections.abc import Sequence

import grain
import jax
from grain import DatasetIterator, IterDataset, MapDataset, ReadOptions
from grain.experimental import RepeatIterDataset, WindowShuffleIterDataset
from jax import P
from jax.sharding import Mesh, PartitionSpec

from jaxformers.data.transforms.base import transform_ds, TransformFn
from jaxformers.distributed.parallel import BATCH


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
                self.array_from_local_process(local_values)
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


class ShardedIterDataset(IterDataset[_T]):
    def __init__(
        self,
        parent: IterDataset[_S],
        pspec: PartitionSpec | None = None,
        mesh: Mesh | None = None,
    ):
        super().__init__(parent)

        self._pspec = pspec or PartitionSpec()
        self._mesh = mesh

    def __iter__(self) -> ShardedIterDataset:
        parent_iter = self._parent.__iter__()
        return _DatasetIteratorWithInputSpec(
            parent_iter, pspec=self._pspec, mesh=self._mesh
        )


def make_simple_loader(
    datasets: Sequence[MapDataset] | Sequence[IterDataset],
    transforms: Sequence[TransformFn] | None,
    mesh: Mesh | None,
    batch_size: int,
    shard: bool = False,
    shuffle: bool = False,
    dataset_weights: Sequence[float] | None = None,
    *,
    num_workers: int = 0,
    num_threads: int = 1,
    prefetch_buffer_size: int | None = 500,
    per_worker_buffer_size: int | None = 1,
    window_size: int = 1000,
    seed: int = 0,
) -> ShardedIterDataset:

    prepared: list[grain.IterDataset] = []
    if shard and not mesh:
        raise ValueError("need mesh if we shard the datasets")

    if isinstance(datasets, (MapDataset, IterDataset)) or not isinstance(
        datasets, Sequence
    ):
        datasets = (datasets,)
    else:
        datasets = tuple(datasets)

    ds_type = type(datasets[0])
    is_map = isinstance(datasets[0], MapDataset)
    if not all(isinstance(x, ds_type) for x in datasets):
        raise ValueError(
            "All datasets must have the same type, either MapDataset or IterDataset"
        )

    process_count = jax.process_count()
    process_index = jax.process_index()
    seed = seed + process_index if shard else seed
    min_num_shards = None
    for ds in datasets:
        if shard:
            if not hasattr(ds, "shard"):
                raise NotImplementedError(
                    "shard is active but ds doenst have shard method"
                )
            ds = ds.shard(process_count, process_index)
        if not is_map and num_workers > 0 and hasattr(ds, "num_shards"):
            ds_num_shards = ds.num_shards
            if min_num_shards is None or ds_num_shards < min_num_shards:
                min_num_shards = ds_num_shards

        if shuffle:
            warnings.warn(
                "Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. "
                "If shuffling is important for your workflow, please pre-shuffle the dataset."
            )
            if hasattr(ds, "shuffle"):
                ds = ds.shuffle(seed)
            else:
                ds = WindowShuffleIterDataset(ds, window_size=window_size, seed=seed)

        ds = ds.repeat() if hasattr(ds, "repeat") else RepeatIterDataset(ds)
        prepared.append(ds)

    if min_num_shards is not None and num_workers > min_num_shards:
        warnings.warn(
            "Reducing num_workers because the streaming dataset has fewer shards "
            f"than workers: num_workers={num_workers}, num_shards={min_num_shards}."
        )
        num_workers = min_num_shards

    mixed = (
        grain.MapDataset.mix(prepared, dataset_weights)
        if is_map
        else grain.IterDataset.mix(prepared, dataset_weights)
    )
    mixed = (
        mixed.to_iter_dataset(
            read_options=ReadOptions(num_threads, prefetch_buffer_size)
        )
        if is_map
        else mixed
    )
    mixed = transform_ds(mixed, *transforms)
    batch_size = batch_size // process_count if shard else batch_size
    mixed = mixed.batch(batch_size=batch_size)
    mp_options = grain.MultiprocessingOptions(num_workers, per_worker_buffer_size)
    mixed = mixed.mp_prefetch(mp_options)
    # think more about local data -> global data
    if shard:
        return ShardedIterDataset(mixed, P(BATCH), mesh)
    return mixed
