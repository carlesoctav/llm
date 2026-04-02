from __future__ import annotations

import logging
import time
import typing as tp
from collections.abc import Sequence
from typing import Callable

import grain
import jax
from grain import DatasetIterator, IterDataset, MapDataset
from jax import P
from jax.sharding import Mesh, PartitionSpec

from jaxformers.data.loader._group import prepare_group
from jaxformers.data.transforms.base import TransformFn
from jaxformers.sharding_utils import BATCH


Batch = tp.Any
_T = tp.TypeVar("_T")
_S = tp.TypeVar("_S")


class _ProcessShardedDatasetIterator(DatasetIterator[_T]):
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


class ProcessShardedIterDataset(IterDataset[_T]):
    def __init__(
        self,
        parent: IterDataset[_S],
        pspec: PartitionSpec | None = None,
        mesh: Mesh | None = None,
    ):
        super().__init__(parent)

        self._pspec = pspec or PartitionSpec()
        self._mesh = mesh

    def __iter__(self) -> ProcessShardedIterDataset:
        parent_iter = self._parent.__iter__()
        return _ProcessShardedDatasetIterator(
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
    batch_fn: Callable | None = None,
    num_epochs: int | None = None,
) -> ProcessShardedIterDataset:
    if shard and not mesh:
        raise ValueError("need mesh if we shard the datasets")

    process_count = jax.process_count()
    process_index = jax.process_index()
    mixed = prepare_group(
        datasets,
        transforms,
        batch_size=batch_size,
        shuffle=shuffle,
        dataset_weights=dataset_weights,
        shard=shard,
        process_count=process_count,
        process_index=process_index,
        num_threads=num_threads,
        prefetch_buffer_size=prefetch_buffer_size,
        window_size=window_size,
        seed=seed,
        batch_fn=batch_fn,
        num_epochs=num_epochs,
    )
    mp_options = grain.MultiprocessingOptions(num_workers, per_worker_buffer_size)
    mixed = mixed.mp_prefetch(mp_options)
    # think more about local data -> global data
    if shard:
        return ProcessShardedIterDataset(mixed, P(BATCH), mesh)
    return mixed


def make(
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
    batch_fn: Callable | None = None,
    num_epochs: int | None = None,
):
    return make_simple_loader(
        datasets,
        transforms,
        mesh,
        batch_size,
        shard,
        shuffle,
        dataset_weights,
        num_workers=num_workers,
        num_threads=num_threads,
        prefetch_buffer_size=prefetch_buffer_size,
        per_worker_buffer_size=per_worker_buffer_size,
        window_size=window_size,
        seed=seed,
        batch_fn=batch_fn,
        num_epochs=num_epochs,
    )
