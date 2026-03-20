from __future__ import annotations

import typing as tp
import warnings
from collections.abc import Sequence

import grain
import jax
from grain import IterDataset, MapDataset, ReadOptions
from grain._src.python.dataset.transformations.zip import ZipIterDataset
from grain.experimental import RepeatIterDataset, WindowShuffleIterDataset
from jax import P
from jax.sharding import Mesh

from jaxformers.data.loader.simple import ProcessShardedIterDataset
from jaxformers.data.transforms.base import transform_ds, TransformFn
from jaxformers.distributed.parallel import BATCH


_T = tp.TypeVar("_T")


def _prepare_group(
    datasets: Sequence[MapDataset] | Sequence[IterDataset],
    transforms: Sequence[TransformFn] | None,
    *,
    process_count: int,
    process_index: int,
    shard: bool,
    shuffle: bool,
    seed: int,
    window_size: int,
    num_threads: int,
    prefetch_buffer_size: int | None,
):
    if isinstance(datasets, (MapDataset, IterDataset)) or not isinstance(
        datasets, Sequence
    ):
        datasets = (datasets,)
    else:
        datasets = tuple(datasets)

    if not datasets:
        raise ValueError("Each zipped dataset group must contain at least one dataset.")

    ds_type = type(datasets[0])
    is_map = isinstance(datasets[0], MapDataset)
    if not all(isinstance(x, ds_type) for x in datasets):
        raise ValueError(
            "All datasets within a zipped group must have the same type, either MapDataset or IterDataset."
        )

    prepared = []
    group_seed = seed + process_index if shard else seed
    for ds in datasets:
        if shard:
            if not hasattr(ds, "shard") and not hasattr(ds, "num_shards"):
                raise NotImplementedError(
                    "shard is active but ds doenst have shard method and num_shards attribute"
                )
            if ds.num_shards < process_count:
                raise ValueError(
                    f"Number of dataset shards (or MapDataset rows) ({ds.num_shards}) is less than the number of processes ({process_count}). "
                    "Some processes will not receive any data."
                )
            ds = ds.shard(process_count, process_index)

        if shuffle:
            warnings.warn(
                "Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. "
                "If shuffling is important for your workflow, please pre-shuffle the dataset."
            )
            if hasattr(ds, "shuffle"):
                ds = ds.shuffle(group_seed)
            else:
                ds = WindowShuffleIterDataset(
                    ds,
                    window_size=window_size,
                    seed=group_seed,
                )

        ds = ds.repeat() if hasattr(ds, "repeat") else RepeatIterDataset(ds)
        prepared.append(ds)

    mixed = (
        grain.MapDataset.mix(prepared) if is_map else grain.IterDataset.mix(prepared)
    )
    mixed = (
        mixed.to_iter_dataset(
            read_options=ReadOptions(num_threads, prefetch_buffer_size)
        )
        if is_map
        else mixed
    )
    return transform_ds(mixed, *(transforms or ())), is_map


def make(
    datasets: Sequence[Sequence[MapDataset] | Sequence[IterDataset]],
    transforms: Sequence[Sequence[TransformFn] | None],
    mesh: Mesh | None,
    batch_size: int,
    shard: bool = False,
    shuffle: bool = False,
    *,
    num_workers: int = 0,
    num_threads: int = 1,
    prefetch_buffer_size: int | None = 500,
    per_worker_buffer_size: int | None = 1,
    window_size: int = 1000,
    seed: int = 0,
) -> ProcessShardedIterDataset | IterDataset[_T]:
    if len(datasets) != len(transforms):
        raise ValueError("datasets and transforms must have the same number of groups")
    if shard and not mesh:
        raise ValueError("need mesh if we shard the datasets")

    process_count = jax.process_count()
    process_index = jax.process_index()

    prepared_groups = []
    group_types = []
    for group_datasets, group_transforms in zip(datasets, transforms, strict=True):
        prepared_group, is_map = _prepare_group(
            group_datasets,
            group_transforms,
            process_count=process_count,
            process_index=process_index,
            shard=shard,
            shuffle=shuffle,
            seed=seed,
            window_size=window_size,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size,
        )
        prepared_groups.append(prepared_group)
        group_types.append(is_map)

    if len(set(group_types)) > 1:
        raise ValueError(
            "All zipped dataset groups must have the same type, either MapDataset or IterDataset."
        )

    zipped = ZipIterDataset(prepared_groups)
    batch_size = batch_size // process_count if shard else batch_size
    zipped = zipped.batch(batch_size=batch_size)
    mp_options = grain.MultiprocessingOptions(num_workers, per_worker_buffer_size)
    zipped = zipped.mp_prefetch(mp_options)
    if shard:
        return ProcessShardedIterDataset(zipped, P(BATCH), mesh)
    return zipped
