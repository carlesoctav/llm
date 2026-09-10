from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Callable

import grain
from grain import IterDataset, MapDataset, ReadOptions
from grain.experimental import RepeatIterDataset, WindowShuffleIterDataset

from jaxformers.data.transforms.base import transform_ds, TransformFn


def prepare_group(
    datasets: Sequence[MapDataset] | Sequence[IterDataset],
    transforms: Sequence[TransformFn] | None,
    *,
    batch_size: int,
    shuffle: bool = False,
    dataset_weights: Sequence[float] | None = None,
    shard: bool = False,
    process_count: int,
    process_index: int,
    num_threads: int,
    prefetch_buffer_size: int | None,
    window_size: int,
    seed: int,
    batch_fn: Callable | None = None,
    num_epochs: int | None = None,
):
    if isinstance(datasets, (MapDataset, IterDataset)) or not isinstance(
        datasets, Sequence
    ):
        datasets = (datasets,)
    else:
        datasets = tuple(datasets)

    if not datasets:
        raise ValueError("Each dataset group must contain at least one dataset.")

    ds_type = type(datasets[0])
    is_map = isinstance(datasets[0], MapDataset)
    if not all(isinstance(x, ds_type) for x in datasets):
        raise ValueError(
            "All datasets within a group must have the same type, either MapDataset or IterDataset."
        )

    group_seed = seed + process_index if shard else seed

    prepared = []
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

        ds = (
            ds.repeat(num_epochs=num_epochs)
            if hasattr(ds, "repeat")
            else RepeatIterDataset(ds, num_epochs=num_epochs)
        )
        prepared.append(ds)

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

    mixed = transform_ds(mixed, *(transforms or ()))
    batch_size = batch_size // process_count if shard else batch_size
    return mixed.batch(batch_size=batch_size, batch_fn=batch_fn)
