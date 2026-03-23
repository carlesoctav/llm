from __future__ import annotations

import typing as tp
from collections.abc import Sequence

import grain
import jax
from grain import IterDataset, MapDataset
from jax import P
from jax.sharding import Mesh

from jaxformers.data.loader._group import prepare_group
from jaxformers.data.loader.simple import ProcessShardedIterDataset
from jaxformers.data.transforms.base import TransformFn
from jaxformers.distributed.parallel import BATCH


_T = tp.TypeVar("_T")


def make(
    groups: Sequence[
        tuple[
            Sequence[MapDataset] | Sequence[IterDataset],
            Sequence[TransformFn] | None,
            dict[str, tp.Any],
        ]
    ],
    mesh: Mesh | None,
    *,
    shard: bool = False,
    num_workers: int = 0,
    num_threads: int = 1,
    prefetch_buffer_size: int | None = 500,
    per_worker_buffer_size: int | None = 1,
    window_size: int = 1000,
    seed: int = 0,
) -> ProcessShardedIterDataset | IterDataset[_T]:
    if not groups:
        raise ValueError("mix loader requires at least one dataset group")

    if shard and not mesh:
        raise ValueError("need mesh if we shard the datasets")

    process_count = jax.process_count()
    process_index = jax.process_index()

    prepared_groups = []
    for group_datasets, group_transforms, group_loader in groups:
        prepared_group = prepare_group(
            group_datasets,
            group_transforms,
            shard=shard,
            **group_loader,
            process_count=process_count,
            process_index=process_index,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size,
            window_size=window_size,
            seed=seed,
        )
        prepared_groups.append(prepared_group)

    mixed = grain.IterDataset.mix(prepared_groups)
    mp_options = grain.MultiprocessingOptions(num_workers, per_worker_buffer_size)
    mixed = mixed.mp_prefetch(mp_options)
    if shard:
        return ProcessShardedIterDataset(mixed, P(BATCH), mesh)
    return mixed
