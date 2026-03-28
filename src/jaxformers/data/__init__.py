from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from grain.experimental import batch_and_pad

from .loader import make_loader, mix as mix_loader, zip as zip_loader
from .source import make_source
from .transforms import make_transforms


def _group_items(data_config: Mapping[str, Any]):
    groups = []
    for name in data_config:
        if name == "loader":
            continue
        groups.append((name, data_config[name]))
    if not groups:
        raise ValueError("data config must define at least one dataset group")
    return groups


def _make_group(group_config: Mapping[str, Any]):
    source_name = (
        group_config["source_name"] if "source_name" in group_config else "huggingface"
    )
    source = make_source(source_name, group_config["source"])
    transforms_config = group_config["transforms"]
    if isinstance(transforms_config, Sequence) and not isinstance(
        transforms_config, (dict, str, bytes)
    ):
        transforms = transforms_config
    else:
        transforms = make_transforms(
            group_config["transforms_name"],
            transforms_config,
        )
    return source, transforms, group_config["loader"]


def make_data(
    data_config: Mapping[str, Any],
    *,
    mesh=None,
):
    loader_config = dict(data_config["loader"])
    shared_loader_config = dict(loader_config)
    shared_loader_config.pop("combine", None)

    groups = []
    for _, group_config in _group_items(data_config):
        groups.append(_make_group(group_config))

    if len(groups) == 1:
        source, transforms, group_loader_config = groups[0]
        merged_loader_config = dict(shared_loader_config)
        merged_loader_config.update(group_loader_config)
        return make_loader(
            "simple",
            source,
            transforms,
            merged_loader_config,
            mesh=mesh,
        )

    combine = loader_config["combine"]
    if combine not in ("zip", "mix"):
        raise ValueError(f"Unsupported data.loader.combine {combine!r}")

    if combine == "zip":
        return zip_loader.make(groups, mesh, **shared_loader_config)
    if combine == "mix":
        return mix_loader.make(groups, mesh, **shared_loader_config)


def make_eval_data(
    data_config: Mapping[str, Any],
    *,
    mesh=None,
):
    source, transforms, loader_config = _make_group(data_config)
    eval_loader_config = dict(loader_config)
    num_workers = (
        eval_loader_config["num_workers"] if "num_workers" in eval_loader_config else 0
    )
    if num_workers > 0:
        warnings.warn("make_eval_data works best with loader.num_workers set to 0.")
    shard = eval_loader_config["shard"] if "shard" in eval_loader_config else False
    if shard:
        warnings.warn("make_eval_data works best with loader.shard set to False.")

    def batch_fn(values):
        batched = batch_and_pad(values, batch_size=loader_config["batch_size"])
        if isinstance(batched, dict):
            valid_size = len(values)
            batched["_mask"] = np.concatenate(
                (
                    np.ones((valid_size,), dtype=np.int32),
                    np.zeros(
                        (loader_config["batch_size"] - valid_size,), dtype=np.int32
                    ),
                )
            )
        return batched

    eval_loader_config["batch_fn"] = batch_fn
    eval_loader_config["num_epochs"] = 1
    return make_loader(
        "simple",
        source,
        transforms,
        eval_loader_config,
        mesh=mesh,
    )


__all__ = [
    "make_data",
    "make_eval_data",
    "make_loader",
    "make_source",
    "make_transforms",
]
