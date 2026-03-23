from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .loader import make_loader
from .loader import mix as mix_loader
from .loader import zip as zip_loader
from .source import make_source
from .transforms import make_transforms


def _group_items(data_config: Mapping[str, Any]):
    groups = []
    for name in data_config:
        if name in ("streaming", "loader"):
            continue
        groups.append((name, data_config[name]))
    if not groups:
        raise ValueError("data config must define at least one dataset group")
    return groups


def make_data(
    data_config: Mapping[str, Any],
    *,
    mesh=None,
):
    streaming = data_config["streaming"] if "streaming" in data_config else False
    loader_config = dict(data_config["loader"])
    shared_loader_config = dict(loader_config)
    shared_loader_config.pop("combine", None)

    groups = []
    for _, group_config in _group_items(data_config):
        source_name = (
            group_config["source_name"]
            if "source_name" in group_config
            else "huggingface"
        )
        source = make_source(
            source_name,
            group_config["source"],
            streaming=streaming,
        )
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
        groups.append((source, transforms, group_config["loader"]))

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


__all__ = [
    "make_data",
    "make_loader",
    "make_source",
    "make_transforms",
]
