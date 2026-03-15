from collections.abc import Sequence
from typing import Any, Literal

import wandb


_ANONYMOUS_UNSET = object()


def make(
    entity: str | None = None,
    project: str | None = None,
    dir: str | None = None,
    id: str | None = None,
    name: str | None = None,
    notes: str | None = None,
    tags: Sequence[str] | None = None,
    config: dict[str, Any] | str | None = None,
    config_exclude_keys: list[str] | None = None,
    config_include_keys: list[str] | None = None,
    allow_val_change: bool | None = None,
    group: str | None = None,
    job_type: str | None = None,
    mode: Literal["online", "offline", "disabled", "shared"] | None = None,
    force: bool | None = None,
    reinit: bool | str | None = None,
    resume: bool | str | None = None,
    resume_from: str | None = None,
    fork_from: str | None = None,
    save_code: bool | None = None,
    tensorboard: bool | None = None,
    sync_tensorboard: bool | None = None,
    monitor_gym: bool | None = None,
    settings: wandb.Settings | dict[str, Any] | None = None,
    anonymous: object = _ANONYMOUS_UNSET,
):
    init_kwargs = {
        "entity": entity,
        "project": project,
        "dir": dir,
        "id": id,
        "name": name,
        "notes": notes,
        "tags": tags,
        "config": config,
        "config_exclude_keys": config_exclude_keys,
        "config_include_keys": config_include_keys,
        "allow_val_change": allow_val_change,
        "group": group,
        "job_type": job_type,
        "mode": mode,
        "force": force,
        "reinit": reinit,
        "resume": resume,
        "resume_from": resume_from,
        "fork_from": fork_from,
        "save_code": save_code,
        "tensorboard": tensorboard,
        "sync_tensorboard": sync_tensorboard,
        "monitor_gym": monitor_gym,
        "settings": settings,
    }
    if anonymous is not _ANONYMOUS_UNSET:
        init_kwargs["anonymous"] = anonymous
    return wandb.init(**init_kwargs)
