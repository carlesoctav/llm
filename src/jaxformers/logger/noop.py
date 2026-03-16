from collections.abc import Sequence
from typing import Any, Literal


class NoopLoggerConfig:
    def update(self, values: dict[str, Any]) -> None:
        _ = values


class NoopLogger:
    def __init__(self):
        self.config = NoopLoggerConfig()

    def log(self, values: dict[str, Any], *, step: int) -> None:
        _ = (values, step)

    def finish(self) -> None:
        return


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
    settings: object | dict[str, Any] | None = None,
    anonymous: object = _ANONYMOUS_UNSET,
):
    _ = (
        entity,
        project,
        dir,
        id,
        name,
        notes,
        tags,
        config,
        config_exclude_keys,
        config_include_keys,
        allow_val_change,
        group,
        job_type,
        mode,
        force,
        reinit,
        resume,
        resume_from,
        fork_from,
        save_code,
        tensorboard,
        sync_tensorboard,
        monitor_gym,
        settings,
        anonymous,
    )
    return NoopLogger()
