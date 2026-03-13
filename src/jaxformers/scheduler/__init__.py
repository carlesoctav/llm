from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any


def make_scheduler(
    scheduler_name: str | None,
    learning_rate: float,
    num_train_steps: int,
    scheduler_config: Mapping[str, Any] | None = None,
):
    if scheduler_name in (None, "constant"):
        return learning_rate

    scheduler_module = importlib.import_module(f"jaxformers.scheduler.{scheduler_name}")
    make = getattr(scheduler_module, "make", None)
    if not callable(make):
        raise ValueError(
            f"{scheduler_module!r} does not have a callable 'make' function; "
            "please ensure you're using the correct lr_scheduler_name."
        )

    return make(
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        **dict(scheduler_config or {}),
    )
