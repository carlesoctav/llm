from __future__ import annotations

import importlib
import shutil
from collections.abc import Mapping
from typing import Any

import numpy as np


def _should_print_scheduler_preview() -> bool:
    try:
        import jax
    except Exception:
        return True

    return jax.process_index() == 0


def _sample_schedule(
    schedule: float | Any,
    num_train_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    max_step = max(num_train_steps - 1, 0)
    terminal_width = shutil.get_terminal_size(fallback=(100, 30)).columns
    sample_count = min(max(terminal_width - 16, 24), 120, max(max_step + 1, 2))
    steps = np.linspace(0, max_step, num=sample_count, dtype=int)
    steps = np.unique(steps)
    values = np.asarray(
        [
            float(np.asarray(schedule(int(step)))) if callable(schedule) else float(schedule)
            for step in steps
        ],
        dtype=np.float64,
    )
    return steps, values


def _render_termplotlib(steps: np.ndarray, values: np.ndarray) -> str:
    import termplotlib as tpl

    terminal_size = shutil.get_terminal_size(fallback=(100, 30))
    plot_width = min(max(terminal_size.columns - 24, 30), 120)
    plot_height = min(max(terminal_size.lines - 18, 6), 16)
    fig = tpl.figure()
    fig.plot(
        steps.tolist(),
        values.tolist(),
        width=plot_width,
        height=plot_height,
    )
    return fig.get_string().rstrip()


def _print_scheduler_preview(
    scheduler_name: str,
    schedule: float | Any,
    *,
    learning_rate: float,
    num_train_steps: int,
) -> None:
    if not _should_print_scheduler_preview():
        return

    try:
        from rich.console import Console, Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except Exception as exc:
        print(f"Skipping scheduler preview; rich/termplotlib unavailable: {exc}")
        return

    try:
        steps, values = _sample_schedule(schedule, num_train_steps)
        plot = _render_termplotlib(steps, values)
    except Exception as exc:
        print(f"Skipping scheduler preview; rich/termplotlib unavailable: {exc}")
        return

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="cyan")
    summary.add_column()
    summary.add_row("scheduler", scheduler_name)
    summary.add_row("train steps", f"{num_train_steps:,}")
    summary.add_row("sample points", f"{len(steps):,}")
    summary.add_row("base lr", f"{learning_rate:.6g}")
    summary.add_row("start lr", f"{values[0]:.6g}")
    summary.add_row("peak lr", f"{values.max():.6g}")
    summary.add_row("final lr", f"{values[-1]:.6g}")

    axis = Table.grid(expand=True)
    axis.add_column()
    axis.add_column(justify="right")
    axis.add_row(f"step {steps[0]}", f"step {steps[-1]}")

    body = Group(summary, Text(plot, no_wrap=True), axis)
    Console().print(
        Panel.fit(body, title="Learning Rate Schedule", border_style="cyan")
    )


def make_scheduler(
    scheduler_name: str | None,
    learning_rate: float,
    num_train_steps: int,
    scheduler_config: Mapping[str, Any] | None = None,
):
    scheduler_label = scheduler_name or "constant"
    if scheduler_name in (None, "constant"):
        schedule = learning_rate
    else:
        scheduler_module = importlib.import_module(
            f"jaxformers.scheduler.{scheduler_name}"
        )
        make = getattr(scheduler_module, "make", None)
        if not callable(make):
            raise ValueError(
                f"{scheduler_module!r} does not have a callable 'make' function; "
                "please ensure you're using the correct lr_scheduler_name."
            )

        schedule = make(
            num_train_steps=num_train_steps,
            learning_rate=learning_rate,
            **dict(scheduler_config or {}),
        )

    _print_scheduler_preview(
        scheduler_label,
        schedule,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
    )
    return schedule
