from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
import numpy as np
import optax


DecaySchedule = Literal["constant", "cosine", "linear", "inv_sqrt", "inv"]


def make(
    *,
    num_train_steps: int,
    learning_rate: float,
    min_lr_ratio: float = 0.1,
    warmup: int | float = 0.01,
    decay: int | float | None = None,
    rewarmup: int | float = 0.0,
    cycle_length: int | float | list[int] | None = None,
    cycles: int | list[int] | None = None,
    decay_schedule: DecaySchedule = "cosine",
) -> Callable[[int], float]:
    """Build a warmup-stable-decay schedule with optional multiple cycles.

    The schedule is split into one or more cycles. Each cycle has:

    1. warmup or rewarmup
    2. an optional stable plateau at `learning_rate`
    3. an optional decay down to `learning_rate * min_lr_ratio`

    The first cycle uses `warmup`; later cycles use `rewarmup`.

    Args:
        num_train_steps: Total number of optimizer steps for the run.
        learning_rate: Peak learning rate reached after warmup.
        min_lr_ratio: Final learning rate floor as a fraction of `learning_rate`.
        warmup: Warmup length for the first cycle, expressed either as a fraction
            of the cycle or as an integer step count.
        decay: Decay length inside each cycle, expressed either as a fraction of
            the cycle or as an integer step count. If `None`, decay consumes all
            non-warmup steps in the cycle.
        rewarmup: Warmup length for cycles after the first.
        cycle_length: Optional cycle length specification. If a float in `[0, 1]`,
            it is interpreted as a fraction of `num_train_steps`. If an integer,
            it is interpreted as a fixed cycle length in steps. If a list, each
            entry is treated as an explicit cycle length in steps.
        cycles: Optional cycle specification as either a number of evenly sized
            cycles or a list of cycle end points. Mutually exclusive with
            `cycle_length`.
        decay_schedule: Decay shape used inside the decay portion of each cycle.
            Supported values are `"constant"`, `"cosine"`, `"linear"`,
            `"inv_sqrt"`, and `"inv"`.

    Returns:
        An Optax-compatible schedule function mapping step -> learning rate.
    """
    if num_train_steps <= 0:
        raise ValueError(f"num_train_steps must be positive, got {num_train_steps}")

    cycle_points = _get_cycle_minima(
        num_train_steps,
        cycle_length=cycle_length,
        cycles=cycles,
    )
    min_lr = learning_rate * min_lr_ratio

    schedules = []
    boundaries = []
    previous_end = 0.0
    schedule = optax.constant_schedule(learning_rate)
    lr_decay_steps = 0

    for cycle, (start, end) in enumerate(zip(cycle_points[:-1], cycle_points[1:])):
        cycle_steps = end - start
        if cycle == 0:
            warmup_steps = _convert_frac_or_steps(warmup, cycle_steps)
        else:
            warmup_steps = _convert_frac_or_steps(rewarmup, cycle_steps)
        warmup_steps = min(warmup_steps, cycle_steps)

        if warmup_steps != 0:
            warmup_schedule = optax.linear_schedule(
                previous_end,
                learning_rate,
                warmup_steps,
            )
            schedules.append(warmup_schedule)
            boundaries.append(start + warmup_steps)

        max_decay_steps = max(cycle_steps - warmup_steps, 0)
        requested_decay_steps = (
            _convert_frac_or_steps(decay, cycle_steps)
            if decay is not None
            else max_decay_steps
        )
        lr_decay_steps = min(max(requested_decay_steps, 0), max_decay_steps)
        stable_steps = cycle_steps - warmup_steps - lr_decay_steps

        if stable_steps > 0:
            stable_schedule = optax.constant_schedule(learning_rate)
            schedules.append(stable_schedule)
            boundaries.append(start + warmup_steps + stable_steps)

        if lr_decay_steps > 0:
            schedule = _make_decay_schedule(
                schedule_name=decay_schedule,
                learning_rate=learning_rate,
                min_lr=min_lr,
                decay_steps=lr_decay_steps,
                warmup_steps=warmup_steps,
                min_lr_ratio=min_lr_ratio,
            )
        else:
            schedule = optax.constant_schedule(learning_rate)

        previous_end = schedule(lr_decay_steps)
        schedules.append(schedule)
        boundaries.append(end)

    if len(schedules) > 1:
        return optax.join_schedules(schedules, boundaries)
    return schedules[0]


def _make_decay_schedule(
    *,
    schedule_name: DecaySchedule,
    learning_rate: float,
    min_lr: float,
    decay_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
):
    match schedule_name:
        case "constant":
            return optax.constant_schedule(learning_rate)
        case "cosine":
            return optax.cosine_decay_schedule(
                learning_rate,
                decay_steps,
                min_lr_ratio,
            )
        case "linear":
            return optax.linear_schedule(learning_rate, min_lr, decay_steps)
        case "inv_sqrt":
            return _inv_sqrt_decay_schedule(
                learning_rate,
                min_lr,
                warmup_steps,
                10000,
            )
        case "inv":
            return _inv_decay_schedule(learning_rate, min_lr, decay_steps)
        case _:
            raise ValueError(f"Unknown lr_schedule: {schedule_name}")


def _get_cycle_minima(
    total_main_steps: int,
    *,
    cycle_length: int | float | list[int] | None,
    cycles: int | list[int] | None,
) -> list[int]:
    """Return cycle boundaries, always including `0` and `total_main_steps`."""
    if cycle_length is not None:
        if cycles is not None:
            raise ValueError("Can't use both cycle_length and cycles.")

        if isinstance(cycle_length, (int, float)):
            cycle_length_steps = _convert_frac_or_steps(cycle_length, total_main_steps)
            if cycle_length_steps <= 0:
                raise ValueError("cycle_length must resolve to at least one step.")
            cycle_points = [
                i * cycle_length_steps
                for i in range(1, total_main_steps // cycle_length_steps)
            ]
        elif isinstance(cycle_length, list):
            lengths = np.array(cycle_length)
            steps = np.cumsum(lengths)
            if steps[-1] > total_main_steps:
                raise ValueError(
                    f"Cycle lengths exceed total number of steps: {steps[-1]} > {total_main_steps}"
                )
            cycle_points = steps.tolist()
        else:
            raise ValueError(
                "Invalid cycle_length. Must be a fraction, number of steps, or a list of steps."
            )
    elif isinstance(cycles, int):
        if cycles <= 0:
            raise ValueError("cycles must be positive.")
        cycle_points = [
            int(total_main_steps / cycles * (i + 1))
            for i in range(cycles - 1)
        ]
    elif isinstance(cycles, list):
        cycle_points = list(cycles)
    else:
        cycle_points = []

    cycle_points.insert(0, 0)
    if cycle_points[-1] != total_main_steps:
        cycle_points.append(total_main_steps)
    return cycle_points


def _inv_sqrt_decay_schedule(
    lr: float,
    min_lr: float,
    warmup_steps: int,
    timescale: float = 10000,
):
    def schedule(count):
        decay = jnp.minimum(
            1.0,
            1.0 / jnp.sqrt(jnp.maximum(count + warmup_steps, 1) / timescale),
        )
        return jnp.maximum(lr * decay, min_lr)

    return schedule


def _inv_decay_schedule(lr: float, min_lr: float, decay_steps: int):
    def schedule(count):
        decay = jnp.minimum(
            1.0,
            1.0
            / ((lr / min_lr - 1) * jnp.maximum(count, 1) / decay_steps + 1),
        )
        return jnp.maximum(lr * decay, min_lr)

    return schedule


def _convert_frac_or_steps(frac_or_steps: float | int, num_train_steps: int) -> int:
    """Interpret values <= 1 as fractions and larger integers as explicit steps."""
    if frac_or_steps < 0.0 or (
        frac_or_steps > 1.0 and frac_or_steps % 1 != 0
    ):
        raise ValueError(
            f"Invalid fraction {frac_or_steps}. Must be between 0 and 1. "
            "You can also use whole steps."
        )
    if frac_or_steps <= 1.0:
        return int(frac_or_steps * num_train_steps)
    return int(frac_or_steps)
