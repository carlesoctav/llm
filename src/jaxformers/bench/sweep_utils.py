import os
import runpy
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any

import sws
from sws.simpleeval import EvalWithCompoundTypes


class ConfigLoadError(RuntimeError):
    pass


class SweepConfigError(ValueError):
    pass


def _range_list(*args):
    if len(args) == 1:
        start = 0
        stop = args[0]
        step = 1
    elif len(args) == 2:
        start, stop = args
        step = 1
    elif len(args) == 3:
        start, stop, step = args
    else:
        raise TypeError(f"range expected 1-3 arguments, got {len(args)}")

    if all(isinstance(x, int) for x in (start, stop, step)):
        return list(range(start, stop, step))

    start = float(start)
    stop = float(stop)
    step = float(step)
    if step == 0:
        raise ValueError("range() arg 3 must not be zero")

    out: list[float] = []
    cur = start
    max_steps = 100_000
    for _ in range(max_steps):
        if step > 0 and cur >= stop:
            break
        if step < 0 and cur <= stop:
            break
        out.append(cur)
        cur += step
    else:
        raise ValueError("range() produced too many elements")

    return out


_EVAL = EvalWithCompoundTypes(functions={"Fn": sws.Fn, "range": _range_list})


def parse_maybe_expr(value: Any) -> Any:
    if isinstance(value, range):
        return list(value)
    if not isinstance(value, str):
        return value
    expr = value.strip()
    if not expr:
        return value
    try:
        return _EVAL.eval(expr)
    except Exception:
        return value


def load_config_builder(config_path: str, *, default_func: str = "get_config") -> sws.Config:
    path = config_path
    func_name = default_func
    if ":" in path:
        path, func_name = path.split(":", 1)

    path = os.path.abspath(path)
    factory = runpy.run_path(path).get(func_name)
    if not callable(factory):
        raise ConfigLoadError(f"Function {func_name!r} not found in {path}")
    builder = factory()
    if not isinstance(builder, sws.Config):
        raise ConfigLoadError(f"Config factory in {path} must return a sws.Config")
    return builder


def normalize_key(key: str) -> str:
    return str(key).removeprefix("c.")


def flatten_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def rec(prefix: str, value: Any) -> None:
        value = parse_maybe_expr(value)
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                next_key = f"{prefix}.{child_key}" if prefix else str(child_key)
                rec(next_key, child_value)
            return
        out[prefix] = value

    for key, value in overrides.items():
        rec(str(key), value)
    return out


def format_cli_value(value: Any) -> str:
    return repr(value)


@dataclass(frozen=True)
class SweepSpace:
    fixed: dict[str, Any]
    dimensions: dict[str, list[Any]]
    groups: list[dict[str, Any]]

    @property
    def run_count(self) -> int:
        runs = len(self.groups) if self.groups else 1
        for values in self.dimensions.values():
            runs *= len(values)
        return runs

    def iter_runs(self) -> Iterable[tuple[int | None, dict[str, Any]]]:
        dims: list[list[tuple[str, Any]]] = []
        for key, values in self.dimensions.items():
            dims.append([(key, value) for value in values])

        group_choices = self.groups if self.groups else [{}]
        for group_idx, group_item in enumerate(group_choices):
            for combo in product(*dims) if dims else [()]:
                run: dict[str, Any] = dict(self.fixed)
                run.update(group_item)
                for key, value in combo:
                    run[key] = value
                yield (group_idx if self.groups else None), run


def build_sweep_space(
    *,
    base_store: Mapping[str, Any],
    overrides: Mapping[str, Any],
    group: list[Mapping[str, Any]] | None,
) -> SweepSpace:
    known = set(str(key) for key in base_store)
    base_groups: set[str] = set()
    for full_key in known:
        parts = str(full_key).split(".")
        for idx in range(1, len(parts)):
            base_groups.add(".".join(parts[:idx]))

    def _validate_key(raw_key: str) -> str:
        key = normalize_key(raw_key)
        if key in known:
            return key
        if "." in key:
            parent = key.rsplit(".", 1)[0]
            if parent in base_groups:
                return key
        raise SweepConfigError(
            f"Unknown sweep key {raw_key!r}. "
            "Use exact keys, or add a new leaf under an existing parent (with ':=')."
        )

    fixed: dict[str, Any] = {}
    dims: dict[str, list[Any]] = {}
    for raw_key, raw_value in overrides.items():
        key = _validate_key(raw_key)
        value = parse_maybe_expr(raw_value)
        if isinstance(value, (list, tuple)):
            base_value = base_store.get(key)
            base_is_list = isinstance(base_value, (list, tuple))
            if base_is_list:
                as_list = list(value)
                is_list_of_lists = all(
                    isinstance(candidate, (list, tuple)) for candidate in as_list
                )
                if is_list_of_lists:
                    dims[key] = as_list
                else:
                    fixed[key] = value
            else:
                dims[key] = list(value)
        else:
            fixed[key] = value

    groups: list[dict[str, Any]] = []
    group_keys: set[str] = set()
    for raw_item in list(group or []):
        item = parse_maybe_expr(raw_item)
        if not isinstance(item, Mapping):
            raise SweepConfigError(
                "Each `group` item must be a dict. "
                "Tip: use flat dotted keys like {'optimizer.b1': 0.9}."
            )
        flat_item = flatten_overrides(dict(item))
        parsed: dict[str, Any] = {}
        for raw_key, raw_value in flat_item.items():
            key = _validate_key(raw_key)
            parsed[key] = raw_value
            group_keys.add(key)
        groups.append(parsed)

    overlap = sorted(set(dims).intersection(group_keys))
    if overlap:
        raise SweepConfigError(
            "Keys cannot appear in both free sweep params and group: "
            + ", ".join(overlap)
        )

    return SweepSpace(fixed=fixed, dimensions=dims, groups=groups)
