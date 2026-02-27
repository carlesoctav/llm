import os
import runpy
from collections.abc import Iterable, Mapping
from itertools import product
from typing import Any

import sws


class ConfigLoadError(RuntimeError):
    pass


class SweepConfigError(ValueError):
    pass


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


def build_sweep_space(
    *,
    overrides: Mapping[str, Any],
    group: list[Mapping[str, Any]] | None,
) -> list[tuple[int | None, dict[str, Any]]]:
    dims: dict[str, list[Any]] = {}
    for raw_key, raw_value in overrides.items():
        key = str(raw_key)
        value = raw_value
        if isinstance(value, (list, tuple)):
            dims[key] = list(value)
        else:
            dims[key] = [value]

    groups: list[dict[str, Any]] = []
    group_keys: set[str] = set()
    for raw_item in list(group or []):
        item = raw_item
        if not isinstance(item, Mapping):
            raise SweepConfigError(
                "Each `group` item must be a dict. "
                "Tip: use flat dotted keys like {'optimizer.b1': 0.9}."
            )
        parsed: dict[str, Any] = {}
        for raw_key, raw_value in dict(item).items():
            key = str(raw_key)
            parsed[key] = raw_value
            group_keys.add(key)
        groups.append(parsed)

    overlap = sorted(set(dims).intersection(group_keys))
    if overlap:
        raise SweepConfigError(
            "Keys cannot appear in both free sweep params and group: "
            + ", ".join(overlap)
        )

    runs: list[tuple[int | None, dict[str, Any]]] = []
    dims_for_product: list[list[tuple[str, Any]]] = []
    for key, values in dims.items():
        dims_for_product.append([(key, value) for value in values])

    group_choices = groups if groups else [{}]
    for group_idx, group_item in enumerate(group_choices):
        combos: Iterable[tuple[tuple[str, Any], ...]]
        combos = product(*dims_for_product) if dims_for_product else [()]
        for combo in combos:
            run: dict[str, Any] = dict(group_item)
            for key, value in combo:
                run[key] = value
            runs.append((group_idx if groups else None, run))

    return runs
