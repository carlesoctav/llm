import os
import runpy
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
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


def format_cli_value(value: Any) -> str:
    return repr(value)


@dataclass(frozen=True)
class SweepSpace:
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
                run: dict[str, Any] = dict(group_item)
                for key, value in combo:
                    run[key] = value
                yield (group_idx if self.groups else None), run


def build_sweep_space(
    *,
    overrides: Mapping[str, Any],
    group: list[Mapping[str, Any]] | None,
) -> SweepSpace:
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

    return SweepSpace(dimensions=dims, groups=groups)
