"""Generate per-run config files from one base config.

Usage:
  python src/jaxformers/bench/make_sweep.py \
    base_config=src/jaxformers/config/config_qwen_0_6b_loss_bench.py \
    sweep_path=/mnt/carles/llm/.agents/sweeps \
    sweep_name=loss_impl \
    'loss_implementation:=[\"xla_chunked\",\"reference\"]' \
    'data.transforms.max_length:=[512,1024]' \
    'group:=[{\"optimizer_name\":\"adam\"},{\"optimizer_name\":\"sgd\"}]'

This writes:
  <sweep_path>/<sweep_name>/0.py
  <sweep_path>/<sweep_name>/1.py
  ...
Each generated file defines `get_config()` that loads `base_config` and applies
the run overrides.
"""

import json
import os
from typing import Any

from etils import epath
import sws

from jaxformers.bench.sweep_utils import SweepConfigError, build_sweep_space


_META_KEYS = {
    "base_config",
    "sweep_path",
    "sweep_name",
    "group",
    "dry_run",
    "max_runs",
}


def get_config() -> sws.Config:
    c = sws.Config()
    c.base_config = "src/jaxformers/config/config_qwen_0_6b_loss_bench.py"
    c.sweep_path = "/mnt/carles/llm/.agents/sweeps"
    c.sweep_name = ""
    c.group = []
    c.dry_run = False
    c.max_runs = None
    return c


def _extract_group(raw_group: Any) -> list[dict[str, Any]]:
    if raw_group is None:
        return []
    if not isinstance(raw_group, list):
        raise SweepConfigError("Expected `group` to be a list of flat dicts.")
    if not all(isinstance(item, dict) for item in raw_group):
        raise SweepConfigError("Expected each `group` item to be a dict.")
    return raw_group


def _render_config_file(base_config_path: str, overrides: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("import sws")
    lines.append("")
    lines.append("from jaxformers.bench.sweep_utils import load_config_builder")
    lines.append("")
    lines.append("BASE_CONFIG_PATH = " + repr(base_config_path))
    lines.append("OVERRIDES = " + repr(list(overrides.items())))
    lines.append("")
    lines.append("def get_config() -> sws.Config:")
    lines.append("    c = load_config_builder(BASE_CONFIG_PATH)")
    lines.append("    for key, value in OVERRIDES:")
    lines.append("        c[key] = value")
    lines.append("    return c")
    lines.append("")
    return "\n".join(lines)


def _is_list_value(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _split_overrides(
    *,
    raw_overrides: dict[str, Any],
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    sweep_overrides: dict[str, list[Any]] = {}
    fixed_overrides: dict[str, Any] = {}

    for key, value in raw_overrides.items():
        if _is_list_value(value):
            sweep_overrides[key] = list(value)
        else:
            fixed_overrides[key] = value

    return sweep_overrides, fixed_overrides


def _resolve_output_dir(flat: dict[str, Any]) -> epath.Path:
    sweep_path = epath.Path(str(flat["sweep_path"]))
    sweep_name = str(flat.get("sweep_name", "")).strip()
    if not sweep_name:
        return sweep_path
    return sweep_path / sweep_name


def main(config: sws.FinalConfig) -> None:
    flat = config.to_flat_dict()
    group = _extract_group(flat.get("group"))
    raw_overrides = {key: value for key, value in flat.items() if key not in _META_KEYS}
    sweep_overrides, fixed_overrides = _split_overrides(
        raw_overrides=raw_overrides,
    )

    space = build_sweep_space(overrides=sweep_overrides, group=group)
    run_items: list[tuple[int | None, dict[str, Any]]] = []
    for group_idx, run in space.iter_runs():
        merged = dict(fixed_overrides)
        merged.update(run)
        run_items.append((group_idx, merged))

    max_runs = flat.get("max_runs")
    if max_runs is not None:
        run_items = run_items[: int(max_runs)]

    out_dir = _resolve_output_dir(flat)
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(flat.get("dry_run", False)):
        print(
            "Planned generated configs:",
            len(run_items),
            "| group:",
            len(space.groups) if space.groups else 1,
            "| sweep_keys:",
            list(space.dimensions.keys()),
            "| fixed_keys:",
            list(fixed_overrides.keys()),
            "| out_dir:",
            str(out_dir),
        )
        return

    base_config_path = os.path.abspath(str(flat["base_config"]))
    manifest: list[dict[str, Any]] = []
    for run_idx, (group_idx, run_overrides) in enumerate(run_items):
        cfg_path = out_dir / f"{run_idx}.py"
        cfg_path.write_text(_render_config_file(base_config_path, run_overrides))
        manifest.append(
            {
                "run_idx": run_idx,
                "group_idx": group_idx,
                "config_path": str(cfg_path),
                "overrides": run_overrides,
            }
        )

    (out_dir / "index.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Generated {len(manifest)} configs in {out_dir}")


if __name__ == "__main__":
    sws.run(main)
