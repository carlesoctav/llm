"""Generate per-run config files from one base config.

Usage:
  python src/jaxformers/make_sweep.py \
    base_config=src/jaxformers/config/config_qwen_0_6b_loss_bench.py \
    dir=/mnt/carles/llm/.agents/sweeps \
    name=loss_impl \
    'loss_implementation:=[\"xla_chunked\",\"reference\"]' \
    'data.transforms.max_length:=[512,1024]' \
    'group:=[{\"optimizer_name\":\"adam\"},{\"optimizer_name\":\"sgd\"}]'

This writes:
  <dir>/<name>/0.py
  <dir>/<name>/1.py
  ...
Each generated file defines `get_config()` that loads `base_config` and applies
the run overrides.
"""

import json
import os
from pathlib import Path
from typing import Any

from etils import epath
import sws

from jaxformers.sweep_utils import build_sweep_space
from jaxformers.sws_utils import run as sws_run


_META_KEYS = {
    "base_config",
    "dir",
    "name",
    "group",
    "dry_run",
}


def get_config() -> sws.Config:
    c = sws.Config()
    c.base_config = ""
    c.dir = ""
    c.name = ""
    c.group = []
    c.dry_run = False
    return c


def render_config_file(base_config_path: str, overrides: dict[str, Any]) -> str:
    lines: list[str] = []
    base_config_name = Path(base_config_path).name
    lines.append("import sws")
    lines.append("from pathlib import Path")
    lines.append("")
    lines.append("from jaxformers.sws_utils import load_config_builder")
    lines.append("")
    lines.append(f"BASE_CONFIG_PATH = str(Path(__file__).with_name({base_config_name!r}))")
    lines.append("OVERRIDES = " + repr(list(overrides.items())))
    lines.append("")
    lines.append("def get_config() -> sws.Config:")
    lines.append("    c = load_config_builder(BASE_CONFIG_PATH)")
    lines.append("    for key, value in OVERRIDES:")
    lines.append("        c[key] = value")
    lines.append("    return c")
    lines.append("")
    return "\n".join(lines)


def is_list_value(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def split_overrides(
    *,
    raw_overrides: dict[str, Any],
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    sweep_overrides: dict[str, list[Any]] = {}
    fixed_overrides: dict[str, Any] = {}

    for key, value in raw_overrides.items():
        if is_list_value(value):
            sweep_overrides[key] = list(value)
        else:
            fixed_overrides[key] = value

    return sweep_overrides, fixed_overrides


def resolve_output_dir(dir_path: str, name: str) -> epath.Path:
    return epath.Path(dir_path) / name


def copy_base_config(base_config_path: str, out_dir: epath.Path) -> epath.Path:
    source = epath.Path(base_config_path)
    destination = out_dir / Path(base_config_path).name

    if source != destination:
        destination.write_text(source.read_text())

    return destination


def main(config: sws.FinalConfig) -> None:
    flat = config.to_flat_dict()
    for required in ("base_config", "dir", "name"):
        if not str(flat[required]).strip():
            raise ValueError(f"`{required}` is required")

    group = flat["group"]
    raw_overrides = {key: value for key, value in flat.items() if key not in _META_KEYS}
    sweep_overrides, fixed_overrides = split_overrides(
        raw_overrides=raw_overrides,
    )

    space = build_sweep_space(overrides=sweep_overrides, group=group)
    run_items: list[tuple[int | None, dict[str, Any]]] = []
    for group_idx, run in space:
        merged = dict(fixed_overrides)
        merged.update(run)
        run_items.append((group_idx, merged))

    out_dir = resolve_output_dir(str(flat["dir"]), str(flat["name"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    if flat["dry_run"]:
        print(
            "Planned generated configs:",
            len(run_items),
            "| group:",
            len(group) if group else 1,
            "| sweep_keys:",
            list(sweep_overrides.keys()),
            "| fixed_keys:",
            list(fixed_overrides.keys()),
            "| out_dir:",
            str(out_dir),
        )
        return

    base_config_path = os.path.abspath(str(flat["base_config"]))
    copied_base_config_path = copy_base_config(base_config_path, out_dir)
    manifest: list[dict[str, Any]] = []
    for run_idx, (group_idx, run_overrides) in enumerate(run_items):
        cfg_path = out_dir / f"{run_idx}.py"
        cfg_path.write_text(render_config_file(str(copied_base_config_path), run_overrides))
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
    sws_run(main)
