"""Sweep NTP configs and run each candidate in a separate process.

Usage is native `sws.run(main)`: put sweep dimensions directly on this config,
for example `learning_rate:=[1e-5,1e-4]`. Single values are also valid
dimensions (size 1), so `learning_rate:=1e-5` is equivalent to a one-value sweep.

Use `group=[{...}, {...}]` for bundled parameters. Group keys should be flat
dotted keys (for example `optimizer.b1`) and must use exact keys from the base
NTP config.
"""

import gc
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import sws
from etils import epath

from jaxformers.bench.sweep_utils import (
    build_sweep_space,
    format_cli_value,
    load_config_builder,
    parse_maybe_expr,
    SweepConfigError,
)


_SWEEP_KEYS = {
    "ntp_config_path",
    "sweep_path",
    "sweep_name",
    "group",
    "dry_run",
    "max_runs",
    "hf_home",
}


def get_config() -> sws.Config:
    c = sws.Config()
    c.ntp_config_path = "src/jaxformers/config/config_qwen_0_6b_loss_bench.py"
    c.sweep_path = "/mnt/carles/llm/.agents/reports"
    c.sweep_name = "loss_impl_bench"
    c.group = []
    c.dry_run = False
    c.max_runs = None
    c.hf_home = "/mnt/carles/.cache"

    c.log_name = "noop"
    c.data.transforms.max_length = [8192]
    c.loss_implementation = ["xla_chunked", "reference"]
    return c


def _process_index() -> int:
    try:
        import jax

        return int(jax.process_index())
    except Exception:
        return 0


def _cleanup_after_run() -> None:
    gc.collect()
    try:
        os.remove("/tmp/libtpu_lockfile")
    except FileNotFoundError:
        pass
    except OSError:
        pass


@contextmanager
def _cpu_probe_backend():
    old_platforms = os.environ.get("JAX_PLATFORMS")
    old_platform_name = os.environ.get("JAX_PLATFORM_NAME")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    try:
        yield
    finally:
        if old_platforms is None:
            os.environ.pop("JAX_PLATFORMS", None)
        else:
            os.environ["JAX_PLATFORMS"] = old_platforms
        if old_platform_name is None:
            os.environ.pop("JAX_PLATFORM_NAME", None)
        else:
            os.environ["JAX_PLATFORM_NAME"] = old_platform_name


def _run_one(
    *,
    ntp_config_path: str,
    overrides: dict[str, Any],
    hf_home: str | None,
) -> dict[str, Any]:
    env = os.environ.copy()
    if hf_home:
        env["HF_HOME"] = hf_home

    ntp_script = os.path.abspath("src/jaxformers/train/ntp.py")
    cmd = [sys.executable, ntp_script, "--config", ntp_config_path]
    for key, value in overrides.items():
        cmd.append(f"{key}:={format_cli_value(value)}")

    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")

    parsed: dict[str, Any] | None = None
    for line in out.splitlines():
        if line.startswith("NTP_RESULT "):
            try:
                parsed = json.loads(line.removeprefix("NTP_RESULT ").strip())
            except Exception:
                parsed = None

    if parsed is None:
        parsed = {
            "status": "error",
            "error": f"no NTP_RESULT line (exit_code={proc.returncode})",
        }

    parsed["exit_code"] = proc.returncode
    parsed["log_tail"] = "\n".join(out.splitlines()[-80:])
    _cleanup_after_run()
    return parsed


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4g}"
    except Exception:
        return "-"


def _extract_group(raw_group: Any) -> list[dict[str, Any]]:
    parsed = parse_maybe_expr(raw_group)
    if parsed is None:
        return []
    if not isinstance(parsed, list):
        raise SweepConfigError(
            "Expected `group` to be a list of dicts. "
            "Tip: group keys should be flat dotted keys like {'optimizer.b1': 0.9}."
        )
    return parsed


def _write_report(
    *,
    sweep_cfg: sws.FinalConfig,
    sweep_space,
    results: list[dict[str, Any]],
) -> None:
    report_dir = epath.Path(str(sweep_cfg.sweep_path)) / str(sweep_cfg.sweep_name)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"
    raw_path = report_dir / "results.json"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines: list[str] = []
    lines.append(f"## {sweep_cfg.sweep_name} ({now})\n")
    lines.append(f"- NTP config: `{os.path.abspath(sweep_cfg.ntp_config_path)}`\n")
    lines.append(
        f"- Runs: group={len(sweep_space.groups) if sweep_space.groups else 1}, "
        f"free_dims={{{', '.join(sweep_space.dimensions.keys())}}}, total={len(results)}\n"
    )
    lines.append(
        "- Note: group keys should be flat dotted keys, e.g. "
        '`group=[{"optimizer.b1":0.9}]`.\n'
    )
    lines.append("\n")

    lines.append(
        "| run | group | status | optimizer | loss_impl | max_length | compile_s | program_s | tokens/s | total_mem_gb |\n"
    )
    lines.append("|---:|---:|---|---|---|---:|---:|---:|---:|---:|\n")
    for row in results:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("run_idx", "-")),
                    str(row.get("group_idx", "-")),
                    str(row.get("status", "-")),
                    str(row.get("optimizer_name", "-")),
                    str(row.get("loss_implementation", "-")),
                    str(row.get("max_length", "-")),
                    _format_float(metrics.get("compile_time")),
                    _format_float(metrics.get("program_time")),
                    _format_float(metrics.get("tokens_per_s")),
                    _format_float(metrics.get("total_gb")),
                ]
            )
            + " |\n"
        )

    lines.append("\n### Raw results\n")
    lines.append("```json\n")
    lines.append(json.dumps(results, indent=2, sort_keys=True))
    lines.append("\n```\n")

    exists = report_path.exists() and report_path.stat().length > 0
    with report_path.open("a" if exists else "w") as handle:
        if exists:
            handle.write("\n\n---\n\n")
        else:
            handle.write("# NTP sweep report\n\n")
        handle.writelines(lines)

    with raw_path.open("w") as handle:
        handle.write(json.dumps(results, indent=2, sort_keys=True))

    print(f"Wrote report to {report_path} ({'appended' if exists else 'created'})")


def main(config: sws.FinalConfig) -> None:
    with _cpu_probe_backend():
        base_builder = load_config_builder(os.path.abspath(config.ntp_config_path))
        base_store = base_builder._store  # noqa: SLF001

    sweep_flat = config.to_flat_dict()
    group = _extract_group(sweep_flat.get("group"))
    overrides = {
        key: parse_maybe_expr(value)
        for key, value in sweep_flat.items()
        if key not in _SWEEP_KEYS
    }

    space = build_sweep_space(base_store=base_store, overrides=overrides, group=group)

    max_runs = parse_maybe_expr(sweep_flat.get("max_runs"))
    max_runs = None if max_runs is None else int(max_runs)
    planned_runs = (
        space.run_count if max_runs is None else min(space.run_count, max_runs)
    )

    if bool(parse_maybe_expr(sweep_flat.get("dry_run", False))):
        print(
            "Planned runs:",
            planned_runs,
            "| group:",
            len(space.groups) if space.groups else 1,
            "| free_sweep_keys:",
            list(space.dimensions.keys()),
        )
        return

    results: list[dict[str, Any]] = []
    run_idx = 0
    for group_idx, run_overrides in space.iter_runs():
        if max_runs is not None and run_idx >= max_runs:
            break
        run_result = _run_one(
            ntp_config_path=os.path.abspath(config.ntp_config_path),
            overrides=run_overrides,
            hf_home=getattr(config, "hf_home", None),
        )
        run_result["run_idx"] = run_idx
        run_result["group_idx"] = group_idx
        run_result["sweep_overrides"] = dict(run_overrides)
        results.append(run_result)
        run_idx += 1

    if _process_index() != 0:
        print("Skipping report write on non-zero process index.")
        return
    _write_report(sweep_cfg=config, sweep_space=space, results=results)


if __name__ == "__main__":
    sws.run(main)
