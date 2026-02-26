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
import multiprocessing as mp
import os
import sys
import traceback
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from typing import Any

import sws
from etils import epath

from jaxformers.bench.sweep_utils import (
    SweepConfigError,
    build_sweep_space,
    format_cli_value,
    load_config_builder,
)


_SWEEP_KEYS = {
    "ntp_config_path",
    "sweep_path",
    "sweep_name",
    "range",
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
    c.range = None
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


def _run_one(
    *,
    ntp_config_path: str,
    overrides: dict[str, Any],
    hf_home: str | None,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)

    proc = ctx.Process(
        target=_run_ntp_worker,
        kwargs={
            "send_conn": send_conn,
            "ntp_config_path": ntp_config_path,
            "overrides": overrides,
            "hf_home": hf_home,
        },
    )
    proc.start()
    send_conn.close()
    proc.join()

    parsed: dict[str, Any] | None = None
    if recv_conn.poll():
        try:
            parsed = recv_conn.recv()
        except Exception:
            parsed = None
    recv_conn.close()

    if parsed is None:
        parsed = {
            "status": "error",
            "error": f"no NTP_RESULT line (exit_code={proc.exitcode})",
            "log_tail": "",
        }

    parsed["exit_code"] = proc.exitcode
    if proc.exitcode not in (0, None) and parsed.get("status") == "ok":
        parsed["status"] = "error"
        parsed["error"] = f"child process exited with code {proc.exitcode}"

    _cleanup_after_run()
    return parsed


def _run_ntp_worker(
    *,
    send_conn,
    ntp_config_path: str,
    overrides: dict[str, Any],
    hf_home: str | None,
) -> None:
    out_stream = StringIO()
    err_stream = StringIO()
    exit_code = 0

    if hf_home:
        os.environ["HF_HOME"] = hf_home

    with redirect_stdout(out_stream), redirect_stderr(err_stream):
        try:
            from jaxformers.train import ntp as ntp_train

            builder = load_config_builder(ntp_config_path)
            tokens = [f"{k}:={format_cli_value(v)}" for k, v in overrides.items()]
            config = builder.finalize(tokens)
            ntp_train.main(config)
        except BaseException:
            exit_code = 1
            traceback.print_exc()

    out = out_stream.getvalue()
    err = err_stream.getvalue()
    combined = out + ("\n" + err if err else "")

    parsed: dict[str, Any] | None = None
    for line in combined.splitlines():
        if line.startswith("NTP_RESULT "):
            try:
                parsed = json.loads(line.removeprefix("NTP_RESULT ").strip())
            except Exception:
                parsed = None

    if parsed is None:
        parsed = {
            "status": "error",
            "error": f"no NTP_RESULT line (exit_code={exit_code})",
        }

    parsed["exit_code"] = exit_code
    parsed["log_tail"] = "\n".join(combined.splitlines()[-80:])

    try:
        send_conn.send(parsed)
    finally:
        send_conn.close()


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4g}"
    except Exception:
        return "-"


def _extract_group(raw_group: Any) -> list[dict[str, Any]]:
    if raw_group is None:
        return []
    if not isinstance(raw_group, list):
        raise SweepConfigError(
            "Expected `group` to be a list of dicts. "
            "Tip: group keys should be flat dotted keys like {'optimizer.b1': 0.9}."
        )
    return raw_group


def _extract_raw_token_value(argv: list[str], key: str) -> str | None:
    prefixes = (
        f"{key}:=",
        f"{key}=",
        f"c.{key}:=",
        f"c.{key}=",
    )
    for tok in reversed(argv):
        for prefix in prefixes:
            if tok.startswith(prefix):
                return tok[len(prefix) :]
    return None


def _parse_run_range(raw_value: Any, argv: list[str]) -> tuple[int, int] | None:
    source = _extract_raw_token_value(argv, "range")
    value = source if source is not None else raw_value
    if value is None:
        return None

    if isinstance(value, int):
        return value, value

    text = str(value).strip()
    if not text:
        return None

    if "--" in text:
        start_txt, end_txt = text.split("--", 1)
    elif "-" in text:
        start_txt, end_txt = text.split("-", 1)
    else:
        idx = int(text)
        return idx, idx

    start = int(start_txt)
    end = int(end_txt)
    if end < start:
        raise ValueError(f"Invalid range {text!r}: end must be >= start.")
    return start, end


def _count_runs_in_range(total_runs: int, run_range: tuple[int, int] | None) -> int:
    if total_runs <= 0:
        return 0
    if run_range is None:
        return total_runs
    start, end = run_range
    start = max(start, 0)
    end = min(end, total_runs - 1)
    if end < start:
        return 0
    return end - start + 1


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
    argv = sys.argv[1:]
    sweep_flat = config.to_flat_dict()
    group = _extract_group(sweep_flat.get("group"))
    run_range = _parse_run_range(sweep_flat.get("range"), argv)
    overrides = {
        key: value
        for key, value in sweep_flat.items()
        if key not in _SWEEP_KEYS
    }

    space = build_sweep_space(overrides=overrides, group=group)

    max_runs = sweep_flat.get("max_runs")
    max_runs = None if max_runs is None else int(max_runs)
    planned_runs = _count_runs_in_range(space.run_count, run_range)
    if max_runs is not None:
        planned_runs = min(planned_runs, max_runs)

    if bool(sweep_flat.get("dry_run", False)):
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
    exec_count = 0
    for full_idx, (group_idx, run_overrides) in enumerate(space.iter_runs()):
        if run_range is not None and not (run_range[0] <= full_idx <= run_range[1]):
            continue
        if max_runs is not None and exec_count >= max_runs:
            break
        run_result = _run_one(
            ntp_config_path=os.path.abspath(config.ntp_config_path),
            overrides=run_overrides,
            hf_home=getattr(config, "hf_home", None),
        )
        run_result["run_idx"] = full_idx
        run_result["group_idx"] = group_idx
        run_result["sweep_overrides"] = dict(run_overrides)
        results.append(run_result)
        exec_count += 1

    if _process_index() != 0:
        print("Skipping report write on non-zero process index.")
        return
    _write_report(sweep_cfg=config, sweep_space=space, results=results)


if __name__ == "__main__":
    sws.run(main)
