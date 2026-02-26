"""Run generated sweep configs against a train script.

Usage:
  python src/jaxformers/bench/sweep.py \
    src/jaxformers/train/ntp.py \
    .agents/sweeps/attn_loss \
    .agents/reports/attn_loss_runs \
    0--3

Args:
  1) train_script: Python file that defines `main(config)`.
  2) sweep_path: Directory containing numbered configs (`0.py`, `1.py`, ...).
  3) output_dir: Directory where per-run JSON results are written (`<idx>.json`).
  4) run_range (optional): `<idx>` or `<start>--<end>` (inclusive).
"""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
import runpy
import sys
import traceback
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Callable

from etils import epath

from jaxformers.bench.sweep_utils import ConfigLoadError


def _usage() -> str:
    return (
        "Usage: python src/jaxformers/bench/sweep.py "
        "[train_script] [sweep_path] [output_dir] [range(optional: xx or xx--yy)]"
    )


def _parse_range(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    if value.startswith("range="):
        value = value[len("range=") :]

    if "--" in value:
        left, right = value.split("--", 1)
    elif "-" in value:
        left, right = value.split("-", 1)
    else:
        idx = int(value)
        if idx < 0:
            raise ValueError("range index must be >= 0")
        return idx, idx

    start = int(left)
    end = int(right)
    if start < 0 or end < 0:
        raise ValueError("range bounds must be >= 0")
    if end < start:
        raise ValueError("range end must be >= start")
    return start, end


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None:
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _load_train_main(train_script_path: str) -> Callable[[Any], Any]:
    module = runpy.run_path(train_script_path)
    train_main = module.get("main")
    if not callable(train_main):
        raise ConfigLoadError(f"`main` function not found in {train_script_path}")
    return train_main


def _load_config_builder_any(path: epath.Path, *, default_func: str = "get_config"):
    path_str = str(path)
    if path_str.startswith("gs://"):
        source = path.read_text()
        namespace: dict[str, Any] = {}
        exec(compile(source, path_str, "exec"), namespace)
        factory = namespace.get(default_func)
        if not callable(factory):
            raise ConfigLoadError(f"Function {default_func!r} not found in {path_str}")
        builder = factory()
        return builder

    module = runpy.run_path(path_str)
    factory = module.get(default_func)
    if not callable(factory):
        raise ConfigLoadError(f"Function {default_func!r} not found in {path_str}")
    return factory()


def _discover_run_configs(sweep_path: epath.Path) -> list[tuple[int, epath.Path]]:
    runs: list[tuple[int, epath.Path]] = []
    for cfg_path in sweep_path.iterdir():
        if cfg_path.suffix != ".py":
            continue
        stem = cfg_path.stem
        if not stem.isdigit():
            continue
        runs.append((int(stem), cfg_path))
    runs.sort(key=lambda item: item[0])
    return runs


def _extract_ntp_result(log_text: str) -> dict[str, Any] | None:
    parsed: dict[str, Any] | None = None
    for line in log_text.splitlines():
        if line.startswith("NTP_RESULT "):
            raw = line.removeprefix("NTP_RESULT ").strip()
            try:
                maybe = json.loads(raw)
                if isinstance(maybe, dict):
                    parsed = maybe
            except Exception:
                continue
    return parsed


def _run_train_worker(
    *,
    send_conn,
    train_script_path: str,
    run_config_path: str,
) -> None:
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    exit_code = 0
    return_value: Any = None

    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        try:
            builder = _load_config_builder_any(epath.Path(run_config_path))
            config = builder.finalize([])
            train_main = _load_train_main(train_script_path)
            return_value = train_main(config)
        except BaseException:
            exit_code = 1
            traceback.print_exc()

    combined_log = stdout_buffer.getvalue()
    stderr_text = stderr_buffer.getvalue()
    if stderr_text:
        combined_log = combined_log + ("\n" if combined_log else "") + stderr_text

    result_payload = None
    if return_value is not None:
        result_payload = _jsonable(return_value)
    if result_payload is None:
        parsed = _extract_ntp_result(combined_log)
        if parsed is not None:
            result_payload = parsed
    if result_payload is None:
        result_payload = {
            "status": "error",
            "error": "no return value and no NTP_RESULT line found",
        }

    if isinstance(result_payload, dict):
        result = dict(result_payload)
    else:
        result = {"result": result_payload}

    if exit_code != 0 and result.get("status") == "ok":
        result["status"] = "error"
        result["error"] = "child process failed"

    result["exit_code"] = exit_code
    result["log_tail"] = "\n".join(combined_log.splitlines()[-120:])

    try:
        send_conn.send(result)
    finally:
        send_conn.close()


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
    train_script_path: str,
    run_config_path: str,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)

    proc = ctx.Process(
        target=_run_train_worker,
        kwargs={
            "send_conn": send_conn,
            "train_script_path": train_script_path,
            "run_config_path": run_config_path,
        },
    )
    proc.start()
    send_conn.close()
    proc.join()

    result: dict[str, Any] | None = None
    if recv_conn.poll():
        try:
            raw = recv_conn.recv()
            if isinstance(raw, dict):
                result = raw
        except Exception:
            result = None
    recv_conn.close()

    if result is None:
        result = {
            "status": "error",
            "error": f"no worker payload (exit_code={proc.exitcode})",
            "exit_code": proc.exitcode,
            "log_tail": "",
        }
    else:
        result["proc_exit_code"] = proc.exitcode
        if proc.exitcode not in (0, None) and result.get("status") == "ok":
            result["status"] = "error"
            result["error"] = f"child process exited with code {proc.exitcode}"

    _cleanup_after_run()
    return result


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print(_usage())
        return 2

    train_script = os.path.abspath(argv[1])
    sweep_path = epath.Path(argv[2])
    output_dir = epath.Path(argv[3])
    run_range = _parse_range(argv[4]) if len(argv) >= 5 else None

    if not os.path.isfile(train_script):
        raise FileNotFoundError(f"train_script not found: {train_script}")
    if not sweep_path.exists():
        raise FileNotFoundError(f"sweep_path not found: {sweep_path}")

    run_configs = _discover_run_configs(sweep_path)
    if not run_configs:
        raise FileNotFoundError(f"No numbered config files found under {sweep_path}")

    if run_range is not None:
        start, end = run_range
        run_configs = [
            (idx, cfg_path)
            for idx, cfg_path in run_configs
            if start <= idx <= end
        ]
        if not run_configs:
            raise ValueError(f"No runs matched range {start}--{end}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Running {len(run_configs)} configs from {sweep_path} "
        f"with {train_script}; writing to {output_dir}"
    )
    for idx, cfg_path in run_configs:
        print(f"[run {idx}] config={cfg_path}")
        result = _run_one(
            train_script_path=train_script,
            run_config_path=str(cfg_path),
        )
        out_path = output_dir / f"{idx}.json"
        out_path.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        print(f"[run {idx}] wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
