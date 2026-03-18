"""Run generated sweep configs against a train script."""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
import runpy
import sys
import traceback
from typing import Any, Callable

from etils import epath
from tqdm.auto import tqdm

from jaxformers.sweep_utils import ConfigLoadError


def usage() -> str:
    return (
        "Usage: python src/jaxformers/sweep.py "
        "[train_script] [sweep_path] [output_dir] [range(optional: xx or xx--yy)]"
    )


def parse_range(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    if "--" in value:
        start_raw, end_raw = value.split("--", 1)
        return int(start_raw), int(end_raw)
    idx = int(value)
    return idx, idx


def process_index() -> int:
    value = os.environ.get("JAX_PROCESS_INDEX")
    if value is None:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


def load_train_main(train_script_path: str) -> Callable[[Any], Any]:
    module = runpy.run_path(train_script_path)
    train_main = module.get("main")
    if not callable(train_main):
        raise ConfigLoadError(f"`main` function not found in {train_script_path}")
    return train_main


def load_module(path: epath.Path) -> dict[str, Any]:
    path_str = str(path)
    if path_str.startswith("gs://"):
        source = path.read_text()
        namespace: dict[str, Any] = {}
        exec(compile(source, path_str, "exec"), namespace)
        return namespace
    return runpy.run_path(path_str)


def load_config_builder(path: epath.Path, default_func: str = "get_config"):
    module = load_module(path)
    factory = module.get(default_func)
    if not callable(factory):
        raise ConfigLoadError(f"Function {default_func!r} not found in {path}")
    return factory()


def discover_run_configs(sweep_path: epath.Path) -> list[tuple[int, epath.Path]]:
    runs: list[tuple[int, epath.Path]] = []
    for cfg_path in sweep_path.iterdir():
        if cfg_path.suffix != ".py":
            continue
        if not cfg_path.stem.isdigit():
            continue
        runs.append((int(cfg_path.stem), cfg_path))
    runs.sort(key=lambda item: item[0])
    return runs


def run_worker(
    *,
    send_conn,
    train_script_path: str,
    run_config_path: str,
) -> None:
    payload: dict[str, Any] = {
        "ok": False,
        "result": None,
        "error": None,
        "overrides": None,
        "config_json": None,
        "run_config_path": run_config_path,
    }
    try:
        module = load_module(epath.Path(run_config_path))
        payload["overrides"] = module.get("OVERRIDES")
        factory = module.get("get_config")
        if not callable(factory):
            raise ConfigLoadError(
                f"`get_config` function not found in {run_config_path}"
            )
        builder = factory()
        config = builder.finalize([])
        payload["config_json"] = json.loads(config.to_json())
        train_main = load_train_main(train_script_path)
        result = train_main(config)
        payload["ok"] = True
        payload["result"] = result
    except BaseException as exc:
        payload["ok"] = False
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
    finally:
        send_conn.send(json.loads(json.dumps(payload, default=str)))
        send_conn.close()


def cleanup_after_run() -> None:
    gc.collect()
    try:
        os.remove("/tmp/libtpu_lockfile")
    except FileNotFoundError:
        pass
    except OSError:
        pass


def run_one(
    *,
    train_script_path: str,
    run_config_path: str,
):
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=run_worker,
        kwargs={
            "send_conn": send_conn,
            "train_script_path": train_script_path,
            "run_config_path": run_config_path,
        },
    )
    proc.start()
    send_conn.close()
    proc.join()

    payload = None
    if recv_conn.poll():
        try:
            payload = recv_conn.recv()
        except Exception as exc:
            payload = {
                "ok": False,
                "result": None,
                "error": f"failed to receive worker payload: {type(exc).__name__}: {exc}",
                "overrides": None,
                "config_json": None,
                "run_config_path": run_config_path,
            }
    recv_conn.close()
    cleanup_after_run()

    if not isinstance(payload, dict):
        payload = {
            "ok": False,
            "result": None,
            "error": f"run failed with no payload (exit_code={proc.exitcode})",
            "overrides": None,
            "config_json": None,
            "run_config_path": run_config_path,
        }

    payload["proc_exit_code"] = proc.exitcode
    if proc.exitcode not in (0, None) and payload.get("ok", False):
        payload["ok"] = False
        payload["error"] = f"child exited with code {proc.exitcode}"
    return payload


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print(usage())
        return 2

    train_script = os.path.abspath(argv[1])
    sweep_path = epath.Path(argv[2])
    output_dir = epath.Path(argv[3])
    run_range = parse_range(argv[4]) if len(argv) >= 5 else None

    if not os.path.isfile(train_script):
        raise FileNotFoundError(f"train_script not found: {train_script}")
    if not sweep_path.exists():
        raise FileNotFoundError(f"sweep_path not found: {sweep_path}")

    run_configs = discover_run_configs(sweep_path)
    if run_range is not None:
        start, end = run_range
        run_configs = [(idx, path) for idx, path in run_configs if start <= idx <= end]
    if not run_configs:
        raise ValueError("No runs found to execute")

    if process_index() != 0:
        print("Skipping sweep execution on non-zero process index.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(
        run_configs, total=len(run_configs), dynamic_ncols=True, file=sys.stdout
    )
    for idx, cfg_path in progress:
        preview = cfg_path.name
        try:
            module = load_module(cfg_path)
            if module.get("OVERRIDES") is not None:
                preview = module.get("OVERRIDES")
        except Exception:
            pass
        progress.set_description(f"run {idx} {preview}")
        payload = run_one(
            train_script_path=train_script,
            run_config_path=str(cfg_path),
        )
        payload["run_idx"] = idx
        payload["config_path"] = str(cfg_path)
        out_path = output_dir / f"{idx}.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
