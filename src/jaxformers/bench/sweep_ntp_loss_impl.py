import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


MAX_LENGTHS = (512, 1024, 2048, 4096, 8192)
MAX_LENGTHS = (8192,)
LOSS_IMPLS = ("xla_chunked", "reference")


def _run_one(
    *,
    python: str,
    bench_script: str,
    config_path: str,
    hf_home: str | None,
    loss_impl: str,
    max_length: int,
    exp_name: str,
    timed_steps: int,
    warmup_steps: int,
    batch_size: int | None,
) -> dict:
    env = os.environ.copy()
    if hf_home:
        env["HF_HOME"] = hf_home

    cmd = [
        python,
        bench_script,
        "--config",
        config_path,
        "--loss-impl",
        loss_impl,
        "--max-length",
        str(max_length),
        "--warmup-steps",
        str(warmup_steps),
        "--timed-steps",
        str(timed_steps),
        "--exp-name",
        exp_name,
    ]
    if batch_size is not None:
        cmd += ["--batch-size", str(batch_size)]

    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    bench = None
    for line in out.splitlines():
        if line.startswith("BENCH_RESULT "):
            try:
                bench = json.loads(line.removeprefix("BENCH_RESULT ").strip())
            except Exception:
                bench = None
    if bench is None:
        bench = {
            "status": "error",
            "loss_impl": loss_impl,
            "max_length": max_length,
            "exp_name": exp_name,
            "error": f"no BENCH_RESULT line (exit_code={proc.returncode})",
        }
    bench["exit_code"] = proc.returncode
    bench["log_tail"] = "\n".join(out.splitlines()[-50:])
    return bench


def _format_float(x) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.4g}"
    except Exception:
        return "-"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
        help="Path to a sws get_config() python file.",
    )
    parser.add_argument("--hf-home", default="/mnt/carles/.cache")
    parser.add_argument("--report-path", default="/.agents/reports/loss_impl_bench.md")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--timed-steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--loss-impls",
        nargs="+",
        default=list(LOSS_IMPLS),
        choices=LOSS_IMPLS,
        help="Loss implementations to benchmark.",
    )
    args = parser.parse_args(argv)

    python = sys.executable
    bench_script = os.path.abspath("src/jaxformers/bench/bench_ntp_loss_impl.py")
    config_path = os.path.abspath(args.config)

    results: list[dict] = []
    for max_length in MAX_LENGTHS:
        for loss_impl in args.loss_impls:
            bs = args.batch_size
            exp_name = f"{bs}_{loss_impl}" if bs is not None else f"{loss_impl}"
            results.append(
                _run_one(
                    python=python,
                    bench_script=bench_script,
                    config_path=config_path,
                    hf_home=args.hf_home,
                    loss_impl=loss_impl,
                    max_length=max_length,
                    exp_name=exp_name,
                    timed_steps=args.timed_steps,
                    warmup_steps=args.warmup_steps,
                    batch_size=args.batch_size,
                )
            )

    report_path = args.report_path
    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "a", encoding="utf-8"):
            pass
    except OSError:
        report_path = os.path.abspath(".agents/reports/loss_impl_bench.md")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    run_lines: list[str] = []
    run_lines.append(f"## Qwen3-0.6B ({now})\n")
    run_lines.append(f"- Config: `{config_path}`\n")
    run_lines.append(
        f"- Sweep: packing=True, max_length ∈ {{{', '.join(map(str, MAX_LENGTHS))}}}, loss_impl ∈ {{{', '.join(args.loss_impls)}}}\n"
    )
    run_lines.append(f"- Steps: warmup={args.warmup_steps}, timed={args.timed_steps}\n")
    if args.batch_size is not None:
        run_lines.append(f"- global_batch_size: `{args.batch_size}`\n")
    optimizer = next((r.get("optimizer") for r in results if r.get("optimizer")), None)
    if optimizer is not None:
        run_lines.append(f"- optimizer: `{optimizer}`\n")
    run_lines.append("\n")

    run_lines.append(
        "| max_length | loss_impl | status | compile_time_s | step_time_s | tokens/step | tokens/s | total_mem_gb |\n"
    )
    run_lines.append("|---:|---|---|---:|---:|---:|---:|---:|\n")
    for r in results:
        mem_total = None
        if isinstance(r.get("memory"), dict):
            mem_total = r["memory"].get("total_gb")
        run_lines.append(
            "| "
            + " | ".join(
                [
                    str(r.get("max_length", "-")),
                    str(r.get("loss_impl", "-")),
                    str(r.get("status", "-")),
                    _format_float(r.get("compile_time_s")),
                    _format_float(r.get("step_time_s")),
                    str(r.get("tokens_per_step", "-")),
                    _format_float(r.get("tokens_per_s")),
                    str(mem_total if mem_total is not None else "-"),
                ]
            )
            + " |\n"
        )

    run_lines.append("\n### Raw results\n")
    run_lines.append("```json\n")
    run_lines.append(json.dumps(results, indent=2, sort_keys=True))
    run_lines.append("\n```\n")

    exists = os.path.exists(report_path) and os.path.getsize(report_path) > 0
    with open(report_path, "a" if exists else "w", encoding="utf-8") as f:
        if exists:
            f.write("\n\n---\n\n")
        else:
            f.write("# Loss implementation benchmark\n\n")
        f.writelines(run_lines)

    print(f"Wrote report to {report_path} ({'appended' if exists else 'created'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
