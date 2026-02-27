import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run jaxformers + vLLM-TPU throughput bench with matched KV cache."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tp-size", type=int, default=4)

    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--random-input-len", type=int, default=1024)
    parser.add_argument("--random-output-len", type=int, default=128)
    parser.add_argument("--random-prefix-len", type=int, default=0)

    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=32)

    parser.add_argument(
        "--page-size",
        type=int,
        action="append",
        default=None,
        help="Repeat to sweep multiple page sizes (e.g. --page-size 16 --page-size 32).",
    )
    parser.add_argument("--prompt-format", type=str, choices=["text", "tokens"], default="text")

    parser.add_argument("--vllm-block-size", type=int, default=16)
    parser.add_argument("--vllm-python", type=str, default="/mnt/carles/vllm-tpu/.venv/bin/python")
    parser.add_argument("--our-python", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/bench_compare")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    our_python = Path(args.our_python) if args.our_python is not None else repo_root / ".venv/bin/python"
    vllm_python = Path(args.vllm_python)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    page_sizes = args.page_size if args.page_size is not None else [64]

    results_summary: list[dict] = []

    for page_size in page_sizes:
        pages_per_seq = int(math.ceil(args.max_model_len / page_size))
        kv_capacity_tokens = args.max_num_seqs * pages_per_seq * page_size
        vllm_num_blocks_override = int(math.ceil(kv_capacity_tokens / args.vllm_block_size))

        stamp = int(time.time())
        ours_json = (output_dir / f"jaxformers_ps{page_size}_{stamp}.json").resolve()
        vllm_json = (output_dir / f"vllm_ps{page_size}_{stamp}.json").resolve()

        print("")
        print(f"== page_size={page_size} ==")
        print(f"jaxformers KV capacity tokens: {kv_capacity_tokens}")
        print(f"vLLM block_size={args.vllm_block_size} -> num_gpu_blocks_override={vllm_num_blocks_override}")

        ours_cmd = [
            str(our_python),
            "src/jaxformers/bench/bench_inference_throughput.py",
            "--backend",
            "jaxformers",
            "--dataset-name",
            "random",
            "--model",
            args.model,
            "--tp-size",
            str(args.tp_size),
            "--num-prompts",
            str(args.num_prompts),
            "--random-input-len",
            str(args.random_input_len),
            "--random-output-len",
            str(args.random_output_len),
            "--random-prefix-len",
            str(args.random_prefix_len),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--max-model-len",
            str(args.max_model_len),
            "--page-size",
            str(page_size),
            "--temperature",
            "1.0",
            "--prompt-format",
            args.prompt_format,
            "--seed",
            str(args.seed),
            "--output-json",
            str(ours_json),
        ]

        print("")
        print("Running jaxformers bench...")
        print(" ".join(ours_cmd))
        subprocess.run(ours_cmd, check=True, cwd=repo_root)

        vllm_env = dict(os.environ)
        vllm_env["VLLM_TARGET_DEVICE"] = "tpu"
        vllm_env["VLLM_XLA_USE_SPMD"] = "1"

        vllm_cmd = [
            str(vllm_python),
            "-m",
            "vllm.entrypoints.cli.main",
            "bench",
            "throughput",
            "--backend",
            "vllm",
            "--dataset-name",
            "random",
            "--model",
            args.model,
            "--tensor-parallel-size",
            str(args.tp_size),
            "--max-model-len",
            str(args.max_model_len),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--num-prompts",
            str(args.num_prompts),
            "--random-input-len",
            str(args.random_input_len),
            "--random-output-len",
            str(args.random_output_len),
            "--random-prefix-len",
            str(args.random_prefix_len),
            "--disable-detokenize",
            "--dtype",
            "bfloat16",
            "--kv-cache-dtype",
            "bfloat16",
            "--block-size",
            str(args.vllm_block_size),
            "--num-gpu-blocks-override",
            str(vllm_num_blocks_override),
            "--seed",
            str(args.seed),
            "--output-json",
            str(vllm_json),
        ]

        print("")
        print("Running vLLM-TPU bench...")
        print(" ".join(vllm_cmd))
        subprocess.run(vllm_cmd, check=True, cwd="/mnt/carles/vllm-tpu", env=vllm_env)

        ours = load_json(ours_json)
        vllm = load_json(vllm_json)

        ours_elapsed = float(ours["elapsed_time"])
        vllm_elapsed = float(vllm["elapsed_time"])
        total_output_tokens = float(args.num_prompts * args.random_output_len)

        ours_out_tps = total_output_tokens / ours_elapsed
        vllm_out_tps = total_output_tokens / vllm_elapsed

        print("")
        print(
            f"jaxformers: {ours['requests_per_second']:.2f} req/s, "
            f"{ours['tokens_per_second']:.2f} tok/s, {ours_out_tps:.2f} out tok/s"
        )
        print(
            f"vllm-tpu:   {vllm['requests_per_second']:.2f} req/s, "
            f"{vllm['tokens_per_second']:.2f} tok/s, {vllm_out_tps:.2f} out tok/s"
        )
        print(f"ratio (jaxformers/vllm) total tok/s: {ours['tokens_per_second'] / vllm['tokens_per_second']:.3f}")

        results_summary.append(
            {
                "page_size": page_size,
                "kv_capacity_tokens": kv_capacity_tokens,
                "vllm_block_size": args.vllm_block_size,
                "vllm_num_gpu_blocks_override": vllm_num_blocks_override,
                "jaxformers": ours,
                "vllm": vllm,
                "jaxformers_output_tokens_per_second": ours_out_tps,
                "vllm_output_tokens_per_second": vllm_out_tps,
            }
        )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(results_summary, f, indent=4)
    print("")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
