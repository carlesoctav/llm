import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from jaxformers.inference import LLM
from jaxformers.models import qwen3


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference throughput (vLLM-style headline).")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tp-size", type=int, default=4)

    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)

    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--page-size", type=int, default=32)

    # Match vLLM bench defaults: temperature=1.0 and ignore_eos=True.
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--respect-eos",
        action="store_true",
        help="Stop sequences early when EOS is generated (vLLM bench ignores EOS).",
    )
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if len(jax.devices()) < args.tp_size:
        raise ValueError(
            f"Need at least tp_size devices ({args.tp_size}), got {len(jax.devices())}"
        )

    parallel_dims = {"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": args.tp_size}
    model = qwen3.load(
        model_id=args.model,
        parallel_dims=parallel_dims,
        devices=jax.devices()[: args.tp_size],
        multihost=False,
        additional_config={
            "attn_implementation": "sdpa",
            "sequence_parallelism": True,
            "gradient_checkpointing": False,
            "loss_parallel": True,
            "remat_layer": False,
        },
        param_dtype=jnp.bfloat16,
    )

    llm = LLM(
        model,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        page_size=args.page_size,
        seed=args.seed,
    )

    vocab_size = int(model.config.vocab_size)
    rng = np.random.default_rng(args.seed)
    prompts = [
        rng.integers(0, vocab_size, size=(args.input_len,), dtype=np.int32).tolist()
        for _ in range(args.num_prompts)
    ]

    # Warmup (compile).
    warmup_prompts = prompts[: min(args.num_prompts, args.max_num_seqs)]
    llm.generate(
        warmup_prompts,
        max_tokens=min(args.output_len, 2),
        ignore_eos=True,
        temperature=args.temperature,
    )

    start = time.perf_counter()
    llm.generate(
        prompts,
        max_tokens=args.output_len,
        ignore_eos=not args.respect_eos,
        temperature=args.temperature,
    )
    end = time.perf_counter()

    elapsed = end - start
    total_prompt_tokens = args.num_prompts * args.input_len
    total_output_tokens = args.num_prompts * args.output_len
    total_tokens = total_prompt_tokens + total_output_tokens

    print(
        f"Throughput: {args.num_prompts / elapsed:.2f} requests/s, "
        f"{total_tokens / elapsed:.2f} total tokens/s, "
        f"{total_output_tokens / elapsed:.2f} output tokens/s"
    )
    print(f"Total num prompt tokens:  {total_prompt_tokens}")
    print(f"Total num output tokens:  {total_output_tokens}")

    if args.output_json:
        results = {
            "elapsed_time": elapsed,
            "num_requests": args.num_prompts,
            "total_num_tokens": total_tokens,
            "requests_per_second": args.num_prompts / elapsed,
            "tokens_per_second": total_tokens / elapsed,
            "output_tokens_per_second": total_output_tokens / elapsed,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
