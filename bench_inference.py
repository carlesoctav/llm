import argparse
import os
import time
from random import randint, seed

import jax
import jax.numpy as jnp

from jaxformers.inference.input_batch import SamplingParams
from jaxformers.inference.model_runner import ModelRunner
from jaxformers.models import qwen3


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark jaxformers inference engine.")
    p.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    p.add_argument("--hf-ckpt-dir", default=os.path.expanduser("~/weights/huggingface"))

    p.add_argument("--num-prompts", type=int, default=32, dest="num_prompts")
    # Backwards-compatible alias.
    p.add_argument("--num-seqs", type=int, dest="num_prompts", help=argparse.SUPPRESS)
    p.add_argument("--min-input-len", type=int, default=64)
    p.add_argument("--max-input-len", type=int, default=256)
    p.add_argument("--min-output-len", type=int, default=128)
    p.add_argument("--max-output-len", type=int, default=256)

    # Engine / KV cache sizing.
    p.add_argument("--max-num-seqs", type=int, default=32, dest="max_num_seqs")
    # Backwards-compatible alias.
    p.add_argument("--max-num-request", type=int, dest="max_num_seqs", help=argparse.SUPPRESS)
    p.add_argument("--max-num-batched-token", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=768)
    p.add_argument("--page-size", type=int, default=64)

    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--ignore-eos", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    seed(args.seed)

    # Helpful when debugging locally; TPU runs should leave this unset.
    if os.environ.get("JAX_PLATFORMS") == "cpu":
        print("JAX_PLATFORMS=cpu (CPU benchmark)")

    if args.max_input_len > args.max_num_batched_token:
        print(
            "Note: --max-input-len exceeds --max-num-batched-token; "
            "prompts will be prefetched in chunks."
        )

    if args.max_model_len < (args.max_input_len + args.max_output_len):
        raise ValueError("--max-model-len must be >= max_input_len + max_output_len to avoid early stops.")

    t0 = time.time()
    devices = jax.devices()
    if len(devices) > 1:
        devices = [devices[0]]
    model = qwen3.load_inference(
        args.model_id,
        parallel_dims={"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1},
        devices=devices,
        hf_ckpt_dir=args.hf_ckpt_dir,
        config_kwargs={
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_token": args.max_num_batched_token,
            "max_model_len": args.max_model_len,
            "page_size": args.page_size,
        },
        param_dtype=jnp.bfloat16,
        kv_dtype=jnp.bfloat16,
    )
    runner = ModelRunner(model)
    init_s = time.time() - t0

    # Warm up (also validates basic correctness).
    runner.add_request(
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_new_tokens=1, temperature=-1.0),
        req_id="warmup",
    )
    while runner.has_pending():
        runner.step()

    prompt_token_ids: list[list[int]] = []
    sampling_params: list[SamplingParams] = []
    for i in range(args.num_prompts):
        in_len = randint(args.min_input_len, args.max_input_len)
        out_len = randint(args.min_output_len, args.max_output_len)
        prompt_token_ids.append([randint(0, 10000) for _ in range(in_len)])
        sampling_params.append(
            SamplingParams(
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                max_new_tokens=out_len,
                ignore_eos=args.ignore_eos,
            )
        )
        runner.add_request(prompt_token_ids[-1], sampling_params[-1], req_id=f"req-{i}")

    total_prompt_tokens = sum(len(x) for x in prompt_token_ids)
    total_output_tokens = sum(sp.max_new_tokens for sp in sampling_params)
    total_tokens = total_prompt_tokens + total_output_tokens

    t1 = time.time()
    finished = 0
    while runner.has_pending():
        done = runner.step()
        finished += len(done)
    dt = time.time() - t1

    out_tok_s = total_output_tokens / max(dt, 1e-9)
    total_tok_s = total_tokens / max(dt, 1e-9)
    print(f"Backend: {jax.default_backend()}, Devices (used): {len(devices)}")
    print(
        f"Init+precompile: {init_s:.2f}s, Prompts: {args.num_prompts}, Finished: {finished}, "
        f"Prompt tokens: {total_prompt_tokens}, Output tokens: {total_output_tokens}, Total: {total_tokens}"
    )
    print(
        f"Time: {dt:.2f}s, Throughput (output tok/s): {out_tok_s:.2f}, "
        f"(total tok/s): {total_tok_s:.2f}"
    )


if __name__ == "__main__":
    main()
