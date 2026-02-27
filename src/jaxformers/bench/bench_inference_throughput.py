import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from jaxformers.inference import LLM
from jaxformers.models import qwen3


def gen_prompt_decode_to_target_len(
    *,
    tokenizer,
    token_sequence: list[int],
    target_token_len: int,
    max_retry: int,
    rng: np.random.Generator,
) -> str:
    remain_num_try = max_retry
    while True:
        prompt = tokenizer.decode(token_sequence)
        token_sequence = tokenizer.encode(prompt, add_special_tokens=False)
        if remain_num_try <= 0:
            break

        if len(token_sequence) == target_token_len:
            break
        if len(token_sequence) < target_token_len:
            extra_tokens = rng.integers(
                0,
                tokenizer.vocab_size,
                size=target_token_len - len(token_sequence),
            ).tolist()
            token_sequence.extend(extra_tokens)
        if len(token_sequence) > target_token_len:
            token_sequence = token_sequence[:target_token_len]

        remain_num_try -= 1
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark inference throughput (vLLM-style headline)."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tp-size", type=int, default=4)

    parser.add_argument("--backend", type=str, default="jaxformers")
    parser.add_argument("--dataset-name", type=str, default="random")
    parser.add_argument("--num-prompts", type=int, default=256)

    # Match vLLM bench random dataset flags.
    parser.add_argument("--random-input-len", type=int, default=None)
    parser.add_argument("--random-output-len", type=int, default=None)
    parser.add_argument("--random-range-ratio", type=float, default=0.0)
    parser.add_argument("--random-prefix-len", type=int, default=0)

    # Back-compat with the first draft of this script.
    parser.add_argument("--input-len", type=int, default=None)
    parser.add_argument("--output-len", type=int, default=None)

    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--page-size", type=int, default=32)

    parser.add_argument(
        "--prompt-format",
        type=str,
        choices=["text", "tokens"],
        default="text",
        help="Use text prompts (vLLM-like) or token-id prompts.",
    )

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

    if args.backend != "jaxformers":
        raise ValueError("Only --backend=jaxformers is supported")
    if args.dataset_name != "random":
        raise ValueError("Only --dataset-name=random is supported")

    if args.random_input_len is None:
        if args.input_len is None:
            random_input_len = 1024
        else:
            random_input_len = args.input_len
    else:
        random_input_len = args.random_input_len

    if args.random_output_len is None:
        if args.output_len is None:
            random_output_len = 128
        else:
            random_output_len = args.output_len
    else:
        random_output_len = args.random_output_len

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

    tok = model.tokenizer
    num_special_tokens = int(tok.num_special_tokens_to_add())
    real_input_len = max(0, int(random_input_len) - num_special_tokens)
    input_len_no_special = args.random_prefix_len + real_input_len
    if input_len_no_special < 1:
        raise ValueError("random input len too small after special tokens")

    rng = np.random.default_rng(args.seed)
    vocab_size = int(tok.vocab_size)
    prohibited_tokens = tok.all_special_ids
    all_tokens = np.arange(vocab_size)
    allowed_tokens = np.array(list(set(all_tokens) - set(prohibited_tokens)))

    if args.random_range_ratio != 0.0:
        raise ValueError("--random-range-ratio is not supported yet (use 0.0)")

    input_lens = np.full((args.num_prompts,), input_len_no_special, dtype=np.int32)
    output_lens = np.full((args.num_prompts,), int(random_output_len), dtype=np.int32)

    offsets = rng.integers(0, vocab_size, size=args.num_prompts)
    prefix_token_ids = (
        allowed_tokens[rng.integers(0, len(allowed_tokens), size=args.random_prefix_len)]
        .astype(np.int32)
        .tolist()
        if args.random_prefix_len > 0
        else []
    )

    prompts: list[str] | list[list[int]]
    if args.prompt_format == "tokens":
        prompts = []
        for i in range(args.num_prompts):
            inner_seq = allowed_tokens[
                (int(offsets[i]) + i + np.arange(int(input_lens[i]) - len(prefix_token_ids)))
                % len(allowed_tokens)
            ].tolist()
            token_ids = prefix_token_ids + inner_seq
            prompts.append([int(x) for x in token_ids])
    else:
        prompts = []
        for i in range(args.num_prompts):
            inner_len = int(input_lens[i]) - len(prefix_token_ids)
            inner_seq = allowed_tokens[
                (int(offsets[i]) + i + np.arange(inner_len)) % len(allowed_tokens)
            ].tolist()
            token_sequence = prefix_token_ids + inner_seq
            prompt = gen_prompt_decode_to_target_len(
                tokenizer=tok,
                token_sequence=[int(x) for x in token_sequence],
                target_token_len=int(input_lens[i]),
                max_retry=10,
                rng=rng,
            )
            prompts.append(prompt)

    warmup_prompts = prompts[: min(args.num_prompts, args.max_num_seqs)]
    llm.generate(
        warmup_prompts,
        max_tokens=min(int(output_lens[0]), 2),
        ignore_eos=True,
        temperature=args.temperature,
    )

    start = time.perf_counter()
    llm.generate(
        prompts,
        max_tokens=int(output_lens[0]),
        ignore_eos=not args.respect_eos,
        temperature=args.temperature,
    )
    end = time.perf_counter()

    elapsed = end - start
    total_prompt_tokens = int(input_lens.sum())
    total_output_tokens = int(output_lens.sum())
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
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
