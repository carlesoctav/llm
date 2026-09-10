import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import jax
import jax.numpy as jnp

from jaxformers.inference.engine import LLMEngine
from jaxformers.inference.input_batch import SamplingParams
from jaxformers.models import qwen3


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check jaxformers Qwen3 generate vs HF generate (CPU).")
    p.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    p.add_argument("--hf-ckpt-dir", default=str(Path("~/weights/huggingface").expanduser()))
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-num-batched-token", type=int, default=32)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--long-prompt-len", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _load_hf_cpu(model_id: str, hf_ckpt_dir: str):
    local_dir = Path(hf_ckpt_dir).expanduser() / model_id
    tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        local_files_only=True,
        low_cpu_mem_usage=False,
        attn_implementation="sdpa",
        dtype=torch.float32,
    )
    model.eval()
    return tokenizer, model


def _hf_generate_one(
    hf_model,
    input_ids: list[int],
    *,
    max_new_tokens: int,
    ignore_eos: bool,
) -> list[int]:
    input_ids_pt = torch.tensor([input_ids], dtype=torch.long)
    attn_mask = torch.ones_like(input_ids_pt, dtype=torch.long)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    if ignore_eos:
        gen_kwargs["eos_token_id"] = None
        # Avoid warnings when eos_token_id=None.
        if hf_model.config.pad_token_id is None and hf_model.config.eos_token_id is not None:
            gen_kwargs["pad_token_id"] = hf_model.config.eos_token_id
    with torch.no_grad():
        out = hf_model.generate(input_ids=input_ids_pt, attention_mask=attn_mask, **gen_kwargs)
    return out[0].tolist()


def _random_prompt(
    vocab_size: int,
    *,
    length: int,
    seed: int,
    banned_token_ids: set[int],
) -> list[int]:
    rng = np.random.RandomState(seed)
    ids = rng.randint(0, vocab_size, size=(length,), dtype=np.int64).tolist()
    out: list[int] = []
    for x in ids:
        if x in banned_token_ids:
            out.append(1)
        else:
            out.append(int(x))
    return out


def main() -> None:
    args = _parse_args()

    hf_tokenizer, hf_model = _load_hf_cpu(args.model_id, args.hf_ckpt_dir)
    vocab_size = int(hf_model.config.vocab_size)
    banned = set()
    for name in ("eos_token_id", "bos_token_id", "pad_token_id"):
        tok = getattr(hf_model.config, name, None)
        if tok is not None:
            banned.add(int(tok))

    text_prompts = [
        "Hello!",
        "Explain reinforcement learning in one sentence.",
    ]
    text_prompt_ids = [hf_tokenizer.encode(p, add_special_tokens=False) for p in text_prompts]
    random_long_prompt_ids = _random_prompt(
        vocab_size,
        length=args.long_prompt_len,
        seed=args.seed,
        banned_token_ids=banned,
    )

    prompts_token_ids = text_prompt_ids + [random_long_prompt_ids]
    max_prompt_len = max(len(x) for x in prompts_token_ids)
    max_model_len = max_prompt_len + args.max_new_tokens

    devices = jax.devices()
    if len(devices) > 1:
        devices = [devices[0]]

    jax_model = qwen3.load_inference(
        args.model_id,
        parallel_dims={"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1},
        devices=devices,
        hf_ckpt_dir=args.hf_ckpt_dir,
        config_kwargs={
            "max_num_seqs": len(prompts_token_ids),
            "max_num_batched_token": args.max_num_batched_token,
            "max_model_len": max_model_len,
            "page_size": args.page_size,
        },
        param_dtype=jnp.float32,
        kv_dtype=jnp.float32,
    )

    engine = LLMEngine(jax_model)
    sampling_params = [
        SamplingParams(
            temperature=-1.0,
            max_new_tokens=args.max_new_tokens,
            ignore_eos=args.ignore_eos,
        )
        for _ in prompts_token_ids
    ]

    jax_out = engine.generate(prompts_token_ids, sampling_params=sampling_params)
    assert len(jax_out) == len(prompts_token_ids)

    ok = True
    for idx, (prompt_ids, out_dict) in enumerate(zip(prompts_token_ids, jax_out)):
        jax_full = prompt_ids + out_dict["token_ids"]
        hf_full = _hf_generate_one(
            hf_model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            ignore_eos=args.ignore_eos,
        )
        if jax_full != hf_full:
            ok = False
            for j, (a, b) in enumerate(zip(jax_full, hf_full)):
                if a != b:
                    print(f"Mismatch prompt[{idx}] at token[{j}]: jax={a} hf={b}")
                    break
            print(f"JAX len={len(jax_full)} HF len={len(hf_full)}")
            print("JAX tail:", jax_full[-16:])
            print("HF  tail:", hf_full[-16:])
        else:
            print(f"prompt[{idx}] OK (len={len(jax_full)})")

    if not ok:
        raise SystemExit(1)
    print("All prompts matched HF generate.")


if __name__ == "__main__":
    main()
