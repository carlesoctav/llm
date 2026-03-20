#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jaxformers.models import gemma3


def _resolve_source(model_id: str, local_dir: str | None) -> str:
    if local_dir is None:
        return model_id
    local_path = Path(local_dir).expanduser() / model_id
    if local_path.exists():
        return str(local_path)
    return model_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CPU correctness check for jaxformers Gemma-3 vs HF."
    )
    parser.add_argument("--model-id", default="google/gemma-3-1b-it")
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--prompt", default="hallo saya makan nasi goreng")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()

    model_source = _resolve_source(args.model_id, args.local_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_source,
            low_cpu_mem_usage=False,
            attn_implementation="eager",
            torch_dtype=torch.float32,
        )
    except Exception as exc:
        print("Failed to load HF model/tokenizer.")
        print(f"Source: {model_source}")
        print(f"Error: {exc}")
        if "gated" in str(exc).lower() or "401" in str(exc):
            print(
                "Hint: Gemma-3 is gated. Run `huggingface-cli login` or use --local-dir with a downloaded checkpoint."
            )
        return 2

    devices = jax.devices("cpu")
    try:
        jax_model = gemma3.load(
            model_id=args.model_id,
            parallel_dims={"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1},
            devices=devices,
            local_dir=args.local_dir,
            additional_config={
                "attn_implementation": "eager",
                "sequence_parallelism": False,
            },
            param_dtype=jnp.float32,
        )
    except Exception as exc:
        print("Failed to load jaxformers Gemma-3 model.")
        print(f"Model id: {args.model_id}")
        print(f"Local dir: {args.local_dir}")
        print(f"Error: {exc}")
        return 3

    if args.seq_len is None:
        hf_token = tokenizer(args.prompt, return_tensors="pt")
        import jax.numpy as jnp

        jax_token = {
            k: jnp.asarray(v)
            for k, v in tokenizer(args.prompt, return_tensors="np").items()
        }
        logits_to_keep = 0
    else:
        seq_len = int(args.seq_len)
        vocab = int(hf_model.config.vocab_size)
        input_ids = (torch.arange(seq_len).unsqueeze(0) % vocab).to(torch.long)
        attention_mask = torch.ones_like(input_ids)
        hf_token = {"input_ids": input_ids, "attention_mask": attention_mask}
        jax_token = {
            "input_ids": jnp.asarray(input_ids.cpu().numpy(), dtype=jnp.int32),
            "attention_mask": jnp.asarray(
                attention_mask.cpu().numpy(), dtype=jnp.int32
            ),
        }
        logits_to_keep = 1

    with torch.no_grad():
        hf_logits = (
            hf_model(**hf_token, logits_to_keep=logits_to_keep)
            .logits.cpu()
            .numpy()
            .astype(np.float32)
        )

    jax_logits = np.asarray(
        jax_model.forward(
            **jax_token,
            weights=jax_model.weights,
            logits_to_keep=logits_to_keep,
        )
    )

    abs_diff = np.abs(jax_logits - hf_logits)
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    print("shape:", jax_logits.shape)
    print("max_abs_diff:", max_abs)
    print("mean_abs_diff:", mean_abs)
    print("atol:", args.atol, "rtol:", args.rtol)

    try:
        np.testing.assert_allclose(
            jax_logits, hf_logits, atol=args.atol, rtol=args.rtol
        )
    except AssertionError as exc:
        print("FAIL: logits are not close enough.")
        print(exc)
        return 1

    print("PASS: logits match within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
