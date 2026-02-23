import os
import sys
import gc
import time
import traceback

import jax
import jax.numpy as jnp

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config.test_config_qwen4b_ntp as cfg
from jaxformers.train import ntp


def _finalize(overrides: list[str]):
    builder = cfg.get_config()
    final, unused = builder.finalize(overrides, return_unused_argv=True)
    if unused:
        raise ValueError(f"Unused argv tokens in overrides: {unused}")
    return final


def _make_batch(batch_size: int, seq_len: int, vocab_size: int, *, seed: int = 0):
    key = jax.random.PRNGKey(seed)
    key_x, key_y = jax.random.split(key, 2)
    input_ids = jax.random.randint(
        key_x, (batch_size, seq_len), 0, vocab_size, dtype=jnp.int32
    )
    labels = jax.random.randint(
        key_y, (batch_size, seq_len), 0, vocab_size, dtype=jnp.int32
    )
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    return {
        "inputs": {"input_ids": input_ids, "attention_mask": attention_mask},
        "labels": labels,
    }


def _try_compile_and_run(config, model, batch, rng):
    try:
        step_fn = jax.jit(lambda m, b, r: ntp.train_step(config, m, b, r))

        t0 = time.monotonic()
        compiled = step_fn.lower(model, batch, rng).compile()
        compile_s = time.monotonic() - t0

        t1 = time.monotonic()
        model_out, aux = compiled(model, batch, rng)
        jax.block_until_ready(aux["token_count"])
        run_s = time.monotonic() - t1

        mem = compiled.memory_analysis()
        return True, model_out, mem, compile_s, run_s, None
    except Exception as e:  # pylint: disable=broad-except
        return False, model, None, None, None, e


def _memory_total_gb(mem) -> float:
    if mem is None:
        return float("nan")
    total_bytes = (
        mem.output_size_in_bytes
        + mem.temp_size_in_bytes
        + mem.argument_size_in_bytes
        - mem.alias_size_in_bytes
    )
    return float(total_bytes) / (1024**3)


def main():
    base = _finalize([])

    model = ntp.load_model(base, base.model_name)
    scheduler = ntp.load_scheduler(base, base.lr_scheduler_name)
    model = ntp.load_optimizer(base, model, base.optimizer_name, scheduler)

    vocab_size = int(getattr(model.config, "vocab_size"))
    rng = jax.random.key(base.train_seed)

    impls = [
        "reference",
        "tpu_pallas",
    ]
    seq_candidates = [1024, 2048, 4096, 8192, 16384]
    batch_candidates = [1, 2, 4, 8, 16, 32]

    results = {}
    for impl in impls:
        config = _finalize([f"loss_implementation={impl}"])
        print(f"\n== loss_implementation={impl} ==")

        max_seq = 0
        for seq_len in seq_candidates:
            batch = _make_batch(1, seq_len, vocab_size)
            ok, model, mem, compile_s, run_s, err = _try_compile_and_run(
                config, model, batch, rng
            )
            if not ok:
                print(f"B=1 T={seq_len}: FAIL ({type(err).__name__})")
                print(str(err))
                break
            max_seq = seq_len
            print(
                f"B=1 T={seq_len}: OK (compile {compile_s:.1f}s, run {run_s:.1f}s, total_mem={_memory_total_gb(mem):.1f}GB)"
            )
            jax.clear_caches()
            gc.collect()

        probe_seq = max_seq or 2048
        if probe_seq > 2048:
            probe_seq = 2048

        max_batch = 0
        for batch_size in batch_candidates:
            batch = _make_batch(batch_size, probe_seq, vocab_size)
            ok, model, mem, compile_s, run_s, err = _try_compile_and_run(
                config, model, batch, rng
            )
            if not ok:
                print(f"B={batch_size} T={probe_seq}: FAIL ({type(err).__name__})")
                print(str(err))
                break
            max_batch = batch_size
            print(
                f"B={batch_size} T={probe_seq}: OK (compile {compile_s:.1f}s, run {run_s:.1f}s, total_mem={_memory_total_gb(mem):.1f}GB)"
            )
            jax.clear_caches()
            gc.collect()

        results[impl] = {"max_seq_at_b1": max_seq, "max_batch_at_t2048": max_batch}

    print("\n== Summary ==")
    for impl, stats in results.items():
        print(
            f"{impl}: max_seq_at_b1={stats['max_seq_at_b1']}, max_batch_at_t2048={stats['max_batch_at_t2048']}"
        )


if __name__ == "__main__":
    main()
