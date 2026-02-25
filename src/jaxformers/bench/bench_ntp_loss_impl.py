import argparse
import json
import os
import runpy
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import sws

from jaxformers.benchmark_utils import print_compiled_memory_stats
from jaxformers.ops.cross_entropy.config import infer_block_sizes
from jaxformers.train.ntp import (
    _preparse_absl_flags,
    load_dataset,
    load_model,
    load_optimizer,
    load_scheduler,
    train_step,
)


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        s in msg
        for s in (
            "out of memory",
            "oom",
            "resourceexhausted",
            "resource exhausted",
            "memory exhausted",
            "hbm",
        )
    )


def _load_final_config(config_path: str, overrides: list[str]) -> sws.FinalConfig:
    config_path = os.path.abspath(config_path)
    factory = runpy.run_path(config_path).get("get_config")
    if not callable(factory):
        raise AttributeError(f"Function 'get_config' not found in {config_path}")
    builder = factory()
    if not isinstance(builder, sws.Config):
        raise TypeError("Config factory must return a sws.Config")
    final, unused = builder.finalize(overrides, return_unused_argv=True)
    if unused:
        raise ValueError(f"Unused extra arguments: {unused}")
    return final


def main(argv: list[str]) -> int:
    _preparse_absl_flags()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
        help="Path to a sws get_config() python file.",
    )
    parser.add_argument(
        "--loss-impl", choices=("xla_chunked", "reference"), required=True
    )
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--timed-steps", type=int, default=30)
    parser.add_argument("--exp-name", default=None)
    args = parser.parse_args(argv)

    result: dict[str, Any] = {
        "status": "error",
        "config_path": os.path.abspath(args.config),
        "loss_impl": args.loss_impl,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "exp_name": args.exp_name,
    }

    try:
        overrides = [
            f"data.transforms.max_length={args.max_length}",
            f"loss_implementation={args.loss_impl}",
        ]
        if args.batch_size is not None:
            overrides.append(f"train_loader.global_batch_size={args.batch_size}")
        if args.exp_name is not None:
            overrides.append(f"exp_name={args.exp_name}")

        config = _load_final_config(args.config, overrides)

        rngs = jax.random.key(config.train_seed) if config.train_seed else None

        model = load_model(config, config.model_name)
        scheduler = load_scheduler(config, config.lr_scheduler_name)
        model = load_optimizer(config, model, "sgd", scheduler)

        train_ds, _ = load_dataset(config, config.data_name)
        batch = next(iter(train_ds))

        w = model.weights[model.lm_head_key]
        if args.loss_impl == "xla_chunked":
            inferred = infer_block_sizes(
                "xla_chunked",
                int(batch["labels"].size),
                int(w.shape[1]),
                int(w.shape[0]),
                dtype=jnp.float32,
            )
            result["block_sizes"] = {"b": inferred.b, "h": inferred.h, "v": inferred.v}
        else:
            result["block_sizes"] = None
        result["optimizer"] = "sgd"

        start = time.perf_counter()
        compiled = (
            jax.jit(lambda m, b, r: train_step(config, m, b, r))
            .lower(
                model, batch, jax.random.fold_in(rngs, 0) if rngs is not None else None
            )
            .compile()
        )
        compile_time_s = time.perf_counter() - start

        memory_stats = print_compiled_memory_stats(compiled.memory_analysis())

        tokens_per_step = None
        times: list[float] = []
        total_steps = args.warmup_steps + args.timed_steps
        for i in range(total_steps):
            step_rng = jax.random.fold_in(rngs, i) if rngs is not None else None
            t0 = time.perf_counter()
            model, aux = compiled(model, batch, step_rng)
            jax.block_until_ready(aux["loss"][0])
            dt = time.perf_counter() - t0
            if i >= args.warmup_steps:
                times.append(dt)
            if tokens_per_step is None:
                try:
                    tokens_per_step = int(jax.device_get(aux["token_count"]).item())
                except Exception:
                    tokens_per_step = None

        step_time_s = sum(times) / len(times) if times else None

        result.update(
            {
                "status": "ok",
                "device_backend": jax.default_backend(),
                "device_count": jax.device_count(),
                "model_id": getattr(config.model, "model_id", None),
                "packing": getattr(config.data.transforms, "packing", None),
                "global_batch_size": getattr(
                    config.train_loader, "global_batch_size", None
                ),
                "compile_time_s": compile_time_s,
                "step_time_s": step_time_s,
                "tokens_per_step": tokens_per_step,
                "tokens_per_s": (
                    (tokens_per_step / step_time_s)
                    if (tokens_per_step is not None and step_time_s)
                    else None
                ),
                "memory": memory_stats,
            }
        )

    except BaseException as e:
        result["status"] = "oom" if _is_oom(e) else "error"
        result["error"] = f"{type(e).__name__}: {e}"

    print("BENCH_RESULT", json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
