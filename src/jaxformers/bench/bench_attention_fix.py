import time

import jax
import jax.numpy as jnp
import sws
from tabulate import tabulate

from jaxformers.attention_utils import ATTENTION_INTERFACE
from jaxformers.benchmark_utils import print_compiled_memory_stats
from jaxformers.distributed.parallel import (
    from_logical_rules,
    make_logical_axis_rules,
    make_mesh,
    with_logical_axis,
)
from jaxformers.masking_utils import make_causal_mask


@jax.custom_vjp
def optimization_barrier(x):
    return jax.lax.optimization_barrier(x)


def _optimization_barrier_fwd(x):
    return optimization_barrier(x), None


def _optimization_barrier_bwd(_, dout):
    return (dout,)


optimization_barrier.defvjp(_optimization_barrier_fwd, _optimization_barrier_bwd)


def get_config():
    config = sws.Config()
    config.step = 10
    config.mode = "bwd_vjp"
    config.parallel.parallel_dims = {"dp_replicate": 1, "dp_shard": 4, "cp": 1, "tp": 1}
    config.num_devices = 4
    config.impls = list(ATTENTION_INTERFACE.keys())
    config.devices = lambda: jax.devices()[:config.num_devices]
    config.b = 4
    config.t = 8192
    config.n = 4
    config.k = 1
    config.h = 512
    config.mask_mode = "causal"
    config.cotangent = "output"
    config.use_q_sharding = False
    return config


def make_compiled_bwd(impl, b, t, mask_mode, cotangent, use_q_sharding):
    fn = ATTENTION_INTERFACE[impl]

    if mask_mode not in ("bool", "causal", "none"):
        raise ValueError(f"Unsupported mask_mode: {mask_mode}")

    if cotangent not in ("output", "ones"):
        raise ValueError(f"Unsupported cotangent: {cotangent}")

    def attention_fn(q, k, v):
        q_sharding = jax.typeof(q).sharding if use_q_sharding else None
        mask = None
        is_causal = False
        if mask_mode == "bool":
            mask = make_causal_mask(impl, q.reshape(b, t, -1))
        elif mask_mode == "causal":
            is_causal = True
        return fn(
            q,
            k,
            v,
            mask=mask,
            is_causal=is_causal,
            precision=jax.lax.Precision.HIGHEST,
            q_sharding=q_sharding,
        )

    def bwd_fn(q, k, v):
        out, pullback = jax.vjp(
            lambda q_in, k_in, v_in: optimization_barrier(attention_fn(q_in, k_in, v_in)),
            q,
            k,
            v,
        )
        if cotangent == "output":
            dout = out
        else:
            dout = jnp.ones_like(out)
        return pullback(dout)

    return jax.jit(bwd_fn)


def main(config: sws.FinalConfig):
    key = jax.random.key(10)
    mesh = make_mesh(config.parallel.parallel_dims.to_dict(), config.devices)
    print("DEBUGPRINT {mesh}:", mesh)
    rule = make_logical_axis_rules(
        config.parallel.parallel_dims, sequence_parallelism=False
    )
    stats = {}
    print(config.devices)
    print(config.b, config.t, config.n, config.k, config.h)
    print("mask_mode", config.mask_mode)
    print("cotangent", config.cotangent)
    print("use_q_sharding", config.use_q_sharding)
    with jax.set_mesh(mesh), with_logical_axis(rule):
        for impl in config.impls:
            print(f"==========={impl}===========")
            total_token = config.b * config.t
            stats[impl] = {}
            q = jax.random.normal(
                key,
                (config.b, config.t, config.n, config.h),
                out_sharding=from_logical_rules(("batch", "sequence", "model", None)),
                dtype=jnp.bfloat16,
            )
            k = jax.random.normal(
                key,
                (config.b, config.t, config.k, config.h),
                out_sharding=from_logical_rules(("batch", "sequence", "model", None)),
                dtype=jnp.bfloat16,
            )
            v = jax.random.normal(
                key,
                (config.b, config.t, config.k, config.h),
                out_sharding=from_logical_rules(("batch", "sequence", "model", None)),
                dtype=jnp.bfloat16,
            )
            array_q_sharding = jax.typeof(q).sharding
            passed_q_sharding = array_q_sharding if config.use_q_sharding else None
            print("DEBUGPRINT {array_q_sharding}:", array_q_sharding)
            print("DEBUGPRINT {passed_q_sharding}:", passed_q_sharding)
            bwd_fn = make_compiled_bwd(
                impl,
                config.b,
                config.t,
                config.mask_mode,
                config.cotangent,
                config.use_q_sharding,
            )
            t0 = time.monotonic()
            compile_fn = bwd_fn.lower(q, k, v).compile()
            stats[impl]["compile_time"] = time.monotonic() - t0
            stats[impl].update(
                **print_compiled_memory_stats(compile_fn.memory_analysis())
            )
            for _ in range(config.step):
                t0 = time.monotonic()
                grads = compile_fn(q, k, v)
                jax.block_until_ready(grads)
                diff = time.monotonic() - t0
                stats[impl].setdefault("time", []).append(diff)
                stats[impl].setdefault("tok/s", []).append(total_token / diff)

            stats[impl]["mean_time"] = sum(stats[impl]["time"]) / config.step
            stats[impl]["mean_tok/s"] = sum(stats[impl]["tok/s"]) / config.step
        rows = [
            {
                "impl": impl,
                "compile_s": impl_stats["compile_time"],
                "step_s": impl_stats["mean_time"],
                "tok/s": impl_stats["mean_tok/s"],
                "total_gb": impl_stats["total_gb"],
                "output_gb": impl_stats["output_gb"],
                "temp_gb": impl_stats["temp_gb"],
                "argument_gb": impl_stats["argument_gb"],
                "host_temp_gb": impl_stats["host_temp_gb"],
                "alias_gb": impl_stats["alias_gb"],
            }
            for impl, impl_stats in stats.items()
        ]
        print(tabulate(rows, headers="keys", tablefmt="github", floatfmt=".4f"))


if __name__ == "__main__":
    sws.run(main)
