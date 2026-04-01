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


def get_config():
    config = sws.Config()
    config.step = 10
    config.mode = "bwd"
    config.parallel.parallel_dims = {"dp_replicate": 1, "dp_shard": 4, "cp": 1, "tp": 1}
    config.num_devices = 4
    config.impls = list(ATTENTION_INTERFACE.keys())
    config.devices = lambda: jax.devices()[:config.num_devices]
    config.b = 4
    config.t = 8192
    config.n = 4
    config.k = 1
    config.h = 512
    return config


def make_compiled_bwd(impl, b, t, n, h):
    fn = ATTENTION_INTERFACE[impl]
    def loss_fn(q, k, v):
        # random_out = jax.random.normal(jax.random.key(100), (b, t, n, h), dtype = jnp.bfloat16)
        q_sharding = jax.typeof(q).sharding
        mask = make_causal_mask(impl, q.reshape(b, t, -1))
        out = fn(q, k, v, q_sharding = q_sharding, mask = mask)
        return jnp.sum(out, dtype = jnp.float32)

    grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1, 2)))
    return grad_fn


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
    with jax.set_mesh(mesh), with_logical_axis(rule):
        for impl in config.impls:
            print(f"==========={impl}===========")
            total_token = config.b * config.t
            stats[impl] = {}
            q = jax.random.normal(
                key,
                (config.b, config.t, config.n, config.h),
                out_sharding=from_logical_rules(("batch", "sequence", "model", None)),
                dtype = jnp.bfloat16
            )
            k = jax.random.normal(
                key,
                (config.b, config.t, config.k, config.h),
                out_sharding=from_logical_rules(("batch", "sequence", "model", None)),
                dtype = jnp.bfloat16
            )
            v = jax.random.normal(
                key,
                (config.b, config.t, config.k, config.h),
                out_sharding=from_logical_rules(("batch", "sequence", "model", None)),
                dtype = jnp.bfloat16
            )
            q_sharding = jax.typeof(q).sharding
            print("DEBUGPRINT {q_sharding}:", q_sharding)
            grad_fn = make_compiled_bwd(impl, config.b, config.t, config.n, config.h)
            t0 = time.monotonic()
            compile_fn = jax.jit(grad_fn).lower(q, k, v).compile()
            stats[impl]["compile_time"] = time.monotonic() - t0
            stats[impl].update(
                **print_compiled_memory_stats(compile_fn.memory_analysis())
            )
            for _ in range(config.step):
                t0 = time.monotonic()
                grad = compile_fn(q, k, v)
                jax.block_until_ready(grad)
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
