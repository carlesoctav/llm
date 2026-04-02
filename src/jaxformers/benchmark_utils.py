import functools
import time

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

import jaxformers.tree_util


def print_compiled_memory_stats(compiled_stats):
    if compiled_stats is None:
        return

    def bytes_to_gb(num_bytes):
        return num_bytes / (1024**3)

    output_gb = bytes_to_gb(compiled_stats.output_size_in_bytes)
    temp_gb = bytes_to_gb(compiled_stats.temp_size_in_bytes)
    argument_gb = bytes_to_gb(compiled_stats.argument_size_in_bytes)
    alias_gb = bytes_to_gb(compiled_stats.alias_size_in_bytes)
    host_temp_gb = bytes_to_gb(compiled_stats.host_temp_size_in_bytes)
    peak_gb = bytes_to_gb(compiled_stats.peak_memory_in_bytes)
    total_gb = output_gb + temp_gb + argument_gb - alias_gb

    print(
        f"Total memory size: {total_gb:.1f} GB, Output size: {output_gb:.1f} GB, Temp size: {temp_gb:.1f} GB, "
        f"Argument size: {argument_gb:.1f} GB, Host temp size: {host_temp_gb:.1f} GB, Peak size: {peak_gb:.1f} GB.",
        f"Alias size: {alias_gb:.1f} GB",
    )

    return {
        "total_gb": round(total_gb, 1),
        "output_gb": round(output_gb, 1),
        "temp_gb": round(temp_gb, 1),
        "argument_gb": round(argument_gb, 1),
        "host_temp_gb": round(host_temp_gb, 1),
        "alias_gb": round(alias_gb, 1),
        "peak_gb": round(peak_gb, 1),
    }


def print_flops(compiled_stats):
    tflops = compiled_stats.get("flops") / 1e12
    print(f"estimated tflops per step: {tflops}")
    return {"tflops": tflops}


def print_timing(wrapped, name: str | None = None):
    name = name or wrapped.__name__

    @functools.wraps(wrapped)
    def wrapper(*args, **kwargs):
        t0 = time.monotonic()
        out = wrapped(*args, **kwargs)
        print(f"{name} run in {time.monotonic() - t0:.2f} seconds")
        return out

    return wrapper


def print_train_state_size(train_state):
    def dtype_multiplier(dtype):
        if dtype in (jnp.float32, jnp.int32):
            return 4
        elif dtype in (jnp.bfloat16, jnp.float16):
            return 2
        else:
            raise ValueError

    def sum(name, tree):
        p = 0

        def _sum(path, leaf):
            nonlocal p
            if isinstance(leaf, jax.Array):
                p += np.prod(leaf.shape) * dtype_multiplier(leaf.dtype)

        jtu.tree_map_with_path(_sum, tree)
        print(f"tree {name} use {p / 1e9} GB")
        return p

    train_model, freeze_model= jaxformers.tree_util.partition(
        train_state.model, train_state.train_mask
    )

    p = 0
    p += sum("train_model", train_model)
    p += sum("freeze_model", freeze_model)
    p += sum("opt_state", train_state.opt_state)
    print(f"Total Model use {p / 1e9} GB")
