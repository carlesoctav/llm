import functools
import time
from contextlib import contextmanager


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
    total_gb = output_gb + temp_gb + argument_gb - alias_gb

    print(
        f"Total memory size: {total_gb:.1f} GB, Output size: {output_gb:.1f} GB, Temp size: {temp_gb:.1f} GB, "
        f"Argument size: {argument_gb:.1f} GB, Host temp size: {host_temp_gb:.1f} GB."
    )

    return {
        "total_gb": round(total_gb, 1),
        "output_gb": round(output_gb, 1),
        "temp_gb": round(temp_gb, 1),
        "argument_gb": round(argument_gb, 1),
        "host_temp_gb": round(host_temp_gb, 1),
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
