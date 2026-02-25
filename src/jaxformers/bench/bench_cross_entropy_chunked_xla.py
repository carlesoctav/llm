# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
import time

import jax
import jax.numpy as jnp
import jax.tree as jt

from jaxformers.benchmark_utils import print_compiled_memory_stats
from jaxformers.ops.cross_entropy import BlockSizes, cross_entropy_loss
from jaxformers.ops.cross_entropy.config import infer_block_sizes



def main() -> None:
    print("devices:", jax.devices())

    batch = 32768
    batch = 65536
    embed = 512
    vocab = 128256

    key = jax.random.PRNGKey(0)
    key_x, key_w, key_y = jax.random.split(key, 3)

    x_raw = jax.random.normal(key_x, (batch, embed), dtype=jnp.bfloat16)
    w_raw = jax.random.normal(key_w, (vocab, embed), dtype=jnp.bfloat16)
    y_raw = jax.random.randint(key_y, (batch,), 0, vocab, dtype=jnp.int32)

    implementation = "xla_chunked"
    block_sizes = None  # or: BlockSizes(b=1024, h=512, v=1024)
    if block_sizes is None:
        print(
            "inferred_block_sizes:",
            infer_block_sizes(implementation, batch, embed, vocab, dtype=jnp.float32),
        )

    def compiled_fn(x_in, w_in, y_in):
        return cross_entropy_loss(
            x_in,
            y_in,
            w_in,
            reduction="mean",
            logsumexp_weight=0.0,
            block_sizes=block_sizes,
            dtype=jnp.float32,
            logit_soft_cap=None,
            implementation=implementation,
        )

    fn = jax.value_and_grad(compiled_fn, argnums=(0, 1))
    start = time.perf_counter()
    compiled_fn = jax.jit(fn).lower(x_raw, w_raw, y_raw).compile()
    compile_time = time.perf_counter() - start

    print_compiled_memory_stats(compiled_fn.memory_analysis())

    steps = 3
    start = time.perf_counter()
    for _ in range(steps):
        out = compiled_fn(x_raw, w_raw, y_raw)
        jax.block_until_ready(out)
    steady_time = (time.perf_counter() - start) / steps

    tokens = batch
    print("loss", float(out[0]))
    print("compile_time_s", compile_time)
    print("steady_time_s", steady_time)
    print("tokens_per_s", tokens / steady_time)


if __name__ == "__main__":
    main()
