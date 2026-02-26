# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
import os
import time

import jax
import jax.numpy as jnp

from jaxformers.benchmark_utils import print_compiled_memory_stats
from jaxformers.ops.cross_entropy import cross_entropy_loss
from jaxformers.ops.cross_entropy.config import infer_block_sizes



def main() -> None:
    print("devices:", jax.devices())

    batch = 64
    pos = 2048
    embed = 512
    vocab = 128256

    key = jax.random.PRNGKey(0)
    key_x, key_w, key_y = jax.random.split(key, 3)

    x_raw = jax.random.normal(key_x, (batch * pos, embed), dtype=jnp.bfloat16)
    w_raw = jax.random.normal(key_w, (vocab, embed), dtype=jnp.bfloat16)
    y_raw = jax.random.randint(key_y, (batch * pos,), 0, vocab, dtype=jnp.int32)

    implementation = os.environ.get("CROSS_ENTROPY_IMPL", "xla_chunked")
    block_sizes = None  # or: BlockSizes(b=1024, h=512, v=1024)
    if block_sizes is None:
        print(
            "inferred_block_sizes:",
            infer_block_sizes(implementation, batch * pos, embed, vocab, dtype=jnp.float32),
        )

    def loss_fn(x_in, w_in, y_in):
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

    def grad_fn(x_in, w_in, y_in):
        return jax.grad(loss_fn, argnums=(0, 1))(x_in, w_in, y_in)

    loss_jit = jax.jit(loss_fn)
    grad_jit = jax.jit(grad_fn)

    start = time.perf_counter()
    out = loss_jit(x_raw, w_raw, y_raw)
    out.block_until_ready()
    compile_time = time.perf_counter() - start

    print_compiled_memory_stats(loss_jit.lower(x_raw, w_raw, y_raw).compile().memory_analysis())

    steps = 5
    start = time.perf_counter()
    for _ in range(steps):
        out = loss_jit(x_raw, w_raw, y_raw)
        out.block_until_ready()
    steady_time = (time.perf_counter() - start) / steps

    start = time.perf_counter()
    grad_x, grad_w = grad_jit(x_raw, w_raw, y_raw)
    grad_x.block_until_ready()
    grad_w.block_until_ready()
    bwd_compile_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(steps):
        grad_x, grad_w = grad_jit(x_raw, w_raw, y_raw)
        grad_x.block_until_ready()
        grad_w.block_until_ready()
    bwd_steady_time = (time.perf_counter() - start) / steps

    tokens = batch * pos
    print("loss", float(out))
    print("batch", batch)
    print("pos", pos)
    print("embed", embed)
    print("vocab", vocab)
    print("compile_time_s", compile_time)
    print("steady_time_s", steady_time)
    print("tokens_per_s", tokens / steady_time)
    print("bwd_compile_time_s", bwd_compile_time)
    print("bwd_steady_time_s", bwd_steady_time)
    print("bwd_tokens_per_s", tokens / bwd_steady_time)


if __name__ == "__main__":
    main()
