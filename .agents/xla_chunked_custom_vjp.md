## Goal

Make `xla_chunked` use a Marin-style **streaming custom VJP** for backward (instead of autodiff through nested `lax.fori_loop` + `jax.checkpoint`), then re-benchmark on TPU v4.

Key constraints for this repo:
- Weight layout is `w[V, H]` (vocab-major), not `w[H, V]` like Levanter/Marin.
- Keep public API surface stable: `cross_entropy_loss(..., implementation="xla_chunked")` stays the entrypoint.
- Preserve forward behavior (loss + lse per token), only replace the backward rule.


## Why (Problem Statement)

Current `src/jaxformers/ops/cross_entropy/xla_chunked.py` implements a forward streaming CE by tiling `(B,H,V)` and computes `(loss, lse)`.

Backward today is produced by autodiff through:
- an outer `B` block loop,
- an inner `V` block loop wrapped in `jax.checkpoint`,
- plus an `H` block accumulation loop.

This tends to be:
- slower/less predictable in bwd (extra remat, extra HLO complexity),
- more compile pressure,
- and can materialize larger intermediates than necessary.

Marin’s fix (Levanter) is to keep the same streaming forward, but install a **custom VJP** that computes gradients explicitly by streaming over vocab blocks (and handles `d(loss)` and `d(lse)` cotangents).


## Design

### 1) Keep forward unchanged (or reuse as-is)

Forward returns:
- `loss[B] = lse[B] - label_logit[B]`
- `lse[B] = logsumexp(logits[B, :])`

We keep existing behavior and dtype semantics:
- compute logits in `dtype` (typically fp32),
- optional `logit_soft_cap` via `tanh` scaling,
- mask padded vocab tail with `-inf`.


### 2) Implement a streaming backward (custom VJP)

We install `jax.custom_vjp` on an internal function with positional non-diff args:

`_fused_ce_xla_chunked_custom_vjp(block_sizes, dtype, logit_soft_cap, precision, x, labels, w) -> (loss, lse)`

Residuals returned by fwd rule:
- `x`, `labels`, `w`, `lse` (only what bwd needs)

Cotangents bwd receives:
- `dout_loss[B]` and `dout_lse[B]` (either can be `SymbolicZero`)

Materialize cotangents like Marin:
- If `SymbolicZero`, use `zeros_like(lse)` so we don’t branch on Python.


### 3) Backward math (per `(b_block, v_block)` tile)

For each `b` block `[b0:b0+b_block]` and vocab block `[v0:v0+v_block]`:

1. Recompute logits tile (streaming over `h_block` like forward):
   - `logits[b_block, v_block] = sum_h dot(x_bh[b_block,h_block], w_vh[v_block,h_block])`
   - with `w` stored as `w[V,H]` so the dot contraction is `x.H` with `w.H`:
     - `dot_general(x_bh, w_vh, (((1,), (1,)), ((), ()))) -> (b_block, v_block)`

2. Apply `logit_soft_cap`:
   - `logits = tanh(logits / cap) * cap`
   - store `cap_deriv = 1 - tanh^2(...)` for chain rule.

3. Mask padded vocab tail:
   - `valid = (v0 + arange(v_block)) < V`
   - `logits = where(valid, logits, -inf)`

4. Compute probabilities using saved `lse`:
   - `probs = exp(logits - lse_block[:, None])`

5. Combine cotangents:
   - `delta = (dout_loss + dout_lse)[:, None] * probs`
   - if the gold label is inside this vocab block:
     - `delta[row, label - v0] -= dout_loss[row]`
   - `delta *= cap_deriv`

6. Accumulate gradients:
   - `dx_block += delta @ w_block` (contract vocab dimension):
     - with `w_block` layout `(v_block, H)` the contraction is `delta.v` with `w_block.v`:
       - `dot_general(delta, w_vh, (((1,), (0,)), ((), ()))) -> (b_block, h_block)` (per `h_block`)
   - `dw_block += delta^T @ x_block`:
     - for `w[V,H]` we want `(v_block, h_block)` per `h_block`:
       - `dot_general(delta, x_bh, (((0,), (0,)), ((), ()))) -> (v_block, h_block)`

We accumulate `dw` in fp32 and cast back to `w.dtype` at the end (matches the existing “compute in fp32, return bf16” style used elsewhere).


## Implementation Steps

1. Edit `src/jaxformers/ops/cross_entropy/xla_chunked.py`
   - Factor current forward body into a private function (no custom_vjp).
   - Add `_materialize_cotangent` helper (copy from Marin).
   - Add `_fused_cross_entropy_chunked_xla_bwd(...)` implementing the streaming bwd above for `w[V,H]`.
   - Wrap with `jax.custom_vjp` and keep exported `fused_cross_entropy_chunked_xla` name unchanged.

2. Add a correctness test:
   - New test file: `tests/test_cross_entropy_xla_chunked_custom_vjp.py`
   - Compare `(loss,lse)` and `(dx,dw)` against `cross_entropy_reference` on a small shape that satisfies block constraints:
     - e.g. `B=256,H=128,V=512` with `BlockSizes(b=128,h=128,v=256)`
   - Use objective with nonzero `d(lse)`:
     - `obj = mean(loss) + 1e-4 * mean(lse**2)`

3. Re-benchmark on TPU v4:
   - `source .venv/bin/activate`
   - `CROSS_ENTROPY_IMPL=xla_chunked python src/jaxformers/bench/bench_cross_entropy_chunked_xla.py`
   - Compare `bwd_tokens_per_s` to the previous baseline.


## Expected Outcome

- Correctness: gradients match reference within bf16/streaming tolerances.
- Performance: backward for `xla_chunked` improves vs autodiff, especially on TPU v4, because we avoid backpropagating through nested streaming loops.

