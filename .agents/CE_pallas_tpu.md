# Plan: Integrate TPU Pallas Fused Cross-Entropy (from /mnt/carles/marin) into /mnt/carles/llm

Goal: add a `pallas_tpu` backend for `src/jaxformers/ops/cross_entropy/cross_entropy_loss()` by porting Marin’s fused CE Pallas Mosaic TPU kernel, with **minimal edits to** [api.py](/mnt/carles/llm/src/jaxformers/ops/cross_entropy/api.py). This repo uses `w` shaped **[V, H]** (vocab, embed), while Marin’s kernel assumes **[H, V]**; the port must adapt the kernel math and BlockSpecs accordingly (avoid materializing `w.T`).

## 0) Current State (in this repo)

- Public API: `cross_entropy_loss(x[B,H], labels[B], w[V,H], ...) -> loss` in [api.py](/mnt/carles/llm/src/jaxformers/ops/cross_entropy/api.py).
- Implementations today:
  - `xla_chunked`: streaming CE that already expects `w[V,H]` ([xla_chunked.py](/mnt/carles/llm/src/jaxformers/ops/cross_entropy/xla_chunked.py))
  - `reference`: optax CE on explicit logits (`einsum("bh,vh->bv")`) ([reference.py](/mnt/carles/llm/src/jaxformers/ops/cross_entropy/reference.py))
- Block sizes type: `BlockSizes(v, h, b)` in [config.py](/mnt/carles/llm/src/jaxformers/ops/cross_entropy/config.py).
- There is commented scaffolding for `.pallas_tpu` import in `api.py` already.

## 1) Files to Add / Modify

Add:
- `src/jaxformers/ops/cross_entropy/pallas_tpu.py`
  - Copy from Marin: `/mnt/carles/marin/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_tpu.py`
  - Then adapt to this repo’s API + `w[V,H]`.

Modify (small, targeted):
- [api.py](/mnt/carles/llm/src/jaxformers/ops/cross_entropy/api.py)
  - Add `"pallas_tpu"` to `Implementation` and `IMPLEMENTATIONS`.
  - Preferably try Pallas first on TPU (then fall back to `"xla_chunked"`, `"reference"`).
  - Keep changes minimal: just import/registration and maybe a one-time fallback warning.
- [config.py](/mnt/carles/llm/src/jaxformers/ops/cross_entropy/config.py)
  - Add `infer_block_sizes("pallas_tpu", ...)` with a TPU-v4-safe default (copy Marin’s tuned logic or a simplified subset).

Optional but recommended (for reproducible correctness checks):
- `tests/test_cross_entropy_pallas_tpu.py` (mark `@pytest.mark.require_tpu`)

## 2) Kernel Port Strategy (Key: `w` is [V,H] here)

Marin kernel expects:
- `x: [B,H]`
- `w: [H,V]`
- logits: `x @ w`  -> `[B,V]`
- w-grad: `x.T @ delta` -> `[H,V]`

This repo uses:
- `w: [V,H]`
- logits: `x @ w.T` -> `[B,V]`
- w-grad: `delta.T @ x` -> `[V,H]`

So the kernel must treat the weight tile as `(v_block, h_block)` (not `(h_block, v_block)`), and swap dot_general contraction dims accordingly.

### 2.1 Forward kernel changes

Where Marin does (conceptually):
- Load `w_ref` tile shaped `(h_block, v_block)`, indexed by `(h_index, v_index)`.
- `xw += dot_general(x_ref[B,H], w_ref[H,V], contract x.H with w.H)` (dims `((1,), (0,))`).

Change to `w[V,H]`:
- Load `w_ref` tile shaped `(v_block, h_block)`, indexed by `(v_index, h_index)`.
- `xw += dot_general(x_ref[B,H], w_ref[V,H], contract x.H with w.H)` (dims `((1,), (1,))`).

Pseudocode replacement:
```python
# old (H,V)
dot_general(x_ref, w_ref, (((1,), (0,)), ((), ())))

# new (V,H)
dot_general(x_ref, w_ref, (((1,), (1,)), ((), ())))
```

Padding tail vocab tile:
- Old: `w_ref[:, rem:] = 0` (because vocab is axis=1)
- New: `w_ref[rem:, :] = 0` (because vocab is axis=0)

BlockSpec for `w` in forward `pallas_call`:
- Old:
  - block shape `(h_block, v_block)`
  - start `(h_index, v_index)` expressed as `(k, j)`
- New:
  - block shape `(v_block, h_block)`
  - start `(v_index, h_index)` expressed as `(j, k)`

### 2.2 Backward kernel changes

Stage 0 (recompute logits tile):
- Old: `dot_general(x_ref[B,H], w_ref[H,V], dims ((1),(0))) -> [B,V]`
- New: `dot_general(x_ref[B,H], w_ref[V,H], dims ((1),(1))) -> [B,V]`

Compute `x_grad`:
- Derivation for this repo: `x_grad[b,h] = sum_v delta[b,v] * w[v,h]` so `x_grad = delta @ w`.
- Old (Marin): `dot_general(delta[B,V], w_ref[H,V], contract V with V) -> [B,H]` using dims `((1),(1))`
- New: `dot_general(delta[B,V], w_ref[V,H], contract V with V-axis0) -> [B,H]` using dims `((1),(0))`

Pseudocode replacement:
```python
# old (H,V): delta @ w.T
dot_general(delta, w_ref, (((1,), (1,)), ((), ())))

# new (V,H): delta @ w
dot_general(delta, w_ref, (((1,), (0,)), ((), ())))
```

Compute `w_grad`:
- Derivation for this repo: `w_grad[v,h] = sum_b delta[b,v] * x[b,h]` so `w_grad = delta.T @ x`.
- Old (Marin): `w_grad = dot_general(x_ref[B,H], delta[B,V], contract B) -> [H,V]`
- New: `w_grad = dot_general(delta[B,V], x_ref[B,H], contract B) -> [V,H]`

Pseudocode replacement:
```python
# old (H,V)
dot_general(x_ref, delta, (((0,), (0,)), ((), ())))

# new (V,H)
dot_general(delta, x_ref, (((0,), (0,)), ((), ())))
```

HBM accumulation layout for `w_grad_partial`:
- Old: `(num_cores, H, V)`; vocab dim was last axis.
- New: `(num_cores, V, H)`; vocab dim is axis=1.

So inside the backward kernel:
- `v_dim` must come from `w_grad_hbm_ref.shape[1]` (not `shape[-1]`).
- `w_grad_slice` indexing order swaps:
  - Old: `[core, h_slice, v_slice]`
  - New: `[core, v_slice, h_slice]`
- `w_grad_tile_ref` scratch should be `(v_block, h_block)` instead of `(h_block, v_block)`.
- Tail-vocab handling (`cur_v_block_size`) applies along axis=0 of `w_grad_tile_ref`.

### 2.3 Keep/Drop Non-essentials

To minimize new dependencies, drop Marin-only extras unless needed:
- `with_io_bytes_accessed` / cost estimation: safe to remove initially (compile/execute should still work).
- `return_argmax`: not used by this repo’s CE API; ignore.

Keep:
- `PallasUnsupportedError` and strict runtime validation (backend must be TPU, block sizes multiples of 128, divisibility checks, TPU v4 label-layout constraint for `b_block` when `B>=1024`).
- `custom_vjp` wrapper. Important because `api.py` expects `(loss, lse)` and may apply an LSE penalty, which produces a non-zero cotangent for `lse`.

## 3) Block Size Inference (TPU v4)

Do not reuse `infer_xla_chunked_block_size` for Pallas: it will pick `v=8192` for large vocab which is likely too big for VMEM scratch.

Preferred: port Marin’s inference table logic (from `/mnt/carles/marin/.../tuned_block_sizes.py`) but output this repo’s `BlockSizes(v=?, h=?, b=?)`.

Minimal acceptable initial heuristic for TPU v4:
- `b=1024`, `h=512`, `v=1024` (all multiples of 128; matches Marin TPU v4 table for llama-ish/large-vocab).
- If `B % 1024 != 0` or `H % 512 != 0`, either:
  - sanitize to the largest multiple-of-128 divisor (Marin’s approach), or
  - raise `PallasUnsupportedError` and let `api.py` fall back to XLA.

Implementation hook: in [config.py](/mnt/carles/llm/src/jaxformers/ops/cross_entropy/config.py)
- add:
  - `elif impl == "pallas_tpu": return infer_pallas_tpu_block_sizes(...)`

## 4) Minimal Change to api.py

In [api.py](/mnt/carles/llm/src/jaxformers/ops/cross_entropy/api.py):
- Extend `Implementation` to include `"pallas_tpu"`.
- Add a try-import block (the commented section is already there):
  - `from .pallas_tpu import PallasUnsupportedError, linear_softmax_cross_entropy_loss_pallas`
  - register: `IMPLEMENTATIONS["pallas_tpu"] = linear_softmax_cross_entropy_loss_pallas`
  - update `_DEFAULT_IMPLEMENTATION` to prefer pallas on TPU:
    - simplest: always prepend `"pallas_tpu"` (non-TPU will just throw + fall back)
    - nicer (still small): only prepend if `jax.default_backend() == "tpu"`
- Keep the function signature unchanged.

Note: current `cross_entropy_loss()` already catches `Exception` and continues to the next implementation, so Pallas can fail safely (no extra control flow required).

## 5) Correctness Verification (Fwd + Bwd)

### 5.1 What to compare

Compare Pallas vs `reference` (gold) and optionally vs `xla_chunked`:
- Forward:
  - per-example `loss[B]`
  - `lse[B]` (Pallas returns it; `reference` can compute it)
- Backward:
  - `grad_x` shape `[B,H]`
  - `grad_w` shape `[V,H]`

Also test with and without:
- `logit_soft_cap` (e.g. 30.0)
- `logsumexp_weight` (e.g. 1e-4) to ensure non-zero cotangent for `lse` works
- dtype paths: run at least:
  - inputs `x,w` in `bfloat16`
  - compute `dtype=jnp.float32` (matches bench)

### 5.2 Suggested test harness

Add `tests/test_cross_entropy_pallas_tpu.py`:
- Mark with `@pytest.mark.require_tpu`.
- Generate small-ish shapes that satisfy block constraints, e.g.:
  - `B=1024`, `H=512`, `V=4096` and also `V=128256` if feasible.
  - For quick compile/test, start with `V=4096` (fewer v-blocks).
- Use:
  - `loss_ref, lse_ref = cross_entropy_reference(x, labels, w, dtype=..., logit_soft_cap=..., precision=...)`
  - `loss_pal, lse_pal = linear_softmax_cross_entropy_loss_pallas(x, labels, w, block_sizes=..., ...)`

Numeric checks:
- `jnp.max(jnp.abs(loss_pal - loss_ref))` within tolerance.
- Use relative error for `lse` if magnitudes large.

Gradient checks:
- Define a scalar objective that exercises both outputs:
```python
def obj(x, w):
    loss, lse = impl(x, labels, w, ...)
    return jnp.sum(loss) + alpha * jnp.sum(lse**2)
```
- Compare `jax.grad(obj, argnums=(0,1))` between pallas impl and reference impl.
- Tolerances:
  - start loose for bf16 inputs (e.g. `atol=1e-2`, `rtol=1e-2`), tighten if stable.

If you don’t want to add pytest initially: create `src/jaxformers/bench/verify_cross_entropy_pallas_tpu.py` and run it manually on TPU.

## 6) Backward Speed Benchmark

Use existing script: [bench_cross_entropy_chunked_xla.py](/mnt/carles/llm/src/jaxformers/bench/bench_cross_entropy_chunked_xla.py)
- Set `implementation = "pallas_tpu"` (or add a tiny CLI flag/env var to select impl).
- Keep the same shape:
  - `batch=65536`, `embed=512`, `vocab=128256`
  - inputs `bfloat16`, compute `dtype=float32`

What to record:
- `compile_time_s`
- `steady_time_s` (this includes forward + backward because the script uses `value_and_grad`)
- `tokens_per_s`

Sanity checks before timing:
- Ensure Pallas is actually used:
  - temporarily force `implementation=("pallas_tpu",)` and fail if fallback happens, or
  - print which impl succeeded (minimal instrumentation) during development.

## 7) Debugging / Gotchas

- If Pallas fails and you silently fall back, the benchmark may look “fine” but not be measuring Pallas. During bring-up, temporarily:
  - pass `implementation=("pallas_tpu",)` and let it error, or
  - add a one-time warning on Pallas fallback (Marin has `_warn_pallas_fallback_once`).
- Block sizes:
  - must be multiples of 128
  - must divide B/H
  - TPU v4: when `B>=1024`, require `b_block % 1024 == 0`
- `w` layout: verify every dot_general and every block slice matches the intended axes; any single wrong axis will “work” but produce incorrect grads.

## 8) Concrete Execution Checklist

1. Add `pallas_tpu.py` (copy from Marin; remove cost-estimate helpers; adapt to `w[V,H]`).
2. Extend `infer_block_sizes()` to return sane Pallas block sizes (TPU v4 defaults).
3. Minimal edits to `api.py` to register `"pallas_tpu"` and prefer it on TPU.
4. Run correctness:
   - run new pytest or the verify script on TPU v4
   - compare forward + backward vs reference
5. Run benchmark:
   - update `implementation` in the bench script to `"pallas_tpu"`
   - run `python -m src.jaxformers.bench.bench_cross_entropy_chunked_xla` (or equivalent) on TPU v4
   - record compile + steady times

