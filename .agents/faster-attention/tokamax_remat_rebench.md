# Tokamax `xla_chunked` re-bench with `remat` on q-loop (TPU v4)

Date: 2026-03-05

## Patch

Local (venv) change to Tokamax `xla_chunked`:
- Added `jax.remat` around the **q-chunk scan body** in
  `.venv/lib/python3.11/site-packages/tokamax/_src/ops/attention/xla_chunked.py`
  (implemented via a `make_q_loop_fn(q_chunk_size)` factory so `q_chunk_size` stays static).

## Environment

- Hardware: TPU v4, 4 devices (single process)
- `jax==0.9.0.1`, `jaxlib==0.9.0.1`, `tokamax==0.0.10`
- Bench: `src/jaxformers/bench/bench_attention_tokamax.py`

## Common setup (unless noted)

- Shapes: `B=4`, `T=S`, `N=4`, `K=1`, `H=256`, `layers=1`
- Mask: `ATTENTION_MASK=causal` (i.e. `mask=None`, `is_causal=True`)
- Sharding: `USE_Q_SHARDING=0`
- Steps: `STEPS=3` for `T<=8192`, `STEPS=1` for `T>=16384`
- Throughput: `tok/s = (B*T) / steady_time_s`

---

## `TOKAMAX_ATTENTION_IMPL=xla_chunked` with `TOKAMAX_XLA_CHUNK_SIZE=512,1024`

### `T=4096` (`STEPS=3`)
- fwd: compile `0.7674s`; mem total `0.1GB` (temp `0.0GB`, args `0.0GB`); steady `0.002729s` → `6.00M tok/s`
- bwd: compile `2.7384s`; mem total `0.2GB` (temp `0.1GB`, args `0.0GB`); steady `0.012755s` → `1.28M tok/s`

### `T=8192` (`STEPS=3`)
- fwd: compile `0.8370s`; mem total `0.2GB` (temp `0.1GB`, args `0.1GB`); steady `0.009987s` → `3.28M tok/s`
- bwd: compile `3.4286s`; mem total `0.4GB` (temp `0.2GB`, args `0.1GB`); steady `0.050011s` → `0.655M tok/s`

### `T=16384` (`STEPS=1`)
- fwd: compile `0.9205s`; mem total `0.7GB` (temp `0.4GB`, args `0.2GB`); steady `0.038455s` → `1.70M tok/s`
- bwd: compile `3.5144s`; mem total `0.9GB` (temp `0.5GB`, args `0.2GB`); steady `0.198292s` → `0.331M tok/s`

### `T=32768` (`STEPS=1`)
- fwd: compile `0.9818s`; mem total `1.9GB` (temp `1.3GB`, args `0.4GB`); steady `0.153080s` → `0.856M tok/s`
- bwd: compile `4.1857s`; mem total `2.4GB` (temp `1.7GB`, args `0.4GB`); steady `0.792016s` → `0.165M tok/s`

---

## `TOKAMAX_ATTENTION_IMPL=xla_chunked` with `TOKAMAX_XLA_CHUNK_SIZE=1024,2048`

### `T=4096` (`STEPS=3`)
- fwd: compile `1.9708s`; mem total `0.2GB` (temp `0.1GB`, args `0.0GB`); steady `0.002682s` → `6.11M tok/s`
- bwd: compile `3.1516s`; mem total `0.4GB` (temp `0.3GB`, args `0.0GB`); steady `0.011436s` → `1.43M tok/s`

### `T=8192` (`STEPS=3`)
- fwd: compile `2.1016s`; mem total `0.3GB` (temp `0.1GB`, args `0.1GB`); steady `0.009059s` → `3.62M tok/s`
- bwd: compile `3.0776s`; mem total `0.7GB` (temp `0.5GB`, args `0.1GB`); steady `0.043726s` → `0.749M tok/s`

### `T=16384` (`STEPS=1`)
- fwd: compile `2.0587s`; mem total `0.8GB` (temp `0.5GB`, args `0.2GB`); steady `0.034656s` → `1.89M tok/s`
- bwd: compile `2.7957s`; mem total `1.3GB` (temp `0.9GB`, args `0.2GB`); steady `0.170510s` → `0.384M tok/s`

### `T=32768` (`STEPS=1`)
- fwd: compile `2.3957s`; mem total `2.0GB` (temp `1.4GB`, args `0.4GB`); steady `0.135336s` → `0.968M tok/s`
- bwd: compile `3.8366s`; mem total `2.7GB` (temp `2.0GB`, args `0.4GB`); steady `0.676411s` → `0.194M tok/s`

