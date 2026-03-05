# Tokamax `xla_chunked` bench (TPU v4)

Date: 2026-03-05

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
- fwd: compile `0.7806s`; mem total `0.1GB` (temp `0.0GB`, args `0.0GB`); steady `0.002762s` → `5.93M tok/s`
- bwd: compile `3.2873s`; mem total `0.5GB` (temp `0.4GB`, args `0.0GB`); steady `0.013713s` → `1.19M tok/s`

### `T=8192` (`STEPS=3`)
- fwd: compile `0.8854s`; mem total `0.2GB` (temp `0.1GB`, args `0.1GB`); steady `0.010009s` → `3.27M tok/s`
- bwd: compile `3.6235s`; mem total `1.6GB` (temp `1.4GB`, args `0.1GB`); steady `0.052718s` → `0.622M tok/s`

### `T=16384` (`STEPS=1`)
- fwd: compile `0.8087s`; mem total `0.7GB` (temp `0.4GB`, args `0.2GB`); steady `0.038508s` → `1.70M tok/s`
- bwd: compile `3.3628s`; mem total `5.4GB` (temp `5.0GB`, args `0.2GB`); steady `0.214172s` → `0.306M tok/s`

### `T=32768` (`STEPS=1`)
- fwd: compile `0.9884s`; mem total `1.9GB` (temp `1.3GB`, args `0.4GB`); steady `0.153343s` → `0.855M tok/s`
- bwd: compile `4.3920s`; mem total `19.9GB` (temp `19.1GB`, args `0.4GB`); steady `0.858305s` → `0.153M tok/s`

---

## `TOKAMAX_ATTENTION_IMPL=xla_chunked` with `TOKAMAX_XLA_CHUNK_SIZE=1024,2048`

### `T=4096` (`STEPS=3`)
- fwd: compile `1.9655s`; mem total `0.2GB` (temp `0.1GB`, args `0.0GB`); steady `0.002691s` → `6.09M tok/s`
- bwd: compile `2.7554s`; mem total `0.8GB` (temp `0.7GB`, args `0.0GB`); steady `0.013302s` → `1.23M tok/s`

### `T=8192` (`STEPS=3`)
- fwd: compile `1.9606s`; mem total `0.3GB` (temp `0.1GB`, args `0.1GB`); steady `0.009092s` → `3.60M tok/s`
- bwd: compile `2.7898s`; mem total `1.5GB` (temp `1.4GB`, args `0.1GB`); steady `0.050551s` → `0.648M tok/s`

### `T=16384` (`STEPS=1`)
- fwd: compile `2.0634s`; mem total `0.8GB` (temp `0.5GB`, args `0.2GB`); steady `0.034617s` → `1.89M tok/s`
- bwd: compile `3.1808s`; mem total `3.6GB` (temp `3.3GB`, args `0.2GB`); steady `0.195207s` → `0.336M tok/s`

### `T=32768` (`STEPS=1`)
- fwd: compile `2.4392s`; mem total `2.0GB` (temp `1.4GB`, args `0.4GB`); steady `0.135323s` → `0.969M tok/s`
- bwd: compile `3.8526s`; mem total `12.0GB` (temp `11.3GB`, args `0.4GB`); steady `0.771701s` → `0.170M tok/s`

