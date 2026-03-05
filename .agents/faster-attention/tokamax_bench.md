# Tokamax attention bench (TPU v4)

Date: 2026-03-05

## Environment

- Hardware: TPU v4, 4 devices (single process)
- `jax==0.9.0.1`, `jaxlib==0.9.0.1`, `tokamax==0.0.10`
- Bench: `src/jaxformers/bench/bench_attention_tokamax.py`
- Local Tokamax patch to enable TPU v4 Mosaic: relaxed
  `PallasMosaicTpuFlashAttention.supported_on()` from `generation >= 5` to `>= 4`
  in `.venv/lib/python3.11/site-packages/tokamax/_src/ops/attention/pallas_mosaic_tpu.py`

## Common setup (unless noted)

- Shapes: `B=4`, `T=S`, `N=4`, `K=1`, `H=256`, `layers=1`
- Mask: `ATTENTION_MASK=causal` (i.e. `mask=None`, `is_causal=True`)
- Steps: `STEPS=3` for `T<=8192`, `STEPS=1` for `T>=16384`
- Throughput: `tok/s = (B*T) / steady_time_s`

---

## Results: `ATTENTION_MASK=causal`

### `TOKAMAX_ATTENTION_IMPL=mosaic` (Splash/Mosaic TPU flash-attn)

#### `T=4096`
- fwd: compile `0.7696s`; mem total `0.1GB` (temp `0.0GB`, args `0.0GB`); steady `0.003931s` → `4.17M tok/s`
- bwd: compile `0.7328s`; mem total `2.1GB` (temp `2.0GB`); steady `0.011584s` → `1.41M tok/s`

#### `T=8192`
- fwd: compile `1.2822s`; mem total `0.2GB` (temp `0.0GB`, args `0.1GB`); steady `0.012960s` → `2.53M tok/s`
- bwd: compile `0.9868s`; mem total `8.2GB` (temp `8.0GB`, args `0.1GB`); steady `0.046113s` → `0.711M tok/s`

#### `T=16384`
- fwd: compile `3.5386s`; mem total `0.3GB` (temp `0.0GB`, args `0.2GB`); steady `0.049866s` → `1.31M tok/s`
- bwd: **compile OOM** (program needs ~`32GB`; largest alloc: `f32[4,128,4,16384,256]` ≈ `32GB`)

#### `T=32768`
- fwd: compile `12.1758s`; mem total `0.9GB` (temp `0.3GB`, args `0.4GB`); steady `0.191321s` → `0.685M tok/s`
- bwd: **compile OOM** (tries alloc: `f32[4,256,4,32768,256]` ≈ `128GiB`)

---

### `TOKAMAX_ATTENTION_IMPL=xla`

#### `T=4096`
- fwd: compile `1.1901s`; mem total `1.1GB` (temp `1.0GB`); steady `0.003458s` → `4.74M tok/s`
- bwd: compile `2.4312s`; mem total `2.6GB` (temp `2.5GB`); steady `0.009699s` → `1.69M tok/s`

#### `T=8192`
- fwd: compile `1.5572s`; mem total `4.2GB` (temp `4.0GB`, args `0.1GB`); steady `0.013674s` → `2.40M tok/s`
- bwd: compile `1.8517s`; mem total `10.2GB` (temp `10.0GB`, args `0.1GB`); steady `0.038171s` → `0.858M tok/s`

#### `T=16384`
- fwd: compile `1.7275s`; mem total `16.6GB` (temp `16.3GB`, args `0.2GB`); steady `0.054803s` → `1.20M tok/s`
- bwd: **compile OOM** (program needs ~`40GB`; top allocations are `T×T`-sized `f32` logits/temps)

#### `T=32768`
- fwd: **OOM** (tries alloc: `f32[4,4,32768,32768]` ≈ `64GiB`)

---

### `TOKAMAX_ATTENTION_IMPL=xla_chunked` with custom chunking

Env:
- `TOKAMAX_XLA_CHUNK_SIZE=1024,4096` (i.e. `q_chunk=1024`, `kv_chunk=4096`)

#### `T=4096`
- fwd: compile `1.2972s`; mem total `0.3GB` (temp `0.3GB`); steady `0.003604s` → `4.55M tok/s`
- bwd: compile `3.0061s`; mem total `0.9GB` (temp `0.8GB`); steady `0.013641s` → `1.20M tok/s`

#### `T=8192`
- fwd: compile `1.3784s`; mem total `0.5GB` (temp `0.4GB`, args `0.1GB`); steady `0.014132s` → `2.32M tok/s`
- bwd: compile `2.4806s`; mem total `1.9GB` (temp `1.7GB`, args `0.1GB`); steady `0.057929s` → `0.566M tok/s`

#### `T=16384`
- fwd: compile `1.6997s`; mem total `1.1GB` (temp `0.8GB`, args `0.2GB`); steady `0.057908s` → `1.13M tok/s`
- bwd: compile `3.2657s`; mem total `3.3GB` (temp `2.9GB`, args `0.2GB`); steady `0.226935s` → `0.289M tok/s`

#### `T=32768`
- fwd: compile `1.6794s`; mem total `1.9GB` (temp `1.3GB`, args `0.4GB`); steady `0.228024s` → `0.575M tok/s`
- bwd: compile `3.1128s`; mem total `8.1GB` (temp `7.3GB`, args `0.4GB`); steady `0.920838s` → `0.142M tok/s`

Notes:
- Tokamax `xla_chunked` currently calls `Mask.as_array(seq_q, seq_kv)` even for `is_causal=True`,
  which builds a full `[T,S]` boolean causal mask (unbatched) inside the program.
  At `T=32768`, that’s ~`1GiB` of booleans.

---

## User-provided results: `ATTENTION_MASK=bool` (from terminal logs)

### `TOKAMAX_ATTENTION_IMPL=auto`

#### `T=2048`
- fwd mem total `0.3GB` (temp `0.3GB`); steady `0.001038s` → `7.89M tok/s`
- bwd mem total `0.7GB` (temp `0.6GB`); steady `0.002766s` → `2.96M tok/s`

#### `T=8192`
- fwd mem total `4.7GB` (temp `4.3GB`, args `0.3GB`); steady `0.014090s` → `2.33M tok/s`
- bwd mem total `10.7GB` (temp `10.3GB`, args `0.3GB`); steady `0.040335s` → `0.812M tok/s`

### `TOKAMAX_ATTENTION_IMPL=xla_chunked` (default chunking)

#### `T=8192` (default chunk size `128`)
- fwd mem total `0.4GB` (temp `0.0GB`, args `0.3GB`); steady `0.042590s` → `0.769M tok/s`
- bwd mem total `9.2GB` (temp `8.8GB`, args `0.3GB`); steady `0.249588s` → `0.131M tok/s`

### `TOKAMAX_ATTENTION_IMPL=xla_chunked` (custom chunking; user noted `(1024, 4196)`)

#### `T=8192`
- fwd mem total `0.8GB` (temp `0.4GB`, args `0.3GB`); steady `0.015961s` → `2.05M tok/s`
- bwd mem total `2.5GB` (temp `2.0GB`, args `0.3GB`); steady `0.065147s` → `0.503M tok/s`



• Code

  - Added TPU flash-attn op src/jaxformers/ops/attention/flash_attention.py:1 (wraps jax.experimental.pallas.ops.tpu.flash_attention.flash_attention, uses jax.shard_map automatically when
    inputs are sharded).
  - Exported + wired into interface/bench:
      - src/jaxformers/ops/attention/__init__.py:1
      - src/jaxformers/attention_utils.py:1
      - src/jaxformers/bench/bench_attention_impls.py:1 (new ATTENTION_IMPL=flash_attention)
  - Added TPU-only correctness/grad test tests/test_attention_flash_attention.py:1.

  Bench (TPU, ATTENTION_MASK=causal, Gemma3-ish N=4 K=1 H=256)

  - B=16 T=4096, MESH_B=4 MESH_N=1:
      - flash_attention: fwd 15.00M tok/s, bwd 4.23M tok/s, bwd temp 0.1 GB
      - sdpa: fwd 18.06M, bwd 6.43M, bwd temp 2.0 GB
      - xla_chunked (QUERY=256 KEY=256): fwd 6.36M, bwd 1.60M, bwd temp 1.1 GB
      - tokamax_xla_chunked: fwd 6.64M, bwd 1.20M, bwd temp 2.1 GB
  - B=16 T=8192, MESH_B=4 MESH_N=1:
      - flash_attention: fwd 8.16M, bwd 2.23M, bwd temp 0.4 GB
      - sdpa: fwd 9.49M, bwd 3.30M, bwd temp 8.1 GB
      - xla_chunked (QUERY=256 KEY=512): fwd 4.00M, bwd 0.93M, bwd temp 2.3 GB
      - tokamax_xla_chunked: fwd 3.47M, bwd 0.54M, bwd temp 8.6 GB
  - Also works with head-sharding (MESH_B=1 MESH_N=4) on B=16 T=4096: fwd 15.97M, bwd 4.07M tok/s.

  Repro command (example):
  USE_Q_SHARDING=1 MESH_B=4 MESH_N=1 ATTENTION_MASK=causal BATCH=16 SEQLEN=4096 Q_HEADS=4 KV_HEADS=1 HEAD_DIM=256 STEPS=3 ATTENTION_IMPL=flash_attention uv run python src/jaxformers/bench/
  bench_attention_impls.py

---

## Jaxformers: `ATTENTION_IMPL=flash_attention` (TPU v4)

Bench: `src/jaxformers/bench/bench_attention_impls.py`

Common setup:
- `B=4`, `T=S`, `N=4`, `K=1`, `H=256`, `layers=1`
- `ATTENTION_MASK=causal`, `USE_Q_SHARDING=0`
- `STEPS=3` for `T<=8192`, `STEPS=1` for `T>=16384`

### `T=4096` (`STEPS=3`)
- fwd: compile `0.4966s`; mem total `0.1GB` (temp `0.0GB`, args `0.0GB`); steady `0.004151s` → `3.95M tok/s`
- bwd: compile `0.7668s`; mem total `0.2GB` (temp `0.1GB`, args `0.0GB`); steady `0.015386s` → `1.06M tok/s`

### `T=8192` (`STEPS=3`)
- fwd: compile `0.5352s`; mem total `0.2GB` (temp `0.1GB`, args `0.1GB`); steady `0.015893s` → `2.06M tok/s`
- bwd: compile `0.8618s`; mem total `0.6GB` (temp `0.4GB`, args `0.1GB`); steady `0.060186s` → `0.544M tok/s`

### `T=16384` (`STEPS=1`)
- fwd: compile `0.6510s`; mem total `0.6GB` (temp `0.3GB`, args `0.2GB`); steady `0.061930s` → `1.06M tok/s`
- bwd: compile `1.1881s`; mem total `1.3GB` (temp `0.9GB`, args `0.2GB`); steady `0.235141s` → `0.279M tok/s`

### `T=32768` (`STEPS=1`)
- fwd: compile `0.8998s`; mem total `1.4GB` (temp `0.8GB`, args `0.4GB`); steady `0.243979s` → `0.537M tok/s`
- bwd: compile `1.8906s`; mem total `2.8GB` (temp `2.0GB`, args `0.4GB`); steady `0.920753s` → `0.142M tok/s`

---

## Jaxformers: `ATTENTION_IMPL=xla_chunked` (TPU v4)

Bench: `src/jaxformers/bench/bench_attention_impls.py`

Env:
- `QUERY_CHUNK_SIZE=512`, `KEY_CHUNK_SIZE=1024`
- `B=4`, `T=S`, `N=4`, `K=1`, `H=256`, `layers=1`
- `ATTENTION_MASK=causal`, `USE_Q_SHARDING=0`
- `STEPS=3` for `T<=8192`, `STEPS=1` for `T>=16384`

### `T=4096` (`STEPS=3`)
- fwd: compile `4.6523s`; mem total `0.1GB` (temp `0.0GB`, args `0.0GB`); steady `0.006865s` → `2.39M tok/s`
- bwd: compile `11.0430s`; mem total `0.2GB` (temp `0.1GB`, args `0.0GB`); steady `0.024461s` → `0.670M tok/s`

### `T=8192` (`STEPS=3`)
- fwd: compile `5.1911s`; mem total `0.2GB` (temp `0.1GB`, args `0.1GB`); steady `0.026339s` → `1.24M tok/s`
- bwd: compile `13.8041s`; mem total `0.3GB` (temp `0.1GB`, args `0.1GB`); steady `0.095550s` → `0.343M tok/s`

### `T=16384` (`STEPS=1`)
- fwd: compile `4.7590s`; mem total `0.4GB` (temp `0.1GB`, args `0.2GB`); steady `0.104097s` → `0.630M tok/s`
- bwd: compile `11.1844s`; mem total `0.7GB` (temp `0.3GB`, args `0.2GB`); steady `0.389076s` → `0.168M tok/s`

### `T=32768` (`STEPS=1`)
- fwd: compile `5.1243s`; mem total `0.9GB` (temp `0.3GB`, args `0.4GB`); steady `0.413522s` → `0.317M tok/s`
- bwd: compile `13.3086s`; mem total `1.5GB` (temp `0.8GB`, args `0.4GB`); steady `1.555700s` → `0.0843M tok/s`

---

## Jaxformers: `ATTENTION_IMPL=xla_chunked` (TPU v4, bigger chunks)

Bench: `src/jaxformers/bench/bench_attention_impls.py`

Env:
- `QUERY_CHUNK_SIZE=1024`, `KEY_CHUNK_SIZE=2048`
- `B=4`, `T=S`, `N=4`, `K=1`, `H=256`, `layers=1`
- `ATTENTION_MASK=causal`, `USE_Q_SHARDING=0`
- `STEPS=3` for `T<=8192`, `STEPS=1` for `T>=16384`

### `T=4096` (`STEPS=3`)
- fwd: compile `3.8243s`; mem total `0.1GB` (temp `0.0GB`, args `0.0GB`); steady `0.005367s` → `3.05M tok/s`
- bwd: compile `7.3888s`; mem total `0.4GB` (temp `0.3GB`, args `0.0GB`); steady `0.022193s` → `0.738M tok/s`

### `T=8192` (`STEPS=3`)
- fwd: compile `5.1752s`; mem total `0.2GB` (temp `0.1GB`, args `0.1GB`); steady `0.022029s` → `1.49M tok/s`
- bwd: compile `6.4779s`; mem total `0.7GB` (temp `0.5GB`, args `0.1GB`); steady `0.086426s` → `0.379M tok/s`

### `T=16384` (`STEPS=1`)
- fwd: compile `5.1745s`; mem total `0.4GB` (temp `0.1GB`, args `0.2GB`); steady `0.086535s` → `0.757M tok/s`
- bwd: compile `6.8166s`; mem total `1.1GB` (temp `0.7GB`, args `0.2GB`); steady `0.342191s` → `0.192M tok/s`

### `T=32768` (`STEPS=1`)
- fwd: compile `6.7127s`; mem total `0.9GB` (temp `0.3GB`, args `0.4GB`); steady `0.343477s` → `0.382M tok/s`
- bwd: compile `8.3024s`; mem total `1.9GB` (temp `1.1GB`, args `0.4GB`); steady `1.363071s` → `0.0962M tok/s`
