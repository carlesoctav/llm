# Faster attention notes + benches (TPU v4)

All benches here use the same core setup unless noted:
- TPU v4, 4 devices (single process)
- `B=4`, `T=S`, `N=4` (q heads), `K=1` (kv heads), `H=256` (head_dim), `layers=1`
- Mask: `ATTENTION_MASK=causal` (i.e. `mask=None`, `is_causal=True`)
- Steps: `STEPS=3` for `T<=8192`, `STEPS=1` for `T>=16384`

Source logs:
- `tokamax_bench.md`
- `tokamanx_new_bench.md`
- `tokamax_remat_rebench.md`

## Tokamax impl sweep (`tokamax_bench.md`)

Comparing Tokamax implementations on TPU v4 (note: local Tokamax venv patch was used to enable Mosaic on TPU v4).

### `T=4096`

| impl | fwd tok/s | bwd tok/s | bwd temp (GB) | notes |
|---|---:|---:|---:|---|
| `mosaic` | 4.17M | 1.41M | 2.0 |  |
| `xla` | 4.74M | 1.69M | 2.5 |  |
| `xla_chunked` (`q=1024, kv=4096`) | 4.55M | 1.20M | 0.8 |  |

### `T=8192`

| impl | fwd tok/s | bwd tok/s | bwd temp (GB) | notes |
|---|---:|---:|---:|---|
| `mosaic` | 2.53M | 0.711M | 8.0 |  |
| `xla` | 2.40M | 0.858M | 10.0 |  |
| `xla_chunked` (`q=1024, kv=4096`) | 2.32M | 0.566M | 1.7 |  |

### `T=16384`

| impl | fwd tok/s | bwd tok/s | bwd temp (GB) | notes |
|---|---:|---:|---:|---|
| `mosaic` | 1.31M | — | — | bwd compile OOM |
| `xla` | 1.20M | — | — | bwd compile OOM |
| `xla_chunked` (`q=1024, kv=4096`) | 1.13M | 0.289M | 2.9 |  |

### `T=32768`

| impl | fwd tok/s | bwd tok/s | bwd temp (GB) | notes |
|---|---:|---:|---:|---|
| `mosaic` | 0.685M | — | — | bwd compile OOM |
| `xla` | — | — | — | fwd OOM |
| `xla_chunked` (`q=1024, kv=4096`) | 0.575M | 0.142M | 7.3 |  |

## Tokamax `xla_chunked`: chunk-size sweep (`tokamanx_new_bench.md`)

Throughput vs memory tradeoffs for Tokamax `xla_chunked` (no remat-on-q-loop patch).

| T | chunk | bwd tok/s | bwd temp (GB) |
|---:|---|---:|---:|
| 4096 | `512,1024` | 1.19M | 0.4 |
| 8192 | `512,1024` | 0.622M | 1.4 |
| 16384 | `512,1024` | 0.306M | 5.0 |
| 32768 | `512,1024` | 0.153M | 19.1 |
| 4096 | `1024,2048` | 1.23M | 0.7 |
| 8192 | `1024,2048` | 0.648M | 1.4 |
| 16384 | `1024,2048` | 0.336M | 3.3 |
| 32768 | `1024,2048` | 0.170M | 11.3 |

## Tokamax `xla_chunked` + q-loop `remat` (`tokamax_remat_rebench.md`)

Same chunk-size sweep, but with a local Tokamax patch that rematerializes the q-chunk scan body.

| T | chunk | bwd tok/s | bwd temp (GB) |
|---:|---|---:|---:|
| 4096 | `512,1024` | 1.28M | 0.1 |
| 8192 | `512,1024` | 0.655M | 0.2 |
| 16384 | `512,1024` | 0.331M | 0.5 |
| 32768 | `512,1024` | 0.165M | 1.7 |
| 4096 | `1024,2048` | 1.43M | 0.3 |
| 8192 | `1024,2048` | 0.749M | 0.5 |
| 16384 | `1024,2048` | 0.384M | 0.9 |
| 32768 | `1024,2048` | 0.194M | 2.0 |

