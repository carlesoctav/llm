# jaxformers

`jaxformers` is a pure-JAX post-training library for transformer models. It is built around explicit weight pytrees, direct JAX transforms, and mesh-based sharding instead of Flax/Linen abstractions. The current focus is practical fine-tuning and next-token-prediction workloads with smaller, more predictable compile graphs, 4D parallelism, and lower memory pressure.

## Highlights

- Pure JAX training stack, no Flax.
- Compile-friendly explicit training step with optional loop-based forward paths.
- 4D parallelism with `dp_replicate`, `dp_shard`, `cp`, and `tp`.
- Torchtitan-like sharding layout for data, parameter, context, and tensor partitioning.
- Chunked attention and chunked cross entropy paths for lower peak memory pressure.
- Hugging Face checkpoint loading from `safetensors`.
- LoRA support for parameter-efficient post-training.
- Python-first config system built on `sws`.
- Built-in logging, checkpointing, benchmark, and sweep utilities.

## Current Scope

This repo is centered on post-training and fine-tuning rather than a full pretraining platform. The main entrypoints today are:

- `src/jaxformers/train/ntp.py` for next-token prediction and module-based fine-tuning
- `src/jaxformers/train/ntp_with_kl_regularzier.py` for KL-regularized training

Evaluation is currently not implemented, so the example configs run with `skip_eval = True`.

## Installation

Requirements:

- Python 3.11+
- `uv`
- A working JAX runtime

The default dependency set in `pyproject.toml` is TPU-oriented because it pins `jax[tpu]==0.9.0`. If you are targeting a different backend, adjust dependencies accordingly.

```bash
uv sync
```

## Quick Start

The main training command is:

```bash
uv python src/jaxformers/train/ntp.py --config path/to/config.py
```

If you want to use the repo's existing example config:

```bash
uv python src/jaxformers/train/ntp.py --config experiments/tunix-repro/lora_gemma3_config.py
```

Note: the correct path is `src/jaxformers/train/ntp.py`, not `src/jaxformers/trian/ntp.py`.

The `--config` target is a Python file that returns an `sws.Config`. Example configs live under `experiments/`.

## Configuration

Configs are Python builders, not static YAML files. The training script expects plain string values for things like model names, optimizer names, attention implementations, and loss implementations.

Typical top-level config areas are:

- experiment metadata: `exp_name`, `project_name`, `dir`, `ckpt_path`
- runtime: `seed`, `max_train_step`, `forward_dtype`, `grad_accum`
- model: `model_name`, `model.*`
- data: `data.source_name`, `data.source.*`, `data.transforms_name`, `data.transforms.*`, `train_loader_name`, `train_loader.*`
- optimization: `optimizer_*`, `learning_rate`, `lr_scheduler_*`
- logging and callbacks: `logger_name`, `callback_name`
- loss selection: `loss_impl`

A minimal example looks like this:

```python
import jax
import jax.numpy as jnp
import sws
from transformers import AutoTokenizer


def get_config():
    config = sws.Config()

    config.exp_name = "gemma3-lora-sft"
    config.project_name = "jaxformers"
    config.dir = "./artifacts"
    config.ckpt_path = lambda: f"{config.dir}/{config.project_name}/{config.exp_name}"

    config.seed = 42
    config.max_train_step = 1000
    config.skip_eval = True
    config.eval_every = None
    config.forward_dtype = lambda: jnp.bfloat16
    config.loss_impl = "xla_chunked"
    config.grad_accum = 4
    config.weights_impl = "stack"
    config.checkpoint_options.save_interval_steps = 0
    config.checkpoint_options.max_to_keep = 1

    config.logger_name = "noop"

    config.init_lora = None

    config.model_name = (
        "{MODEL_DIR}.huggingface.gemma3.Gemma3ForCausalLM.from_pretrained"
    )
    config.model.model_id = "google/gemma-3-1b-it"
    config.parallel.parallel_dims = {
        "dp_replicate": 1,
        "dp_shard": 4,
        "cp": 1,
        "tp": 1,
    }
    config.parallel.devices = lambda: jax.devices()
    config.parallel.multihost = False
    config.model.additional_config.remat_layer = False
    config.model.additional_config.attn_impl = "xla_chunked"
    config.model.additional_config.forward_impl = "loop"
    config.model.param_dtype = lambda: jnp.bfloat16

    config.data.source_name = "huggingface"
    config.data.source.load_kwargs = [
        {
            "path": "your/dataset",
            "split": "train",
            "streaming": False,
        }
    ]
    config.data.transforms_name = "ntp"
    config.data.transforms.column = "messages"
    config.data.transforms.data_type = "chat"
    config.data.transforms.max_length = 2048
    config.data.transforms.assistant_loss = True
    config.data.transforms.tokenizer = lambda: AutoTokenizer.from_pretrained(
        config.model.model_id
    )

    config.train_loader_name = "simple"
    config.train_loader.global_batch_size = 32
    config.train_loader.shuffle = False
    config.train_loader.drop_remainder = True

    config.optimizer_name = "adam"
    config.optimizer.max_grad_norm = 1.0
    config.optimizer.b1 = 0.9
    config.optimizer.b2 = 0.95
    config.optimizer.eps = 1e-8

    config.learning_rate = 1e-5
    config.lr_scheduler_name = "wsds"
    config.lr_scheduler.min_lr_ratio = 0.1
    config.lr_scheduler.warmup = 0.01

    return config
```

## Parallelism

The model mesh is expressed as four named axes:

- `dp_replicate`: replicated data-parallel axis
- `dp_shard`: sharded data-parallel / FSDP-like axis
- `cp`: context parallel axis
- `tp`: tensor parallel axis

The product of these dimensions should match the number of devices in the mesh. Model loaders translate logical sharding rules into physical `PartitionSpec`s from this 4D layout, and singleton axes are dropped automatically.

For models that support it, `config.model.additional_config.sequence_parallelism` controls whether sequence/context sharding participates in the physical layout.

## Memory-Efficient Paths

The repo includes chunked implementations to reduce peak activation and logits memory, especially for long context lengths or large vocabularies.

### Attention

Set `config.model.additional_config.attn_impl` to one of:

- `"eager"`: simple JAX path
- `"sdpa"`: Tokamax scaled dot-product attention
- `"xla_chunked"`: chunked attention path
- `"chunked_manual"`: explicit manual chunked attention path

In this repo, `"xla_chunked"` is intentionally mapped to the manual chunked implementation because the alternative Tokamax chunked backward can use very large temporary memory.

### Cross Entropy

Set `config.loss_impl` to one of:

- `"xla_chunked"`: fused chunked XLA cross entropy
- `"reference"`: dense reference implementation
- `"pallas_tpu"`: TPU-specific kernel when available

Current limitation: the fused XLA chunked cross entropy path supports batch-axis sharding, but not hidden-axis sharded activations, and it currently assumes replicated vocab weights.

## Data Pipeline

The data stack is split into three typed factory layers:

- `data/source/` for dataset construction
- `data/transforms/` for tokenization and NTP transforms
- `data/loader/` for batching and process-local sharded loading

The built-in `ntp` transform supports:

- `data_type = "chat"`, `"text"`, or `"token"`
- optional chat templates
- assistant-only loss masking
- optional packing with Grain first-fit packing

The default loader is Grain-based and can shard data loading across hosts, batch per process, and materialize arrays directly with mesh-aware sharding metadata.

## Models

Current model loaders live under `src/jaxformers/models/` and include:

- `huggingface.gemma3`
- `qwen3`
- `bert`

Model loading is done from Hugging Face checkpoints and `safetensors`, with weights placed directly onto the configured mesh.

## LoRA

LoRA can be enabled by setting `init_lora` and populating the `lora` config block. The existing example in `experiments/tunix-repro/lora_gemma3_config.py` shows the expected shape:

- `lora.rank`
- `lora.alpha`
- `lora.weights_path`

The optimizer path automatically masks training to LoRA parameters when a model has been converted to LoRA weights.

## Logging and Checkpointing

The training loop supports:

- `logger_name = "wandb"` for Weights & Biases
- `logger_name = "noop"` for no-op local runs
- Orbax checkpoint management through `checkpoint_options`
- callback chaining for metrics like learning rate, grad norm, and throughput

On the first compiled step, the training loop also reports compiled memory statistics and FLOPs.

## Benchmarks and Sweeps

Useful utility scripts:

- `src/jaxformers/bench/bench_attention_impls.py`
- `src/jaxformers/bench/bench_cross_entropy_chunked_xla.py`
- `src/jaxformers/train/make_sweep.py`
- `src/jaxformers/train/sweep.py`

The benchmark scripts are useful when comparing attention or loss implementations for compile time, steady-state runtime, and compiled memory use.

## Project Layout

```text
src/jaxformers/
  train/         training entrypoints
  models/        model loaders and forward passes
  ops/           attention and cross entropy kernels
  distributed/   mesh and sharding helpers
  data/          source, transform, and loader factories
  optimizers/    optimizer construction
  scheduler/     learning-rate schedules
  callbacks/     training callbacks
  logger/        wandb and noop logging backends
experiments/     example config builders
tests/           unit tests
```

## Practical Notes

- If you do not want Weights & Biases, set `logger_name = "noop"`.
- If you use the provided experiment configs, update checkpoint and project paths before launching.
- Keep `skip_eval = True` unless you also implement an evaluation path.
- For local experimentation, start with `loss_impl = "reference"` if you want the simplest loss path, then switch to `"xla_chunked"` when tuning memory and scale.

This Research is supported with Cloud TPUs from Google’s TPU Research Cloud (TRC)
