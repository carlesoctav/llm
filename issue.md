**Title**

`sync_weights` on TorchAx TPU backend bypasses backend-specific fused linear weight processing

**Body**

I’m integrating the TorchAx TPU backend with my own training codebase. My external state intentionally mirrors the HF parameter mapping, so I expected `sync_weights` to reuse the same weight-processing path as the normal initial model load.

From tracing the code, it looks like that is not what happens.

## What I observed

With Gemma3 on the TorchAx TPU backend:

1. Initial model load works and generation is sane.
2. I call `sync_weights` using an external JAX state that mirrors the HF-style parameter layout.
3. After `sync_weights`, generation becomes garbage.

## What I expected

I expected `sync_weights` to reuse the same backend/model-specific processing that happens during the initial TorchAx load, especially for fused linear layers.

## What seems to happen instead

During the initial TorchAx load, there are model/backend-specific processing steps for fused linear weights:

- the model loader maps HF weights into fused modules like `QKVParallelLinear` and `MergedColumnParallelLinear`
- then the TorchAx TPU path runs post-load processing for linear layers
- that path calls `process_linear_weights(...)`

So the live runtime state after initial load is not just a raw HF-like mapping copied into the model.

By contrast, `sync_weights` seems to bypass that processing and only does:

- key remap
- optional transpose
- dtype conversion
- reshard / assign into the existing state

Concretely, from tracing the code:

- initial load goes through:
  - `VllmModelWrapper.load_weights`
  - `attach_incremental_weight_loader`
  - `VllmUnquantizedLinearMethod.maybe_process_weights`
  - `VllmUnquantizedLinearMethod.process_weights_after_loading`
  - `process_linear_weights(...)`
- but `sync_weights` goes through:
  - `TPURunner._sync_weights`
  - `transfer_state_with_mappings(...)`

and does not appear to rerun the fused linear weight processing.

## Why I think this matters

Gemma3 uses fused modules like:

- `qkv_proj` as `QKVParallelLinear`
- `gate_up_proj` as `MergedColumnParallelLinear`

So if an external caller provides HF-like weights, `sync_weights` appears to require the caller to already know the final TorchAx TPU internal layout for those fused layers.

That is hard to do correctly outside the backend, and it defeats the point of using HF-like mappings from an external training stack.

## Request / proposal

I think `sync_weights` on the TorchAx TPU backend should either:

1. rerun the same backend-specific fused weight processing used during initial load, or
2. clearly document that `sync_weights` expects already-postprocessed backend-internal weights, not HF-like weights

My preference would be option 1: reuse the same processing path as the initial load, so external training code can provide HF-like weights/mappings and let the backend handle the final fused layout.

## Question

Is the intended contract for TorchAx TPU `sync_weights` that callers must provide the already-processed backend layout for fused linear layers?

If not, would it make sense for `sync_weights` to reuse the initial-load weight processing path for those layers?
