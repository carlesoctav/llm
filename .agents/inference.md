# Continuous Batched Inference (JAX/TPU) — Plan

Goal: build a simple, vLLM-like continuous batching engine for JAX models (starting with `qwen3.py`) that:
- runs in a single process (train + generate colocated),
- uses TPU ragged/paged attention for KV cache,
- compiles a small set of primitives for power-of-2 shapes,
- exposes an `LLM` interface with `generate()` returning token ids + `generation_mask`,
- includes a throughput benchmark with the same headline metrics as `vllm bench throughput`.

## Phase 0 — Repo + references
1. Read/borrow patterns from:
   - `/mnt/carles/vllm-tpu/packages/tpu-inference` (TPU runner, paged KV cache, ragged attention)
   - `/mnt/carles/nano-vllm/nanovllm/engine` (request scheduling + interface shape)
2. Identify model entrypoints:
   - `src/jaxformers/modeling_utils.py` (`Model`)
   - `src/jaxformers/models/qwen3.py` (add `forward_inference_*`)

## Phase 1 — Minimal request/scheduler layer (nano-vllm style, simpler)
1. Add `Sequence` and `Scheduler`:
   - keep *at most* `max_num_seqs` active sequences (slots),
   - chunk prefill to respect `max_num_batched_tokens`,
   - decode one token per active seq per step.
2. Define output state:
   - `token_ids`: prompt + generated ids
   - `generation_mask`: boolean list aligned with `token_ids`
3. Chat inputs:
   - `list[str]` → `generation_mask="last"` (only generated tokens marked)
   - `list[list[dict]]` → use `tokenizer.apply_chat_template(...)`
   - mark `"assistant"` segments as generated when configured (default `["last", "assistant"]`).

## Phase 2 — Paged KV cache layout + metadata
1. Choose `page_size` (start with 32; sweep {16, 32, 64} later).
2. Allocate KV pages per layer:
   - layout required by `jax.experimental.pallas.ops.tpu.ragged_paged_attention`:
     `[num_pages, page_size, num_combined_kv_heads, head_dim]`
   - shard `num_combined_kv_heads` along `tp` axis (model axis).
3. Use fixed page mapping per slot (simple first pass):
   - `pages_per_seq = ceil(max_model_len / page_size)`
   - `page_indices[slot, :] = slot * pages_per_seq + arange(pages_per_seq)`
   - no free-list allocator in v1.
4. Per-step metadata (host-built, padded):
   - `token_ids[max_tokens]`, `positions[max_tokens]`, `page_ids[max_tokens]`, `page_offsets[max_tokens]`
   - `kv_lens[max_num_seqs]`, `cu_q_lens[max_num_seqs+1]`, `num_seqs[1]`

## Phase 3 — `qwen3.py` inference forward using ragged paged attention
1. Add ragged RoPE helpers that take `[tokens, heads, head_dim]` + `positions[tokens]`.
2. Add `forward_inference(...)` that:
   - embeds tokens (ragged),
   - for each layer:
     - projects q/k/v for ragged tokens,
     - applies RoPE with per-token positions,
     - writes k/v into layer KV pages at `(page_id, offset)` (drop padded updates),
     - calls `ragged_paged_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs)`,
     - finishes MLP + residual for ragged tokens.
   - returns last-token hidden states per sequence (for sampling) and updated KV cache.

## Phase 4 — JIT compilation + model runner
1. Implement `JaxModelRunner`:
   - compiles step fn(s) for `max_num_batched_tokens` in power-of-2 sizes (≤ configured max),
   - each compiled step does: forward → unembed logits → sample → returns `next_token_ids` + updated KV cache.
2. Padding policy:
   - choose smallest compiled shape that fits `total_q_tokens`,
   - pad inputs; set padded `page_ids=-1, offsets=-1` and use scatter `mode=FILL_OR_DROP` with `wrap_negative_indices=False`.

## Phase 5 — Worker + public interface
1. `JaxWorker` (single process) owns:
   - KV cache state,
   - scheduler/slots,
   - `JaxModelRunner`.
2. `LLM` wraps `JaxWorker`:
   - `generate(prompts, max_tokens, ...)` returns token ids + generation mask (no detokenize by default).

## Phase 6 — Benchmark (vLLM-bench-compatible headline)
1. Add `bench_inference_throughput.py` with args similar to `vllm bench throughput`:
   - `--model`, `--dataset-name random`, `--num-prompts`, `--input-len`, `--output-len`, plus engine limits.
2. Print:
   - `Throughput: X requests/s, Y total tokens/s, Z output tokens/s`
   - totals for prompt/output tokens
3. Add a helper script to run:
   - vLLM TPU bench (using `/mnt/carles/vllm-tpu/.venv`) and this engine,
   - dump JSON for side-by-side comparison.

## Phase 7 — TPU bench + tuning
1. Run on `Qwen/Qwen3-0.6B` with tp=4.
2. Sweep page_size {16,32,64} and pick best throughput.
3. Compare against vllm-tpu throughput; iterate on hotspots:
   - KV cache update pattern (prefill block-wise),
   - compiled shape set (token count buckets),
   - sampling cost (option: greedy vs categorical).

