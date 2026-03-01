# Chunked Prefill (Prefill-Stage Token Chunking)

## Motivation
Today, when a sequence is in **prefill** (`cached_len < prompt_len`), the scheduler may schedule the *entire remaining prompt* for that sequence in a single step as long as it fits in `max_num_batched_tokens`. Concretely:

- [`Scheduler.schedule()`](/mnt/carles/llm/src/jaxformers/inference/scheduler.py) sets `q_len = min(remaining_prompt, token_budget)`.
- [`InputBatch.prepare_step_inputs()`](/mnt/carles/llm/src/jaxformers/inference/input_batch.py) then copies that full `q_len` slice into the packed `input_ids_cpu[max_num_batched_tokens]` buffer.

For long prompts, this can create very large prefill steps (and large host->device transfers and TPU work) even when we would prefer to:

- keep each prefill iteration bounded (more stable latency),
- leave room for other work (other prefills / decodes) rather than having one request consume the full iteration,
- avoid having “one huge prefill” dominate the step.

vLLM V1 calls this optimization “chunked prefill” and schedules prefills in chunks within the global token budget. See `/mnt/carles/vllm-tpu/packages/vllm/docs/configuration/optimization.md` and the vLLM scheduler config fields in `/mnt/carles/vllm-tpu/packages/vllm/vllm/config/scheduler.py`.

## Goal
Add a **prefill chunk cap** so that a single sequence’s prefill in one iteration is bounded by a configurable `prefill_chunk_size` (or equivalent), even when `max_num_batched_tokens` would allow scheduling more.

This directly changes the amount we “paste” into the packed `input_ids_cpu[max_num_batched_tokens]` buffer during prefill: we copy only up to the chunk size per step, not the entire prompt remainder.

## Non-Goals (Initial Version)
- Prefix caching, external KV cache, preemption, or advanced vLLM scheduling policies.
- Changing the JAX model forward (KV update + ragged paged attention). This is purely a scheduling/input-packing change.
- Streaming outputs or multi-process serving.

## Proposed API / Config
Add a new engine-level knob:

- `prefill_chunk_size: int | None` (default: `None`).
- Semantics: `None` keeps current behavior (no additional cap beyond token budget).
- Validation: if set, must satisfy `1 <= prefill_chunk_size <= max_num_batched_tokens`.
- Applies only to **prefill** tokens (`cached_len < prompt_len`). Decode remains `q_len == 1`.

Wiring:

- Extend [`EngineConfig`](/mnt/carles/llm/src/jaxformers/inference/worker.py) with `prefill_chunk_size`.
- Pass it into [`Scheduler.__init__`](/mnt/carles/llm/src/jaxformers/inference/scheduler.py).
- Expose it on [`LLM.__init__`](/mnt/carles/llm/src/jaxformers/inference/llm.py) so benchmarks can tune it.

Naming note:

- vLLM uses `enable_chunked_prefill` and a “long prefill token threshold” concept. For this repo’s minimal engine, a single explicit `prefill_chunk_size` is clearer and maps directly to the desired behavior.

## Scheduling Semantics
### Current
For prefill:

```python
remaining = prompt_len - cached_len
q_len = min(remaining, token_budget)
```

### New (Chunked Prefill Cap)
For prefill:

```python
remaining = prompt_len - cached_len
cap = remaining if prefill_chunk_size is None else min(remaining, prefill_chunk_size)
q_len = min(cap, token_budget)
```

Other rules remain unchanged:

- Decode is prioritized first (still `q_len = 1` per eligible seq, budget permitting).
- `sample_mask[slot]` is set only when we will produce a new token.
- Decode step: `sample_mask=True` for scheduled decode slots.
- Prefill step: `sample_mask=True` only when `cached_len + q_len == prompt_len` (prompt finished this iteration).
- `kv_lens[slot]` for scheduled prefill becomes `cached_len + q_len`, ensuring the KV cache grows monotonically across prefill chunks.

### Input Packing Impact
No changes required in `InputBatch.prepare_step_inputs()`; it already copies `q_len` tokens per sequence into the packed buffers. The new scheduler cap simply ensures `q_len` is chunked as desired.

## Expected Behavior (Examples)
Assume `max_num_batched_tokens = 2048`, one request with `prompt_len = 4096`, no decode traffic.

- Before: first step schedules `q_len=2048`, second step schedules `q_len=2048`.
- With `prefill_chunk_size=512`: eight steps schedule `q_len=512` each; `sample_mask=True` only on the 8th step when prompt completes.

With mixed decode + prefill:

- Decode tokens are scheduled first, reducing `token_budget`.
- Prefill chunk scheduling uses the remaining budget and also respects `prefill_chunk_size`, preventing a single prefill from consuming all remaining tokens.

## Implementation Plan (Follow-Up Work After This Spec)
1. Extend `EngineConfig` with `prefill_chunk_size: int | None = None`.
1. Add `prefill_chunk_size` to `Scheduler.__init__` and store it.
1. Modify the prefill loop in `Scheduler.schedule()` to apply the cap.
1. Add validation (error if `prefill_chunk_size < 1` or `prefill_chunk_size > max_num_batched_tokens`).
1. Add tests for scheduler behavior (prompt chunking across steps, `sample_mask` behavior, and decode priority).
1. (Optional) Add a micro-benchmark to show the reduction in bucket size / transfer size when `prefill_chunk_size` is small and there are few concurrent requests.

## Test Cases (Concrete)
**Scheduler unit test**

Config: `max_num_batched_tokens=8`, `prefill_chunk_size=3`, `max_num_seqs=1`, `max_model_len=32`.

Request: `prompt_len=8`, `max_tokens=1`.

Expected schedule sequence:

1. step 1: `q_len=3`, `sample_mask=False`, `kv_len=3`
1. step 2: `q_len=3`, `sample_mask=False`, `kv_len=6`
1. step 3: `q_len=2`, `sample_mask=True`, `kv_len=8` (prompt finished, first token sampled)

**Integration sanity**

Run `LLM.generate` with a long tokenized prompt and `prefill_chunk_size` enabled.

Verify output token ids match the baseline (chunking should be numerically identical assuming deterministic sampling mode).

## Open Questions
- What default (if any) do we want for `prefill_chunk_size` in benchmarks (e.g. `None` for no behavior change vs `512`/`1024` for bounded step size at the cost of single-request TTFT)?
