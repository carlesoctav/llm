# Next Steps — Continuous Batching Inference v2 (inference-2 branch)

Goal: close the performance gap vs `vllm-tpu` (especially long prefill + long decode) by
making our engine architecture closer to `tpu-inference`:
- persistent `InputBatch` buffers with in-place numpy mutation,
- minimal scheduler output (no token/materialization inside scheduler),
- precompiled subprograms (backbone → select → logits → sample),
- avoid logits/unembed/sampling work when no sequences need sampling.

## Why we’re slow today (hypothesis to validate with bench)
1. We compute logits + sample every step even when `sample_mask` is false for all seqs
   (chunked prefill steps). This is catastrophic for long prompts (e.g. 2048 tokens)
   because we do many prefill chunks but only need to sample once at the end.
2. We always compute logits for `max_num_seqs` even if only a subset needs sampling.
3. We do host-side array construction + many `device_put/device_get` every step.

## Phase A — Create vLLM-like data model + scheduler output
1. Add `SamplingParams` (subset of vLLM params we support now):
   - `max_tokens`, `temperature`, `top_p`, `ignore_eos`, `seed`, `n` (keep `n=1` for now).
2. Define `SchedulerOutput` (vLLM-style) emitted per step:
   - `num_scheduled_tokens_per_slot[max_num_seqs]`
   - `slot_ids_scheduled` (packed)
   - `slot_ids_sampled` (packed) + `num_sampled`
   - `total_num_scheduled_tokens`, `padded_total_tokens` (bucket)
   - `padded_num_seqs` (bucket for sampling/logits)
   - `is_decode_only` / distribution markers (future attention kernel parity)
3. Refactor `Scheduler.schedule()` to NOT build `token_ids/positions`.
   - Scheduler only decides (a) prefill chunk lengths, (b) decode 1-token per seq, (c) which seqs sample.

## Phase B — Persistent `InputBatch` + `_prepare_inputs` (CPU → JAX)
1. Implement `InputBatch`:
   - fixed numpy buffers:
     - `token_ids_cpu[max_num_seqs, max_model_len]` (prompt + generated)
     - `num_computed_tokens_cpu[max_num_seqs]`
     - `input_ids_cpu[max_num_batched_tokens_bucket]` (concatenated q tokens)
     - `positions_cpu[max_num_batched_tokens_bucket]`
     - `cu_q_lens_cpu[max_num_seqs+1]`, `kv_lens_cpu[max_num_seqs]`, `num_seqs_cpu[1]`
     - `page_indices_cpu[max_num_seqs, pages_per_seq]` (static mapping)
     - `sample_slot_ids_cpu[max_num_seqs]` (padded with -1)
2. Implement `AttentionMetadata`:
   - derived from `InputBatch` for ragged attention (`kv_lens`, `cu_q_lens`, `num_seqs`, `page_indices`).
3. Implement `TPUSamplingMetadata`:
   - `from_input_batch(...)` computes:
     - `num_sampled`, `sample_slot_ids[padded_num_seqs]`
     - `last_token_indices[padded_num_seqs]` (index into hidden states)
4. Port a simplified `_prepare_inputs` pattern:
   - Update `input_ids_cpu/positions_cpu` via `np.take` from `token_ids_cpu` based on scheduled tokens.
   - Build `cu_q_lens_cpu`, `kv_lens_cpu`, `num_seqs_cpu`.
   - Only build `TPUSamplingMetadata` when `num_sampled>0`.
   - Convert to JAX with a small number of `device_put` calls.

## Phase C — Split runner into compiled subprograms (vLLM-like)
1. `backbone_step(bucket_tokens)`:
   - forward inference for scheduled q tokens
   - update KV cache (donate)
   - return `hidden_states_q[total_q_tokens, hidden]` (or padded `[bucket_tokens,...]`)
2. `select_from_array`:
   - gather only the `hidden_states` rows that correspond to sampled slots’ last q token.
   - compiled for `padded_num_seqs` buckets (e.g. {8,16,32,...,max_num_seqs}).
3. `compute_logits`:
   - unembed only the selected hidden states
   - compiled for same `padded_num_seqs` buckets
4. `sample`:
   - compiled for same `padded_num_seqs` buckets
   - compile two variants: `do_sampling=True` and `do_sampling=False`
   - for `temperature==0` use argmax path; else categorical.
5. Wire it together in worker:
   - If `num_sampled==0`: run only `backbone_step` and skip logits/sample entirely.
   - Else: `backbone_step` → `select` → `logits` → `sample` and scatter results to per-slot next tokens.

## Phase D — Precompile everything up front
1. Add a simple `CompilationManager`:
   - enumerates token buckets for backbone: `{16..max_num_batched_tokens} *2`
   - enumerates seq buckets for sampling/logits: `{8..max_num_seqs} *2`
   - runs each compiled fn once with dummy inputs to force XLA compilation.
2. Run compilation at `LLM`/`Worker` init so benches are steady-state immediately.

## Phase E — KV cache closer to vLLM-TPU (incremental)
1. Keep our fixed page mapping per slot (v1), but:
   - rename/structure fields to match vLLM vocabulary (`block_tables/page_indices`, `seq_lens/kv_lens`).
2. Optional (follow-up): switch to `tpu-inference` RPA v3 (fused KV write) if still behind.

## Phase F — Benchmark + validation
1. Short case (current baseline):
   - `input_len=1024`, `output_len=128`, `max_model_len=2048`, `max_num_batched_tokens=2048`, `max_num_seqs=32`.
2. Long case (regression target):
   - `input_len=2048`, `output_len=512`, `max_model_len=4096`, `max_num_batched_tokens=2048`, `max_num_seqs=32`.
3. Use `bench_compare_vllm_tpu.py` and record:
   - total tok/s + output tok/s
   - ratio vs vLLM.

## Deliverables (files)
- `src/jaxformers/inference/input_batch.py` (new)
- `src/jaxformers/inference/sampling.py` (new)
- `src/jaxformers/inference/attention_metadata.py` (new)
- `src/jaxformers/inference/model_runner.py` (refactor into subprograms + precompile hooks)
- `src/jaxformers/inference/worker.py` (use InputBatch + SchedulerOutput)
- `src/jaxformers/inference/scheduler.py` (emit SchedulerOutput, not token lists)
- Bench run results checked for both short and long configs.

