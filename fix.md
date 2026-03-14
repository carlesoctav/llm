# Fix Plan

## Goal

Replace the broken benchmark with a faithful `simply`-style benchmark for:

- dense prefill
- dense shared-cursor decode
- explicit `prefill_size`
- explicit `max_decode_steps`
- decode chunking API, with chunk size set equal to max decode steps for now

## Plan

1. Add a small decoding schedule helper for the benchmark.
   It must match `simply`:
   - `prefill_size`
   - `begin_position = min(prefill_size, min_input_length - 1)`
   - `end_position_exclusive = min(max_seq_len - 1, max_input_length + max_decode_steps - 1)`
   - `chunk_size = max_decode_steps` for now

2. Build a benchmark sampling state that matches the dense path.
   Required fields:
   - `tokens`
   - `decode_state`
   - `position`
   - `input_lens`
   - `max_decode_steps`
   - `eos_ids`

3. Implement prefill exactly once on `tokens[:, :prefill_size]`.
   This should initialize the decode state and set `position = begin_position`.

4. Implement the dense decode loop with the same row-wise branch as `simply`.
   For each step:
   - run the model on `tokens[:, position:position+1]`
   - sample next token
   - if next position is still prompt for a row, use the existing prompt token
   - otherwise use the sampled token
   - write the chosen token back into `tokens[:, position + 1]`
   - advance `position`

5. Count timings separately and honestly.
   - prefill benchmark: only the prefill call
   - decode benchmark: only the post-prefill decode loop
   - report both "KV steps/s" and "generated tokens/s"
   Because `simply` decode may spend some steps finishing prompt tail.

6. Add two benchmark modes.
   - `micro_kv_replay`: the current low-level loop, renamed correctly
   - `simply_generate`: the faithful benchmark above
   Do not call the microbenchmark "simply-like" anymore.

7. Validate on a toy case before large TPU runs.
   Example:
   - batch prompt lengths `[5, 10]`
   - `prefill_size = 8`
   - confirm row 0 starts generating at slot 5
   - confirm row 1 keeps consuming prompt tokens at slots 5, 6, 7, 8, 9

8. Re-run the comparison only after the semantics match.
   Compare:
   - this repo `simply_generate`
   - `jax-llm` microbenchmark, labeled as microbenchmark
   - if needed, a matching microbenchmark in this repo for apples-to-apples kernel comparison

## Non-goals for the immediate fix

- paged attention
- per-row cursor / ragged decode
- reward or RL integration
- changing the model architecture

## Expected output after the fix

The benchmark output must explicitly say whether it is:

- `prefill`
- `decode_kv_steps`
- `generated_tokens`
- `micro_kv_replay`
- `simply_generate`

so we do not mix incompatible numbers again.
