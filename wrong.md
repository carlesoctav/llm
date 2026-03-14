# Wrong

I wrote the benchmark with the wrong decode semantics.

## What I did wrong

1. I called a `fori_loop(0, seq_len)` KV-cache replay a "decode benchmark".
   That is not the `simply` generation path.
2. I did not mimic `simply`'s two-phase flow:
   `prefill(prompt[:prefill_size])` first, then `continue_decode(...)`.
3. I ignored `begin_position = min(prefill_size, min_input_length - 1)`.
   That position is the core of the `simply` schedule.
4. I treated all decode steps as "real generation" steps.
   In `simply`, post-prefill decode can still be walking through prompt tail
   tokens before actual generation starts.
5. I did not model the teacher-forcing branch from `simply`:
   if the next position is still inside the prompt, the next token must come
   from the prompt buffer, not from sampling.
6. I did not separate:
   - prompt length
   - prefill size
   - max decode steps
   - chunk size
   Those are different knobs in `simply`.
7. I produced a benchmark that was closer to the `jax-llm` one-token loop than
   to `simply`, then described it as if it were a `simply`-style benchmark.
8. I should have stopped and said the benchmark semantics were wrong before
   presenting the result as a fair comparison.

## What `simply` actually does

From `/mnt/carles/simply/simply/utils/sampling_lib.py` and
`/mnt/carles/simply/simply/model_lib.py`:

- `prefill_size` is a prompt-prefill budget.
- `begin_position` is `min(prefill_size, min_input_length - 1)`.
- decoding resumes from `begin_position`, not from `0`.
- the decode loop may still consume prompt tokens after prefill.
- real generation starts per row when `position + 1 >= input_len[row]`.
- `max_decode_steps` is an output budget, not "number of KV steps since token 0".

## Bottom line

The benchmark I wrote is not the benchmark you asked for.
It is a useful low-level KV replay microbenchmark, but it is not a faithful
`simply`-style prefill + decode benchmark.
