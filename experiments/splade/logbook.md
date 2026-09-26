# Logbook

## 2026-09-26: CPU smoke run (2 steps)

Command:

```bash
uv run python src/jaxformers/train/splade.py --config experiments/splade/base_config.py /tmp/splade_smoke_override.py
```

Config: base plus override (batch 2, query/doc lengths 16/32, float32, eager
attention, no remat, 8 MS MARCO triplets, checkpoint to `/tmp/splade-smoke`).

Result: both steps ran. `step/loss=0.116 -> 3.32`, `step/nll=0.00479 -> 3.2`,
`step/flops=0.112`. Frozen params are exactly the fold index (0.2 MB), so the
non-float train mask works. Compile took 22s on CPU.

Interpretation: pipeline (load SparseUp weights, tokenize triplets, contrastive
step, optimizer update, checkpoint) works end to end. Loss values are
meaningless at this scale.

Next: run the full base config on an accelerator; add a retrieval eval.
