# SPLADE sparse retrieval

Goal: train a SPLADE-style sparse retriever in pure JAX, following LinkUp SparseUp
(ModernBERT backbone, `log(1 + ReLU(x - 15))`, per-position top-12, vocab folding,
contrastive loss with temperature 6 and FLOPS regularization).

Status: model + data + trainer implemented and smoke-tested on CPU. No full run yet.

Best command:

```bash
uv run python src/jaxformers/train/splade.py --config experiments/splade/base_config.py
```

Best result: none yet (see `logbook.md`).
