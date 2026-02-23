# Cumulative metrics: `cum_token_count` and `global_loss`

This repo’s NTP trainers (`src/jaxformers/train/ntp.py`, `src/jaxformers/train/ntp_lora.py`) currently compute **per-emit** metrics (i.e., per optimizer step, after grad-accum). If you also want **global** (entire-training) metrics like:

- `cum_token_count`: total non-pad tokens processed so far
- `global_loss`: running average loss over *all* tokens seen so far

…you don’t need to store every step’s `aux`. Keep *streaming accumulators* (O(1) memory) and log derived values each emit.

---

## Key idea: store means as `(numerator, denominator)`

Never try to “average averages” without weights.

For token-level means (loss, accuracy, etc.), keep:

- `loss_sum`: sum of per-token losses
- `token_count`: number of valid tokens (e.g., `attention_mask.sum()`)

Represent the metric as a tuple:

```py
aux = {
  "loss": (loss_sum, token_count),   # mean = loss_sum / token_count
  "token_count": token_count,        # sum metric
}
```

That’s already what your `train_step()` does.

---

## Two accumulators: emit-window + global

You typically want **both**:

1. **Emit/window metrics**: “What happened this optimizer step?” (good for short-term behavior)
2. **Cumulative/global metrics**: “Since training started, what’s the running average?” (good for long-term progress)

### Emit/window accumulator (for grad-accum)

Instead of storing `accum_aux: list[dict]`, you can stream-reduce across micro-steps:

- For `"sum"` metrics: add values.
- For `"mean"` metrics stored as tuples: add numerators and denominators.

At emit:

- `emit_token_count = emit_loss_tuple[1]`
- `emit_loss = emit_loss_tuple[0] / emit_loss_tuple[1]`

Then reset the emit accumulator to zero.

### Global accumulator (across the entire training)

Maintain the same raw structure globally:

- `cum_loss = (cum_loss_sum, cum_token_count)`
- `cum_token_count = cum_token_count`

At each emit:

```text
cum_loss_sum     += emit_loss_sum
cum_token_count  += emit_token_count
global_loss       = cum_loss_sum / cum_token_count
```

This gives you a true running, token-weighted loss across the whole pipeline.

---

## Minimal implementation sketch

The pattern below avoids storing per-step `aux` and lets you log both emit + global metrics each emit.

```py
def aux_add(a, b):
    # a and b have the same pytree structure.
    def add_leaf(x, y):
        if isinstance(x, tuple):
            return (x[0] + y[0], x[1] + y[1])
        return x + y
    return jtu.tree_map(add_leaf, a, b, is_leaf=lambda v: isinstance(v, tuple))

def aux_reduce_for_logging(aux_raw):
    # Convert raw (sum,count) tuples to means for display.
    def reduce_leaf(x):
        if isinstance(x, tuple):
            return x[0] / x[1]
        return x
    return jtu.tree_map(reduce_leaf, aux_raw, is_leaf=lambda v: isinstance(v, tuple))

emit_raw = {"loss": (0.0, 0.0), "token_count": 0.0}
cum_raw  = {"loss": (0.0, 0.0), "token_count": 0.0}

for micro_step in ...:
    model, aux = train_step_fn(...)
    emit_raw = aux_add(emit_raw, aux)

    if emit:
        cum_raw = aux_add(cum_raw, emit_raw)

        log_emit = aux_reduce_for_logging(emit_raw)
        log_cum  = aux_reduce_for_logging(cum_raw)

        wandb.log(
            {
                "loss": log_emit["loss"],
                "token_count": log_emit["token_count"],
                "cum/loss": log_cum["loss"],
                "cum/token_count": log_cum["token_count"],
            },
            step=step,
        )

        emit_raw = {"loss": (0.0, 0.0), "token_count": 0.0}
```

Notes:

- Use `token_count` as the denominator for any token-level mean, so it stays correct when sequence lengths vary.
- If you want to keep things JAX-friendly, make the zeros `jnp.zeros((), dtype=...)` and ensure `token_count` uses an integer dtype (`int32`/`int64`) as desired.

---

## Checkpointing (important for “global”)

If you resume training and want `cum_token_count/global_loss` to continue from before, you must checkpoint `cum_raw` (or at least `cum_loss_sum` + `cum_token_count`) alongside `weights/opt_state/step`.

Otherwise, any “global” metric is only global *since the current run started*.

---

## Multi-host / multi-process caveat

If you’re training with multiple JAX processes (multiple hosts), each process will compute its own local `token_count/loss_sum`.

To log true global counts/losses (across all hosts), you need to **sum across processes** before updating `cum_raw` (an all-reduce). The exact mechanism depends on whether your training uses `pmap` axes or mesh-based SPMD, but the invariant is:

- Reduce `loss_sum` and `token_count` across all data-parallel workers **once per emit**.
- Only then update and log `cum_raw` (usually from process 0).

---

## Quick fix in the existing `process_metrics`

If you keep the current `accum_aux: list[...]` approach and set `grad_accum > 1`, make sure sum-reductions actually sum over the list. In `ntp.py` / `ntp_lora.py`, `np.sum(*value)` only works accidentally when `len(value) == 1`.

Prefer something like:

```py
elif method == "sum":
    return sum(value)
```

