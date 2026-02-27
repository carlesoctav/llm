# Refactor plan: NTP bench + sweep (sws.config + grouped sweeps)

## Goals
- Replace `argparse` in the NTP bench + sweep scripts with `sws.Config`-style configs.
- Introduce `ntp_config_path` as the “starting point” for the underlying NTP config builder.
- Make *all* knobs configurable (bench settings + any NTP config fields).
- Add a sweep mechanism where list/tuple values become sweep parameters.
- Add *grouped* sweep parameters (a single “choice” that sets multiple keys together), so
  total runs = `len(group) * Π(len(free_sweep_param_values))`.

## Steps
1. **Inventory current behavior**
   - Read `src/jaxformers/bench/bench_ntp_loss_impl.py` + `src/jaxformers/bench/sweep_ntp_loss_impl.py` to capture current CLI, defaults, and report format.

2. **Shared helpers**
   - Add a small helper module for:
     - loading `get_config()` builders from a python file path (same `path.py[:func]` convention as `sws.run`)
     - parsing `key=value` / `key:=value` tokens into python values (using `sws.simpleeval`)
     - flattening nested dict overrides into dotted keys

3. **Bench refactor**
   - Replace `argparse` with `sws.Config` for bench-only settings (`warmup_steps`, `timed_steps`, `ntp_config_path`, …).
   - Treat any “unknown” `key=value` tokens as *NTP config overrides* (applied to the NTP builder loaded from `ntp_config_path`).
   - Stop hard-coding `sgd`; use `config.optimizer_name` so optimizer sweeps/grouping actually affect the run.

4. **Sweep design + implementation**
   - Load the base NTP builder from `ntp_config_path` (without finalizing/evaluating lazies).
   - Build sweep dimensions from overrides:
     - a value is a sweep dimension when it’s a `list`/`tuple` **and** the base NTP default for that key is *not* a `list`/`tuple`
       (prevents accidentally sweeping “intrinsic list” fields like `lora.weights_path` or `model.devices`).
   - Implement `group`:
     - `group` is a list of dicts; each dict is flattened to dotted keys and applied together
     - keys set by any `group[i]` are excluded from “free” sweep dimensions
     - run space becomes `choices(group) × cartesian_product(free_sweep_dims)`

5. **Sweep execution**
   - Update sweep runner to call the bench runner with:
     - bench meta args (`ntp_config_path`, `warmup_steps`, `timed_steps`, …)
     - per-run NTP overrides (group choice + free sweep choices + fixed overrides)
   - Keep the current “BENCH_RESULT …json…” contract and report generation, but add enough metadata per run to reproduce the override set.

6. **Tests**
   - Add unit tests for sweep expansion:
     - list/tuple → sweep dim only when base default isn’t list/tuple
     - grouped parameters reduce cross-product as intended
     - flattening nested dicts works (`{"optimizer": {"max_grad_norm": 1.0}}` → `optimizer.max_grad_norm=…`)

7. **Docs**
   - Add short usage examples (CLI + config snippets) directly in the sweep script docstring or in a small `README` section.

