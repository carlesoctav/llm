# Experiments Folder Rules

Use `experiments/` for experiment work. The canonical location for a new task is:

```text
experiments/<task-name>/
```

Do not create `src/jaxformers/experiments/`. Keep experiment configs, notes, generated sweep configs, and small task-local helpers under the root `experiments/<task-name>/` folder.

At kickoff, create or switch to a git branch named:

```text
experiments/<task-name>
```

If the user explicitly asks for a different branch name, follow that instead.

## Default Layout

For a new experiment, prefer:

```text
experiments/<task-name>/
  README.md
  logbook.md
  base_config.py
  config/
    lora_rank256.py
    adamw.py
    short_run.py
  sweeps/
```

Rules:
- `README.md` is the short human-facing summary: goal, current status, best command, current best result.
- `logbook.md` is append-only: date, command, config, result, interpretation, next step.
- `base_config.py` holds the readable task default in one file: run metadata, model, data, optimizer, scheduler, and train defaults together.
- Additional root base configs are acceptable when a task has a few genuinely different baselines and separate root files are easier to navigate than stacking many overrides.
- `config/` is optional and holds small override fragments for variants or extensions.
- Name override files by intent, for example `adamw.py`, `lora_rank256.py`, or `short_run.py`.
- Do not split the default task config into one file per factory by default. Keep the common config easy to scan in `base_config.py`.
- Pass the root `base_config.py` first to `--config`, then add `config/*.py` overrides after it when needed.
- `sweeps/` is optional. If you generate sweep configs, keep them inside the same task folder.

## Reuse First

Before adding new code, inspect the existing reusable pieces and decide whether the experiment can be expressed by config only.

Current reusable factory areas in this repo:

- Train entrypoints: `src/jaxformers/train/`
- Model factories and model modules: `src/jaxformers/models/`
- Hugging Face-backed model modules: `src/jaxformers/models/huggingface/`
- Optimizer factory modules: `src/jaxformers/optimizers/`
- Data source factory modules: `src/jaxformers/data/source/`
- Data transform factory modules: `src/jaxformers/data/transforms/`
- Data loader factory modules: `src/jaxformers/data/loader/`
- Scheduler factory modules: `src/jaxformers/scheduler/`

Important:
- The loader implementation lives under `src/jaxformers/data/loader/`, and the current training configs should use global loader settings like `config.data.loader.{combine,shard,...}` plus per-dataset loader settings like `config.data.<name>.loader.{batch_size,shuffle,...}`.
- Reuse the existing train script and transform pair when possible instead of cloning training code into `experiments/`.

Examples:
- Plain SFT or next-token prediction: use `src/jaxformers/train/ntp.py` with a single dataset group such as `config.data.train.{source,transforms,loader}` plus global settings in `config.data.loader`.
- SFT with KL regularization: use `src/jaxformers/train/ntp_with_kl_regularzier.py` with `config.data.sft.{source,transforms,loader}`, `config.data.kl.{source,transforms,loader}`, and global settings in `config.data.loader`.

## When To Add New Factory Code

If the experiment cannot be expressed cleanly with the existing pieces, add the missing reusable piece under `src/jaxformers/...` instead of hardcoding it in `experiments/`.

Order of preference:
1. Reuse existing config only.
2. Add a new concrete factory implementation under an existing typed folder.
3. Add a new train script only if the training loop is materially different.

Factory rules in this repo:
- Read the neighboring factory examples first and match their style.
- Put each concrete implementation in its own file.
- For data/source, data/transforms, data/loader, optimizer, and scheduler modules, expose a `make(...)` entrypoint unless the domain already uses a different pattern.
- Keep `__init__.py` factories thin: import the concrete module and call it.
- Pass config into factories from `sws.Config` via `.to_dict()`.
- Use plain strings in configs like `"huggingface"`, `"ntp"`, `"simple"`, `"adam"`, `"wsds"`, and compare enum values with `==` inside code.
- Do not add backward-compat wrappers or loose `**kwargs` just to make configs more permissive.

Rules of thumb:
- If the only difference is dataset path, tokenizer, max length, LoRA rank, batch size, or scheduler settings, keep it in `base_config.py` or a small override under `experiments/<task-name>/config/`.
- If multiple experiments would reuse the same dataset preparation logic, add a transform or loader in `src/jaxformers/data/...`.
- If multiple experiments would reuse the same optimization or LR logic, add an optimizer or scheduler module in `src/jaxformers/...`.
- If the loss/training step changes in a real way, add or extend a train entrypoint in `src/jaxformers/train/`.

## Config Composition

Use Python config builders that return `sws.Config`, matching the rest of the repo.

Prefer a base config plus optional override files:

```python
import sws


def get_config():
    config = sws.Config()
    ...
    return config
```

Rules:
- Keep the main config path in `experiments/<task-name>/base_config.py`.
- If a task needs more than one real baseline, multiple root base config files are acceptable.
- Put optional override fragments under `experiments/<task-name>/config/`.
- Prefer composing configs at the CLI with multiple `--config` paths instead of creating a merged Python entrypoint by default.
- The first config path should usually be the root `base_config.py`, followed by any `config/*.py` overrides.
- If you want a named root shortcut config file for a common variant, add it only when it improves usability.
- If you need a generated frozen config file, use `jaxformers.sws_utils.combine_and_write(...)`, but keep the output inside the same task folder.

## Run Workflow

For each new task:
1. Create or switch to the branch `experiments/<task-name>`.
2. Create `experiments/<task-name>/`.
3. Write `README.md` with the goal and the baseline command.
4. Decide whether the task fits an existing train script and factory pieces.
5. Write `base_config.py`.
6. Add `config/*.py` only for overrides or variants.
7. Run the chosen train script with `--config experiments/<task-name>/base_config.py ...`.
8. Append results to `logbook.md`.

Typical commands:

```bash
uv python src/jaxformers/train/ntp.py --config experiments/<task-name>/base_config.py
```

```bash
uv python src/jaxformers/train/ntp_with_kl_regularzier.py --config experiments/<task-name>/base_config.py experiments/<task-name>/config/<variant>.py
```

If you use `src/jaxformers/train/make_sweep.py`, write the generated configs into `experiments/<task-name>/sweeps/`.

## Weights & Biases

Use a simple rule here:

- If the run should be tracked in W&B, set `logger_name = "wandb"`.
- If the run is local or disposable, set `logger_name = "noop"`.
- Keep runs that need direct comparison in the same W&B project.
- Decide the W&B project early instead of splitting the same task across multiple projects.
- Set `logger.project` and `logger.name` from the task config, usually from `project_name` and `exp_name`.
- Add the W&B run URL to `README.md` or `logbook.md` when the run matters.

## Simple Research Flow

Keep the workflow lightweight and local to the task folder. Do not create extra issue-tracking machinery unless the user explicitly asks for it.

Minimum required artifacts per task:
- `experiments/<task-name>/README.md`
- `experiments/<task-name>/logbook.md`
- `experiments/<task-name>/base_config.py`

Optional:
- `experiments/<task-name>/config/*.py`
- `experiments/<task-name>/*_config.py`

What to record in `logbook.md`:
- date/time
- exact command
- important config values
- result or failure
- short interpretation
- next action

Keep everything reproducible:
- record the exact config path you ran
- keep generated configs and small helpers inside the same task folder
- do not commit large artifacts or checkpoints into git; store paths or links instead
- prefer updating the task README with the current best run rather than scattering notes elsewhere

## Avoid

- Do not put new experiment configs at the top level of `experiments/`; always create a dedicated `experiments/<task-name>/` folder.
- Do not duplicate train scripts inside `experiments/`.
- Do not add one-off code to `src/` before checking whether the current factories already cover the use case.
- Do not create a new factory module when the change belongs in config.
- Do not create compatibility layers for old config names when a new experiment can just use the correct current names.
- Do not default to `config/base.py`, `config/model.py`, `config/data.py`, and similar one-file-per-factory splits for every task.
- Do not hide the main runnable config inside `experiments/<task-name>/config/`; keep the base config in the task root.
