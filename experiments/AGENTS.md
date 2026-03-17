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
  config/
    base.py
    model.py
    data.py
    optimizer.py
    scheduler.py
    train.py
  <task-name>_config.py
  sweeps/
```

Rules:
- `README.md` is the short human-facing summary: goal, current status, best command, current best result.
- `logbook.md` is append-only: date, command, config, result, interpretation, next step.
- `config/base.py` holds shared runtime and experiment metadata such as `exp_name`, `project_name`, `dir`, `ckpt_path`, `seed`, `max_train_step`, `logger_name`, `checkpoint_options`, and other run-level defaults.
- `config/model.py` holds only `model_name`, `init_model`, `init_lora`, and `model.*` or `lora.*`.
- `config/data.py` holds `data.source_name`, `data.source.*`, `data.transforms_name`, `data.transforms.*`, `train_loader_name`, and `train_loader.*`.
- `config/optimizer.py` holds `optimizer_name` and `optimizer.*`.
- `config/scheduler.py` holds `learning_rate`, `lr_scheduler_name`, and `lr_scheduler.*`.
- `config/train.py` is optional. Use it only for train-script-specific knobs such as `loss_ratio.*` for KL runs.
- `<task-name>_config.py` is the config entrypoint passed to `--config`. It should combine the files from `./config`.
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
- There is no separate `src/jaxformers/data/train_loader/` factory today. The config still uses `train_loader_name` and `train_loader.*`, but the implementation lives under `src/jaxformers/data/loader/`.
- Reuse the existing train script and transform pair when possible instead of cloning training code into `experiments/`.

Examples:
- Plain SFT or next-token prediction: use `src/jaxformers/train/ntp.py` with `data.transforms_name = "ntp"`.
- SFT with KL regularization: use `src/jaxformers/train/ntp_with_kl_regularzier.py` with `data.transforms_name = "ntp_kl"`.

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
- If the only difference is dataset path, tokenizer, max length, LoRA rank, batch size, or scheduler settings, stay in `experiments/<task-name>/config/`.
- If multiple experiments would reuse the same dataset preparation logic, add a transform or loader in `src/jaxformers/data/...`.
- If multiple experiments would reuse the same optimization or LR logic, add an optimizer or scheduler module in `src/jaxformers/...`.
- If the loss/training step changes in a real way, add or extend a train entrypoint in `src/jaxformers/train/`.

## Config Composition

Use Python config builders that return `sws.Config`, matching the rest of the repo.

Prefer a task entrypoint like:

```python
from pathlib import Path

from jaxformers.sws_utils import merge_config_builders


CONFIG_PATHS = [
    str(Path(__file__).parent / "config" / "base.py"),
    str(Path(__file__).parent / "config" / "model.py"),
    str(Path(__file__).parent / "config" / "data.py"),
    str(Path(__file__).parent / "config" / "optimizer.py"),
    str(Path(__file__).parent / "config" / "scheduler.py"),
]


def get_config():
    return merge_config_builders(CONFIG_PATHS)
```

Rules:
- Keep the main `--config` entrypoint in `experiments/<task-name>/`.
- Put reusable per-task config pieces under `experiments/<task-name>/config/`.
- Add `config/train.py` to `CONFIG_PATHS` only when the chosen train script needs extra config fields.
- Use `merge_config_builders(...)` to combine config builders.
- If you need a generated frozen config file, use `jaxformers.sws_utils.combine_and_write(...)`, but keep the output inside the same task folder.

## Run Workflow

For each new task:
1. Create or switch to the branch `experiments/<task-name>`.
2. Create `experiments/<task-name>/`.
3. Write `README.md` with the goal and the baseline command.
4. Decide whether the task fits an existing train script and factory pieces.
5. Build `config/*.py`.
6. Create `<task-name>_config.py`.
7. Run the chosen train script with that config.
8. Append results to `logbook.md`.

Typical commands:

```bash
uv python src/jaxformers/train/ntp.py --config experiments/<task-name>/<task-name>_config.py
```

```bash
uv python src/jaxformers/train/ntp_with_kl_regularzier.py --config experiments/<task-name>/<task-name>_config.py
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
- `experiments/<task-name>/config/*.py`
- `experiments/<task-name>/<task-name>_config.py`

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
