# Repo Rules

## StrEnum Usage

For `StrEnum`, treat the enum as a comparison helper, not as the public data shape.

Rules:
- In config and other external/public inputs, prefer plain strings like `"chat"` or `"loop"`.
- Use the `StrEnum` members only for validation and comparison inside the code.
- Compare `StrEnum` values with `==`, not `is`.
- Do not coerce user/config values with `Enum(value)` just to normalize them.
- Avoid backward-compat conversion helpers when the caller should already provide the correct string value.

## Typed Factory Layout

Prefer a configurable layout with one file per concrete type under its typed folder.

Rules:
- Use folders like `data/source/`, `data/transforms/`, and `data/loader/`.
- Put each concrete implementation in its own file, for example `data/source/huggingface.py`.
- Each concrete implementation file must expose a `make(...)` function.
- `__init__.py` factory helpers like `make_source`, `make_transforms`, and `make_loader` should only:
  1. import the target module from the typed folder
  2. call that module’s `make(...)`
- Pass config into `make(...)` from a dict produced by the main `sws.Config`, and call it as `make(**config_xx)`.
- Do not use catch-all `**kwargs` in those concrete `make(...)` functions; extra config should fail fast with a normal Python argument error.

Intentional exceptions:
- It is acceptable for some factories to use a different concrete entrypoint name when the domain needs it, for example model modules using `.load(...)` or `.init(...)` instead of `.make(...)`.
- It is acceptable for scheduler and optimizer factories to pass shared leading runtime args from `__init__.py` when those args are guaranteed across all concrete implementations, for example `learning_rate` and `num_train_steps` for schedulers, or `scheduler` and `model` for optimizers.
- It is acceptable to keep small built-in cases in `__init__.py` when a dedicated file would be pointless, for example a constant scheduler that is just `learning_rate`.
- Chained factories are expected in places like callbacks: `config.xx_name` may be either a single string or a list of names, and the per-item config should live under `config.xx[name]` after `config.xx.to_dict()`.

## Jaxtyping Shape Strings

In type annotations, quoted strings are used for two different things:
- regular strings
- forward references to types defined later

Some Python tooling assumes quoted annotations are only forward references and
produces false positives for jaxtyping-style shape strings.

Rules:
- Multi-dimensional shape annotations like `Float32[jax.Array, "b c"]` may raise
  `F722` (`syntax error in forward annotation`). This error is expected for
  jaxtyping shape strings and should be disabled globally in flake8 or Ruff.
- Single-dimensional shape annotations like `Float32[jax.Array, "x"]` may raise
  `F821` (`undefined name`). To avoid that, prepend a leading space and write
  them as `Float32[jax.Array, " x"]`.
- Jaxtyping treats `"x"` and `" x"` the same way, so prefer the leading-space
  form for single-axis shapes in this repo.
