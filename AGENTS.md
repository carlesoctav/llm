# Agent Preferences

## Refactor Style

- Put package-level factory entry points in package `__init__.py` files.
- Keep per-module constructors like `load(...)`, `init(...)`, and similar module APIs unchanged unless explicitly requested.
- Prefer `make_*` names for package-level factories.
- Use `make_model(...)` directly in train scripts instead of extra wrapper helpers like `make_model_from_config(...)`.
- For LoRA, keep the enum and dispatch logic in `src/jaxformers/dispatch/lora.py` and make it mirror the style of `src/jaxformers/models/__init__.py`.

## Config Handling

- Do not add backward compatibility layers unless explicitly requested.
- Remove old config paths instead of supporting both old and new behavior.
- Do not pass full config objects into package-level `make_*` functions when specific arguments are enough.
- Pass the exact required arguments into factories, for example:
  - optimizer: `optimizer_name`, `model`, `scheduler`, `optimizer_config`
  - callbacks: `callback_specs`
  - data: explicit data and loader config objects
- Do not use `getattr(...)` for required config fields in this refactor path. Access them directly and let missing fields fail.
- Avoid helper functions like `_as_dict`, `_resolve_*`, `_make_source`, `make_*_from_config`, or similar thin wrappers when a single direct function is enough.

## LoRA

- `init_lora` should support `None`, `"random"`, and `"pytree"`.
- Default `init_lora` is `None`.
- Skip LoRA entirely when `init_lora` is `None`.
- Do the `init_lora` handling inside `make_lora(...)`, not in the training scripts.
- For `"pytree"`, raise `NotImplementedError` for now.

## Callbacks

- Prefer a single `make_callbacks(...)` function instead of multiple thin callback factory helpers.
- When processing callbacks in training, gate on `callback_state` only; if callback state exists, callback functions are assumed to exist.

## API Cleanliness

- Do not keep alias compatibility like `load = make_*` or `load_* = make_*` unless explicitly requested.
- Prefer direct, explicit code over defensive compatibility code.
