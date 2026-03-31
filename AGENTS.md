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

Prefer a configurable layout with one file per concrete type under its typed folder, and keep the top-level factory thin.

Rules:
- Use typed folders like `data/source/`, `data/transforms/`, and `data/loader/`.
- Put each concrete implementation in its own file, for example `data/source/huggingface.py`.
- Each concrete implementation file must expose a `make(...)` function.
- `__init__.py` factory helpers like `make_source`, `make_transforms`, and `make_loader` should only import the target module from the typed folder and call that module’s `make(...)`.
- Pass config into `make(...)` from a dict produced by the main `sws.Config`, and call it as `make(**config_xx)`.
- Do not use catch-all `**kwargs` in those concrete `make(...)` functions; extra config should fail fast with a normal Python argument error.

Intentional exceptions:
- Models are not a `make(...)` factory folder in this repo. Model targets are resolved from a string or callable and may point to a module function, classmethod, or other callable such as `Gemma3ForCausalLM.from_pretrained`.
- It is acceptable for factories to pass shared leading runtime args from `__init__.py` when those args are guaranteed across all concrete implementations, for example `datasets`, `transforms`, and `mesh` for loaders, or `scheduler` and `train_state` for optimizers.
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

## Don't
- Prefer inlining trivial helper functions (one to three lines) rather than creating a separate function. For example, instead of:
   def get_forward_impl(config: Config | PreTrainedConfig) -> str:
       return config.additional_config.get("forward_impl", ForwardImpl.LOOP)
  access the value directly:
   config.additional_config["forward_impl"]

- For enum-based control flow, validate first with `if value not in tuple(Enum): raise`, then use explicit `if` / `elif` / `elif` branches. Do not add a redundant trailing `else: raise` after that upfront validation.

- Do not rely on implicit fallthrough for enum cases or other closed sets. Even if only one case remains, spell it out as an explicit branch.

- Avoid using getattr(...) or dict.get(...) when you are certain the attribute or key exists. Use direct attribute access (obj.attr) or indexing (dict[key]) instead. Ensure those attributes or keys are initialized up front (for example in __init__ or when constructing the dict) so callers do not need to defensively probe for presence.

- Example: if additional_config is constructed from DEFAULT_ADDITIONAL_CONFIG and will always contain "forward_impl", prefer:
   additional_config["forward_impl"]
  over calling .get(...) with a fallback. Minimize use of None for required fields on Config or other classes — reserve None only for truly optional values.

- Remove dead compatibility code once the upstream contract or validation already guarantees the invariant.

- Do not add backward-compatible config fallbacks just because another train script or old config used a different shape. Match the active entrypoint exactly, update callers to that contract, and remove config fields that the target code does not use.

- Do not introduce subclass-specific hooks or override points when the base class can implement the behavior directly from `cls` and the existing shared contract. Keep the generic path in one place, and add a narrow hook only when the subclass has a real schema or mapping difference.

- Prefer minimal code paths over defensive abstractions when the caller, config, or upstream library is trusted.

- Avoid redundant type casts like int(...), float(...), or bool(...) when the value type is already guaranteed by config, validation, or the caller. Only cast when converting genuinely untyped external data or when an API explicitly requires a different type.

- If you're not really sure about a runtime behavior, inferred type, or library contract, check it in a REPL instead of guessing.


## After finishing the code

Check the diff against main and remove any AI-generated slop introduced in this branch.

The diff against main should be one of the following, in this order:
- git diff --cached
- git diff
- git diff main..HEAD or git diff master..HEAD

AI-generated slop includes:
- Extra comments that a human wouldn't add or that are inconsistent with the rest of the file.
- Extra defensive checks or try/catch blocks that are abnormal for that area of the codebase (especially if called by trusted/validated code paths).
- Variables or functions that are used only once immediately after declaration — prefer inlining the right-hand side or the function.
- Redundant checks or casts inside a function that the caller already performs.
- Any other style that is inconsistent with the file, including adding type annotations where the file does not use them.
- Changes that are inconsistent with AGENTS.md requirements.
- Code that should have been removed but was retained for "legacy compatibility" or similar reasons.

At the end, include only a 1-3 sentence summary of what you changed
