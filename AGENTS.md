# Repo Rules

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
