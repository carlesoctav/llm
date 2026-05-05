from __future__ import annotations

from collections.abc import Callable

import jax


def get_compiled(
    cache: dict[str, Callable],
    key: str,
    fn: Callable,
    *,
    static_argnames: tuple[str, ...] = (),
    donate_argnums: tuple[int, ...] = (),
    donate_argnames: tuple[str, ...] = (),
) -> Callable:
    compiled = cache.get(key)
    if compiled is not None:
        return compiled

    compiled = jax.jit(
        fn,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        donate_argnames=donate_argnames,
    )
    cache[key] = compiled
    return compiled
