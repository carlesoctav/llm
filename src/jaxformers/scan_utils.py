from typing import Sequence

import jax
import jax.numpy as jnp


def _as_tuple(value: int | str | Sequence[int] | Sequence[str]):
    if isinstance(value, (int, str)):
        return (value,)
    return tuple(value)


def _get_scan_lengths(values, axes):
    lengths = []
    for value, axis in zip(values, axes):
        lengths.extend(
            jax.tree.leaves(jax.tree.map(lambda leaf: leaf.shape[axis], value))
        )
    return lengths


def _take_scan_slice(tree, index: int, axis: int):
    return jax.tree.map(lambda leaf: jnp.take(leaf, indices=index, axis=axis), tree)


def make_scan_fwd(
    fwd,
    length: int,
    argnums: int | Sequence[int],
    *,
    argnames: str | Sequence[str] = (),
    in_axes: int | Sequence[int] = 0,
):
    argnums = _as_tuple(argnums)
    argnames = _as_tuple(argnames)

    if not argnums and not argnames:
        raise ValueError("make_scan_fwd requires at least one scanned argument.")

    if isinstance(in_axes, int):
        in_axes = (in_axes,) * (len(argnums) + len(argnames))
    else:
        in_axes = tuple(in_axes)

    if len(in_axes) != len(argnums) + len(argnames):
        raise ValueError(
            "`in_axes` must align with the scanned positional and keyword arguments."
        )

    def scan_fwd(*args, **kwargs):
        if not args:
            raise TypeError("make_scan_fwd expects the carry as positional argument 0.")

        carry, *fwd_args = args

        if max(argnums, default=-1) >= len(fwd_args):
            raise ValueError("A scanned positional argnum is out of range.")

        missing_argnames = [name for name in argnames if name not in kwargs]
        if missing_argnames:
            raise ValueError(f"Missing scanned keyword arguments: {missing_argnames!r}.")

        scan_values = [fwd_args[idx] for idx in argnums] + [kwargs[name] for name in argnames]
        lengths = _get_scan_lengths(scan_values, in_axes)
        if len(set(lengths)) != 1 or lengths[0] != length:
            raise ValueError(
                f"Expected scanned inputs to have length {length}, got {lengths!r}."
            )

        base_args = list(fwd_args)
        nonscan_kwargs = {
            name: value for name, value in kwargs.items() if name not in argnames
        }
        arg_axes = tuple(in_axes[: len(argnums)])
        kw_axes = tuple(in_axes[len(argnums) :])

        def body_fun(index, carry):
            call_args = list(base_args)
            for idx, axis in zip(argnums, arg_axes):
                call_args[idx] = _take_scan_slice(fwd_args[idx], index, axis)

            call_kwargs = dict(nonscan_kwargs)
            for name, axis in zip(argnames, kw_axes):
                call_kwargs[name] = _take_scan_slice(kwargs[name], index, axis)
            return fwd(carry, *call_args, **call_kwargs)

        return jax.lax.fori_loop(0, length, body_fun, carry)

    return scan_fwd
