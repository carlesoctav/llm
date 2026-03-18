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


def _move_scan_axis(tree, axis: int):
    return jax.tree.map(lambda leaf: jnp.moveaxis(leaf, axis, 0), tree)


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
            raise ValueError(
                f"Missing scanned keyword arguments: {missing_argnames!r}."
            )

        scan_values = [fwd_args[idx] for idx in argnums] + [
            kwargs[name] for name in argnames
        ]
        lengths = _get_scan_lengths(scan_values, in_axes)
        if len(set(lengths)) != 1 or lengths[0] != length:
            raise ValueError(
                f"Expected scanned inputs to have length {length}, got {lengths!r}."
            )

        base_args = list(fwd_args)
        scan_args = tuple(
            _move_scan_axis(fwd_args[idx], axis)
            for idx, axis in zip(argnums, in_axes[: len(argnums)])
        )
        scan_kwargs = {
            name: _move_scan_axis(kwargs[name], axis)
            for name, axis in zip(argnames, in_axes[len(argnums) :])
        }
        nonscan_kwargs = {
            name: value for name, value in kwargs.items() if name not in argnames
        }

        def scan_body(carry, xs):
            step_args, step_kwargs = xs
            call_args = list(base_args)
            for idx, value in zip(argnums, step_args):
                call_args[idx] = value

            call_kwargs = dict(nonscan_kwargs)
            call_kwargs.update(step_kwargs)
            return fwd(carry, *call_args, **call_kwargs), None

        carry, _ = jax.lax.scan(
            scan_body,
            init=carry,
            xs=(scan_args, scan_kwargs),
            length=length,
        )
        return carry

    return scan_fwd
