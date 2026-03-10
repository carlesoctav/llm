import jax.numpy as jnp

from jaxformers.scan_utils import make_scan_fwd


def test_make_scan_fwd_uses_explicit_scanned_args_and_kwargs():
    def fwd(carry, weights, *, bias, scale):
        return carry + weights * scale + bias

    scan_fwd = make_scan_fwd(
        fwd,
        length=3,
        argnums=0,
        argnames="scale",
    )

    out = scan_fwd(
        jnp.asarray(0),
        jnp.asarray([1, 2, 3]),
        bias=jnp.asarray(5),
        scale=jnp.asarray([1, 10, 100]),
    )

    assert int(out) == 336


def test_make_scan_fwd_respects_in_axes():
    def fwd(carry, weights, *, bias):
        return carry + jnp.sum(weights) + bias

    scan_fwd = make_scan_fwd(
        fwd,
        length=3,
        argnums=0,
        argnames="bias",
        in_axes=(1, 0),
    )

    out = scan_fwd(
        jnp.asarray(0),
        jnp.asarray([[1, 2, 3], [4, 5, 6]]),
        bias=jnp.asarray([10, 20, 30]),
    )

    assert int(out) == 81
