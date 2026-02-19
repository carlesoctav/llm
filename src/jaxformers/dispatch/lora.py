"""
copied from quax.examples.lora
"""

import fnmatch
import time
from dataclasses import replace
from typing import cast

import equinox as eqx
import jax
import jax.core
import jax.extend.core as jexc
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import quax
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree, Shaped

from jaxformers.modeling_utils import Model


class LoraArray(quax.ArrayValue):
    """Replaces a matrix `w in R^{n x m}` with `w + a @ b`, where `a in R^{n x k}` and
    `b in R^{k x m}`.

    Typically `k` is much smaller than `n` or `m`, and so `w + a @ b` is described as a
    "low rank adaptation" of `w`. The value of `k` is the "rank" of the adaptation.

    Note that this does not materialise the sum `w + a @ b` into a single matrix, but
    instead stores it as three separate `w`, `a`, `b` matrices. This is because the
    typical use-case for LoRA is to update just the `a` and `b` matrices when
    fine-tuning a neural network.

    This implementation makes use of Quax's multiple-dispatch capabilities to calculate
    matrix-vector products `(w + a @ b) @ x` via `w @ x + a @ (b @ x)`, which turns out
    to be computationally cheaper.
    """

    _w: Shaped[Array, "*batch x y"]
    a: Shaped[Array, "*batch x z"]
    b: Shaped[Array, "*batch z y"]
    alpha: float = eqx.field(static=True)
    stop_gradient: bool = eqx.field(static=True)
    allow_materialise: bool = eqx.field(static=True)

    def __init__(
        self,
        weight: Shaped[Array, "*batch x y"],
        *,
        rank: int,
        alpha: float | None = None,
        scale: float = 0.01,
        allow_materialise: bool = False,
        stop_gradient: bool = True,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `weight`: the original weight to wrap.
        - `rank`: the rank of the low-rank adaptation.
        - `scale`: `a` will be initialised at `Normal(0, scale^2)`. (`b` is initialised
            at zero.)
        - `allow_materialise`: if Quax encounters an operation for which there has not
            been a specific override specified for LoraArrays, should it either (a)
            throw an error (`allow_materialise=False`, the default), or (b) silently
            convert the `LoraArray` back into an JAX array, by explicitly calculating
            `w + a @ b` (`allow_materialise=True`).
        - `stop_gradient`: whether to automatically stop the gradient (prevent training)
            of the original weight matrix `weight`.
        - `key`: used to provide randomness for initialising `a`.
        """

        *batch, x, y = weight.shape
        if alpha is None:
            # Common LoRA default; makes the effective scaling `alpha / rank == 1`.
            alpha = float(rank)
        self._w = weight
        self.a = jr.normal(key, (*batch, x, rank), dtype=weight.dtype) * scale
        self.b = jnp.zeros((*batch, rank, y), dtype=weight.dtype)
        self.alpha = alpha
        self.stop_gradient = stop_gradient
        self.allow_materialise = allow_materialise

    @property
    def w(self):
        if self.stop_gradient:
            return lax.stop_gradient(self._w)
        else:
            return self._w

    def materialise(self):
        if self.allow_materialise:
            batch = tuple(range(self.w.ndim - 2))
            lhs_contract = (self.a.ndim - 1,)
            rhs_contract = (self.b.ndim - 2,)
            dimension_numbers = ((lhs_contract, rhs_contract), (batch, batch))
            scaling = jnp.asarray(self.alpha / self.a.shape[-1], dtype=self.w.dtype)
            return self.w + scaling * lax.dot_general(self.a, self.b, dimension_numbers)
        else:
            raise RuntimeError(
                "Refusing to materialise `LoraArray` with `allow_materialise=False`."
            )

    def aval(self):
        # `aval()` must be pure shape/dtype introspection.
        # Avoid calling `self.w` here, as that may bind JAX primitives like
        # `stop_gradient` while JAX is computing abstract values, which can trigger
        # tracer-leak errors under nested transforms (e.g. `jit(value_and_grad(...))`).
        return jax.core.ShapedArray(self._w.shape, self._w.dtype)


# `jax.remat` / `jax.checkpoint` lowers to a call primitive (`remat2`). If it is
# encountered inside a `quax.quaxify(...)` context, then its body will have been
# traced without Quax's multiple-dispatch active (because `remat` traces via
# `make_jaxpr`). Register a rule that re-traces the body with Quax enabled, while
# preserving the `remat` primitive so gradient checkpointing still works.
try:  # pragma: no cover
    from jax._src.ad_checkpoint import (
        remat_p as _remat_p,  # pyright: ignore[reportPrivateImportUsage]
    )
except Exception:  # pragma: no cover
    _remat_p = None


if _remat_p is not None:  # pragma: no cover

    @quax.register(_remat_p)
    def _remat_quax(*args, jaxpr, prevent_cse, differentiated, policy):
        # Quaxify the remat body (represented as a jaxpr) and then re-stage it as a
        # remat call over the *flattened* leaves, so Value instances like `LoraArray`
        # are passed in as ordinary JAX leaves to the remat body.
        # `remat_p` carries an *unclosed* `core.Jaxpr`; wrap it as a `ClosedJaxpr`
        # with no consts so we can reuse `jaxpr_as_fun`.
        body = quax.quaxify(jexc.jaxpr_as_fun(jexc.ClosedJaxpr(jaxpr, ())))

        in_leaves, in_treedef = jtu.tree_flatten(args)

        def _flat_body(*flat_args):
            return body(*jtu.tree_unflatten(in_treedef, flat_args))

        closed = jax.make_jaxpr(_flat_body)(*in_leaves)

        # `remat_p` expects a jaxpr with no constvars; move any consts into explicit
        # leading invars and pass the corresponding values as leading arguments.
        from jax._src.interpreters import (
            partial_eval as pe,  # pyright: ignore[reportPrivateImportUsage]
        )

        closed_no_constvars, consts = pe.separate_consts(closed)

        # If we were given a per-arg `prevent_cse` mask, it no longer matches after
        # flattening Values; default to the conservative option (prevent all CSE).
        prevent_cse2 = prevent_cse if isinstance(prevent_cse, bool) else True

        return _remat_p.bind(
            *consts,
            *in_leaves,
            jaxpr=closed_no_constvars.jaxpr,
            prevent_cse=prevent_cse2,
            differentiated=differentiated,
            policy=policy,
        )


def _is_match(array_path, weights_path):
    for weight_path in weights_path:
        if fnmatch.fnmatchcase(array_path, weight_path):
            return True
    return False


def loraify(
    model: Model,
    weights_path: list[str],
    rank: int,
    alpha: float,
    scale: float = 0.01,
    allow_materialise: bool = False,
    stop_gradient: bool = True,
    *,
    rngs: PRNGKeyArray,
) -> PyTree:
    counter = 0
    loraify_weight = []
    t0 = time.monotonic()
    def _loraify(path, weight):
        nonlocal rngs, counter
        keystr = jtu.keystr(path, simple=True)
        if _is_match(keystr, weights_path):
            lora_key = jax.random.fold_in(rngs, counter)
            counter += 1
            lora_weight = LoraArray(
                weight,
                rank=rank,
                alpha=alpha,
                scale=scale,
                stop_gradient=stop_gradient,
                allow_materialise=allow_materialise,
                key=lora_key,
            )
            loraify_weight.append(keystr)
            return lora_weight
        else:
            return weight

    weights = jtu.tree_map_with_path(_loraify, model.weights)
    diff = time.monotonic() - t0
    print("Model weights converted to LoRA:", *loraify_weight)
    print(f"loraify takes {diff}s")
    return replace(
        model,
        weights=weights,
        is_lora=True,
        # Don't wrap the whole forward pass in `quaxify`, as some implementations
        # (e.g. shard_map-based attention) are not compatible with Quax's custom trace.
        # Instead, quaxify only the specific ops that touch `LoraArray` weights.
        forward=model.forward,
    )


@quax.quaxify
def _lora_array_matmul_impl(
    w,
    a,
    b,
    rhs,
    lhs_batch,
    ndim,
    scaling,
    dimension_numbers,
    kwargs,
    *,
    out_sharding,
    out2_sharding,
):
    n_sharedbatch = len(lhs_batch)  # = len(rhs_batch)
    # All of the lora batch dimensions that aren't a dot_general batch dimension.
    n_lorabatch = ndim - n_sharedbatch - 2
    assert n_lorabatch >= 0
    out1 = lax.dot_general(
        w, rhs, dimension_numbers, out_sharding=out_sharding, **kwargs
    )
    # out1 has shape (*sharedbatch, *lorabatch, x, *otherbatch)
    # `kwargs` must not include `out_sharding` here; `out2` has a different shape
    # from the outer dot_general output.
    out2 = lax.dot_general(
        b, rhs, dimension_numbers, out_sharding=out2_sharding, **kwargs
    )
    # out2 has shape(*sharedbatch, *lorabatch, z, *otherbatch)
    lhs_contract2 = (w.ndim - 1,)
    rhs_contract2 = (n_sharedbatch + n_lorabatch,)
    rhs_batch2 = tuple(range(n_sharedbatch + n_lorabatch))
    lhs_batch2 = lhs_batch + tuple(i for i in rhs_batch2 if i not in lhs_batch)
    dimension_numbers2 = ((lhs_contract2, rhs_contract2), (lhs_batch2, rhs_batch2))
    out3 = lax.dot_general(
        a, out2, dimension_numbers2, out_sharding=out_sharding, **kwargs
    )
    # out3 has shape (*sharedbatch, *lorabatch, x, *otherbatch)
    return out1 + scaling * out3


@quax.register(lax.dot_general_p)
def _lora_array_matmul(
    lhs: LoraArray,
    rhs: ArrayLike | quax.ArrayValue,
    *,
    dimension_numbers,
    **kwargs,
) -> ArrayLike | quax.ArrayValue:
    rhs = cast("Array", rhs)  # quax(jax) is type broadened.
    ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers
    [ndim] = {lhs.a.ndim, lhs.b.ndim, lhs.w.ndim}
    # `dot_general` in this codebase frequently uses `out_sharding`. Be careful not to
    # forward it to intermediate dot_generals inside the LoRA decomposition, as their
    # output shapes (and sometimes axis orders) differ from the outer op.
    out_sharding = kwargs.get("out_sharding", None)
    inner_kwargs = dict(kwargs)
    inner_kwargs.pop("out_sharding", None)

    scaling = jnp.asarray(lhs.alpha / lhs.a.shape[-1], dtype=lhs.w.dtype)
    # `out2` replaces the lhs uncontracted dimension with `rank`; shard it with
    # replication by default.
    if out_sharding is None:
        out2_sharding = None
    elif isinstance(out_sharding, jax.sharding.NamedSharding):
        spec = tuple(out_sharding.spec)
        spec = spec[: ndim - 2] + (None,) + spec[ndim - 1 :]
        out2_sharding = jax.sharding.NamedSharding(
            out_sharding.mesh, jax.sharding.PartitionSpec(*spec)
        )
    elif isinstance(out_sharding, jax.sharding.PartitionSpec):
        spec = tuple(out_sharding)
        spec = spec[: ndim - 2] + (None,) + spec[ndim - 1 :]
        out2_sharding = jax.sharding.PartitionSpec(*spec)
    else:
        out2_sharding = None

    if lhs_contract == (ndim - 1,) and (ndim - 2 not in lhs_batch):
        out = _lora_array_matmul_impl(
            lhs.w,
            lhs.a,
            lhs.b,
            rhs,
            lhs_batch,
            ndim,
            scaling,
            dimension_numbers,
            inner_kwargs,
            out_sharding=out_sharding,
            out2_sharding=out2_sharding,
        )
    elif lhs_contract == (ndim - 2,) and (ndim - 1 not in lhs_batch):
        T = lambda x: jnp.swapaxes(x, -1, -2)
        lhs_contract = (ndim - 1,)
        dimension_numbers = ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))
        out = _lora_array_matmul_impl(
            T(lhs.w),
            T(lhs.b),
            T(lhs.a),
            rhs,
            lhs_batch,
            ndim,
            scaling,
            dimension_numbers,
            inner_kwargs,
            out_sharding=out_sharding,
            out2_sharding=out2_sharding,
        )
    else:
        return quax.quaxify(lax.dot_general)(
            lhs.materialise(),
            rhs,
            dimension_numbers,
            out_sharding=out_sharding,
            **inner_kwargs,
        )

    return out


@quax.register(lax.dot_general_p)
def _(
    lhs: ArrayLike | quax.ArrayValue,
    rhs: LoraArray,
    *,
    dimension_numbers,
    **kwargs,
) -> ArrayLike | quax.ArrayValue:
    ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers
    dimension_numbers_flipped = ((rhs_contract, lhs_contract), (rhs_batch, lhs_batch))

    out_sharding = kwargs.get("out_sharding", None)
    inner_kwargs = dict(kwargs)
    inner_kwargs.pop("out_sharding", None)

    if out_sharding is None:
        out_sharding_flipped = None
    elif isinstance(out_sharding, jax.sharding.NamedSharding):
        spec = tuple(out_sharding.spec)
        n_sharedbatch = len(lhs_batch)
        n_rhs_uncontracted = rhs.aval().ndim - len(rhs_contract) - len(rhs_batch)
        n_lhs_uncontracted = len(spec) - n_sharedbatch - n_rhs_uncontracted
        spec = (
            spec[:n_sharedbatch]
            + spec[n_sharedbatch + n_lhs_uncontracted :]
            + spec[n_sharedbatch : n_sharedbatch + n_lhs_uncontracted]
        )
        out_sharding_flipped = jax.sharding.NamedSharding(
            out_sharding.mesh, jax.sharding.PartitionSpec(*spec)
        )
    elif isinstance(out_sharding, jax.sharding.PartitionSpec):
        spec = tuple(out_sharding)
        n_sharedbatch = len(lhs_batch)
        n_rhs_uncontracted = rhs.aval().ndim - len(rhs_contract) - len(rhs_batch)
        n_lhs_uncontracted = len(spec) - n_sharedbatch - n_rhs_uncontracted
        spec = (
            spec[:n_sharedbatch]
            + spec[n_sharedbatch + n_lhs_uncontracted :]
            + spec[n_sharedbatch : n_sharedbatch + n_lhs_uncontracted]
        )
        out_sharding_flipped = jax.sharding.PartitionSpec(*spec)
    else:
        out_sharding_flipped = None

    out = _lora_array_matmul(
        rhs,
        lhs,
        dimension_numbers=dimension_numbers_flipped,
        out_sharding=out_sharding_flipped,
        **inner_kwargs,
    )
    # out has shape (*sharedbatch, *rhs_uncontracted, *lhs_uncontracted)
    n_sharedbatch = len(lhs_batch)
    n_rhs_uncontracted = rhs.aval().ndim - len(rhs_contract) - len(rhs_batch)
    src = tuple(range(n_sharedbatch, n_sharedbatch + n_rhs_uncontracted))
    dest = tuple(range(-n_rhs_uncontracted, 0))
    out = quax.quaxify(jnp.moveaxis)(out, src, dest)  # pyright: ignore[reportArgumentType,reportAssignmentType]
    if out_sharding is not None:
        out = jax.sharding.reshard(out, out_sharding)
    return out
