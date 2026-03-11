import argparse
import collections
import dataclasses
import enum
import functools
from pathlib import Path
from typing import Any, Callable, Sequence, TypeAlias

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

# from jaxformers.dispatch.einsum import einsum
# from equinox import tree_pprint

einsum = jnp.einsum
try:
    from jaxformers.ops.cross_entropy.api import cross_entropy_loss
except ImportError:
    cross_entropy_loss = None


@dataclasses.dataclass(frozen=True)
class ReproConfig:
    num_layers: int = 26
    hidden_size: int = 1152
    intermediate_size: int = 6912
    vocab_size: int = 262144
    loss_impl: str = "base"
    optimizer: str = "sgd"
    use_lora: bool = False
    lora_rank: int = 16
    microbatch_size: int = 8
    accum_steps: int = 4
    seq_len: int = 2048
    learning_rate: float = 1e-4
    mesh_devices: int = 4
    random_mask: float | None = None
    random_mask_seed: int = 0
    dump_hlo: bool = True

    @property
    def logical_batch_size(self) -> int:
        return self.microbatch_size * self.accum_steps


def infer_shardings(tree):
    def leaf_sharding(leaf):
        if isinstance(leaf, jax.Array):
            return leaf.sharding
        return None

    return jtu.tree_map(leaf_sharding, tree)


def partition(pytree, filter=None, is_leaf=None):
    if filter is None:
        return pytree, jtu.tree_map(lambda x: None, pytree)

    left = jtu.tree_map(lambda m, v: v if m else None, filter, pytree, is_leaf=is_leaf)
    right = jtu.tree_map(
        lambda m, v: v if not m else None, filter, pytree, is_leaf=is_leaf
    )
    return left, right


def combine(left, right, is_leaf=None):
    def _combine(*args):
        for arg in args:
            if arg is not None:
                return arg

    is_none = lambda x: x is None
    _is_leaf = is_none if is_leaf is None else lambda x: is_none(x) or is_leaf(x)
    return jtu.tree_map(_combine, left, right, is_leaf=_is_leaf)


def apply_updates(weights, updates):
    def _f(w, u):
        if u is None:
            return w
        return w + u

    is_none = lambda x: x is None
    return jtu.tree_map(_f, weights, updates, is_leaf=is_none)


# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

AccumulatorTree: TypeAlias = Any
Function: TypeAlias = Callable[..., Any]
PyTreeFn: TypeAlias = Callable[[Any], Any]
UpdateFn = Callable[[Any, Any, int], Any]
IndividualOutputs = collections.namedtuple("Aux", ["values", "metrics", "aux"])


@dataclasses.dataclass(frozen=True)
class Accumulator:
    init: PyTreeFn
    update: UpdateFn
    finalize: PyTreeFn
    aggregate: PyTreeFn


class AccumulationType(enum.Enum):
    MEAN = enum.auto()
    SUM = enum.auto()
    RUNNING_MEAN = enum.auto()
    CONCAT = enum.auto()


def _with_floating_check(fn: Function) -> Function:
    def wrapper(*args, **kwargs):
        dtypes, _ = jtu.tree_flatten(jtu.tree_map(lambda x: x.dtype, (args, kwargs)))
        if not all(jnp.issubdtype(dtype, jnp.floating) for dtype in dtypes):
            raise ValueError(
                "MEAN and RUNNING_MEAN Accumulators require floating-point values."
            )
        return fn(*args, **kwargs)

    return wrapper


def _identity(value: Any) -> Any:
    return value


def _lift(accumulator: Accumulator) -> Accumulator:
    return Accumulator(
        lambda value: jtu.tree_map(accumulator.init, value),
        lambda carry, value, i: jtu.tree_map(
            lambda c, v: accumulator.update(c, v, i), carry, value
        ),
        lambda carry: jtu.tree_map(accumulator.finalize, carry),
        lambda values: jtu.tree_map(accumulator.aggregate, values),
    )


def _compose(accumulators: AccumulatorTree) -> Accumulator:
    def init(values):
        return jtu.tree_map(
            lambda acc, val: acc.init(val),
            accumulators,
            values,
        )

    def update(carry, value, index):
        return jtu.tree_map(
            lambda acc, car, val: acc.update(car, val, index),
            accumulators,
            carry,
            value,
        )

    def finalize(carry):
        return jtu.tree_map(
            lambda acc, car: acc.finalize(car),
            accumulators,
            carry,
        )

    def aggregate(values):
        return jtu.tree_map(
            lambda acc, val: acc.aggregate(val),
            accumulators,
            values,
        )

    return Accumulator(init, update, finalize, aggregate)


def _sum() -> Accumulator:
    return _lift(
        Accumulator(
            init=_identity,
            update=lambda carry, value, _: carry + value,
            finalize=_identity,
            aggregate=functools.partial(jnp.sum, axis=0),
        )
    )


def _mean(num_microbatches: int) -> Accumulator:
    if num_microbatches <= 0:
        raise ValueError(f"{num_microbatches=} must be positive.")
    return _lift(
        Accumulator(
            init=_with_floating_check(_identity),
            update=lambda carry, value, _: carry + value,
            finalize=lambda carry: carry / num_microbatches,
            aggregate=functools.partial(jnp.mean, axis=0),
        )
    )


def _running_mean() -> Accumulator:
    def update(carry, value, index):
        p = index / (index + 1)
        return carry * p + value * (1 - p)

    return _lift(
        Accumulator(
            init=_with_floating_check(_identity),
            update=update,
            finalize=_identity,
            aggregate=functools.partial(jnp.mean, axis=0),
        )
    )


def _concat(num_microbatches: int) -> Accumulator:
    if num_microbatches <= 0:
        raise ValueError(f"{num_microbatches=} must be positive.")

    def init(value):
        shape = (num_microbatches,) + value.shape
        zeros = jnp.broadcast_to(jnp.zeros_like(value), shape)
        return zeros.at[0].set(value)

    def update(carry, value, index):
        return carry.at[index].set(value)

    def finalize(carry):
        return carry.reshape(-1, *carry.shape[2:], order="F")

    return _lift(Accumulator(init, update, finalize, _identity))


def _canonicalize(
    tree: Accumulator | AccumulationType | AccumulatorTree,
    num_microbatches: int | None,
) -> Accumulator:
    def fun(acc):
        if isinstance(acc, Accumulator):
            return acc
        match acc:
            case AccumulationType.MEAN:
                return _mean(num_microbatches)
            case AccumulationType.SUM:
                return _sum()
            case AccumulationType.RUNNING_MEAN:
                return _running_mean()
            case AccumulationType.CONCAT:
                return _concat(num_microbatches)
        raise ValueError(f"Unknown accumulator: {acc}")

    return _compose(jtu.tree_map(fun, tree))


def _reshape_all_args(
    microbatch_size: int,
    argnums: Sequence[int],
    argnames: Sequence[str],
    in_axes: Sequence[int],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any], int]:
    new_args = list(args)
    new_kwargs = dict(kwargs)
    batch_args = [args[i] for i in argnums] + [kwargs[i] for i in argnames]

    batch_sizes = jtu.tree_flatten(
        jtu.tree_map(
            lambda ax, subtree: jtu.tree_map(lambda x: x.shape[ax], subtree),
            tuple(in_axes),
            tuple(batch_args),
        )
    )[0]

    if len(set(batch_sizes)) > 1:
        raise ValueError(
            f"Batch arguments must have equal-size batch axes, found {batch_sizes}."
        )

    batch_size = batch_sizes[0]
    if batch_size % microbatch_size != 0:
        raise ValueError(f"{batch_size=} must be divisible by {microbatch_size=}.")

    for i, ax in zip(argnums, in_axes):
        new_args[i] = reshape_batch_axis(args[i], microbatch_size, ax)

    for name, ax in zip(argnames, in_axes[len(argnums) :]):
        new_kwargs[name] = reshape_batch_axis(kwargs[name], microbatch_size, ax)

    return tuple(new_args), new_kwargs, batch_size


def _zeros_from_shape_struct(x):
    return jnp.zeros(x.shape, dtype=x.dtype)


def microbatch(
    fun: Function,
    argnums: int | Sequence[int],
    microbatch_size: int | None,
    accumulator: Accumulator | AccumulationType | AccumulatorTree = AccumulationType.SUM,
    *,
    argnames: str | Sequence[str] = (),
    in_axes: int | Sequence[int] = 0,
    num_real_microbatches: int | jax.Array | None = None,
) -> Function:
    if microbatch_size is None:
        return fun

    if isinstance(argnums, int):
        argnums = (argnums,)

    if isinstance(argnames, str):
        argnames = (argnames,)

    if isinstance(in_axes, int):
        in_axes = (in_axes,) * (len(argnums) + len(argnames))

    def microbatched_fun(*args, **kwargs):
        reshaped_args, reshaped_kwargs, batch_size = _reshape_all_args(
            microbatch_size, argnums, argnames, in_axes, args, kwargs
        )
        num_microbatches = batch_size // microbatch_size
        accumulator_ = _canonicalize(accumulator, num_microbatches)

        if accumulator != AccumulationType.SUM:
            raise NotImplementedError(
                "This local microbatch copy only supports AccumulationType.SUM."
            )

        def f(index):
            input_args = list(reshaped_args)
            input_kwargs = dict(reshaped_kwargs)
            for i, ax in zip(argnums, in_axes):
                input_args[i] = jtu.tree_map(
                    functools.partial(jnp.take, indices=index, axis=ax),
                    input_args[i],
                )
            for name, ax in zip(argnames, in_axes[len(argnums) :]):
                input_kwargs[name] = jtu.tree_map(
                    functools.partial(jnp.take, indices=index, axis=ax),
                    input_kwargs[name],
                )
            return fun(*input_args, **input_kwargs)

        def body_fun(index, carry):
            return accumulator_.update(carry, f(index), index)

        early_stop = num_real_microbatches is not None
        loop_bound = num_real_microbatches if early_stop else num_microbatches
        zero_answer = jtu.tree_map(_zeros_from_shape_struct, jax.eval_shape(f, 0))
        answer = jax.lax.fori_loop(0, loop_bound, body_fun, zero_answer)
        return accumulator_.finalize(answer)

    return microbatched_fun


def make_state_shardings(params, opt_state, mesh: Mesh):
    replicated = NamedSharding(mesh, P())
    param_shardings = infer_shardings(params)
    shape_to_sharding = {}
    for param_leaf, sharding_leaf in zip(
        jtu.tree_leaves(params), jtu.tree_leaves(param_shardings), strict=True
    ):
        if isinstance(param_leaf, jax.Array):
            shape_to_sharding.setdefault(tuple(param_leaf.shape), sharding_leaf)

    def opt_leaf_sharding(leaf):
        if not isinstance(leaf, jax.Array):
            return None
        return shape_to_sharding.get(tuple(leaf.shape), replicated)

    opt_state_shardings = jtu.tree_map(opt_leaf_sharding, opt_state)
    return (param_shardings, opt_state_shardings)


def print_compiled_memory_stats(compiled_stats):
    if compiled_stats is None:
        return None

    def bytes_to_gb(num_bytes):
        return num_bytes / (1024**3)

    output_gb = bytes_to_gb(compiled_stats.output_size_in_bytes)
    temp_gb = bytes_to_gb(compiled_stats.temp_size_in_bytes)
    argument_gb = bytes_to_gb(compiled_stats.argument_size_in_bytes)
    alias_gb = bytes_to_gb(compiled_stats.alias_size_in_bytes)
    host_temp_gb = bytes_to_gb(compiled_stats.host_temp_size_in_bytes)
    total_gb = output_gb + temp_gb + argument_gb - alias_gb

    print(
        f"Total memory size: {total_gb:.1f} GB, Output size: {output_gb:.1f} GB, Temp size: {temp_gb:.1f} GB, "
        f"Argument size: {argument_gb:.1f} GB, Host temp size: {host_temp_gb:.1f} GB."
    )

    return {
        "total_gb": round(total_gb, 1),
        "output_gb": round(output_gb, 1),
        "temp_gb": round(temp_gb, 1),
        "argument_gb": round(argument_gb, 1),
        "host_temp_gb": round(host_temp_gb, 1),
    }


def print_flops(compiled_stats):
    tflops = compiled_stats.get("flops") / 1e12
    print(f"estimated tflops per step: {tflops}")
    return {"tflops": tflops}


def validate_mode_config(mode: str, config: ReproConfig):
    if mode == "single":
        return
    if config.logical_batch_size % config.mesh_devices != 0:
        raise ValueError(
            "`logical_batch_size` must be divisible by `mesh_devices` for dp/fsdp."
        )
    if config.microbatch_size % config.mesh_devices != 0:
        raise ValueError(
            "`microbatch_size` must be divisible by `mesh_devices` for dp/fsdp."
        )


def select_devices(mode: str, mesh_devices: int):
    devices = jax.devices()
    if mode == "single":
        return devices[:1]
    if mesh_devices < 2:
        raise ValueError("`mesh_devices` must be >= 2 for dp/fsdp modes.")
    if len(devices) < mesh_devices:
        raise ValueError(
            f"Requested {mesh_devices} devices, but only found {len(devices)}."
        )
    return devices[:mesh_devices]


def make_mesh(mode: str, mesh_devices: int):
    devices = np.asarray(select_devices(mode, mesh_devices))
    return Mesh(devices, ("data",), axis_types=(AxisType.Explicit,))


def make_shardings(mode: str, mesh: Mesh):
    replicated = NamedSharding(mesh, P())
    batch_sharded = NamedSharding(mesh, P("data", None, None))
    fsdp_up = NamedSharding(mesh, P(None, "data"))
    fsdp_down = NamedSharding(mesh, P("data", None))
    fsdp_lm_head = NamedSharding(mesh, P(None, "data"))

    if mode == "single":
        return {
            "batch": replicated,
            "gate": replicated,
            "up": replicated,
            "down": replicated,
            "gate_lora_a": replicated,
            "gate_lora_b": replicated,
            "up_lora_a": replicated,
            "up_lora_b": replicated,
            "down_lora_a": replicated,
            "down_lora_b": replicated,
            "lm_head": replicated,
            "lm_head_lora_a": replicated,
            "lm_head_lora_b": replicated,
        }
    if mode == "dp":
        return {
            "batch": batch_sharded,
            "gate": replicated,
            "up": replicated,
            "down": replicated,
            "gate_lora_a": replicated,
            "gate_lora_b": replicated,
            "up_lora_a": replicated,
            "up_lora_b": replicated,
            "down_lora_a": replicated,
            "down_lora_b": replicated,
            "lm_head": replicated,
            "lm_head_lora_a": replicated,
            "lm_head_lora_b": replicated,
        }
    if mode == "fsdp":
        return {
            "batch": batch_sharded,
            "gate": fsdp_up,
            "up": fsdp_up,
            "down": fsdp_down,
            "gate_lora_a": fsdp_up,
            "gate_lora_b": replicated,
            "up_lora_a": fsdp_up,
            "up_lora_b": replicated,
            "down_lora_a": replicated,
            "down_lora_b": fsdp_down,
            "lm_head": fsdp_lm_head,
            "lm_head_lora_a": fsdp_lm_head,
            "lm_head_lora_b": replicated,
        }
    raise ValueError(f"Unknown mode: {mode}")


def make_linear_params(base, shardings, *, base_key: str, a_key: str, b_key: str, rank: int):
    if rank < 1:
        raise ValueError("`lora_rank` must be >= 1.")
    scale = 1e-2
    base = jax.device_put(base, shardings[base_key])
    lora_a = jax.device_put(
        jnp.zeros((rank, base.shape[1]), dtype=base.dtype) * scale,
        shardings[a_key],
    )
    lora_b = jax.device_put(
        jnp.zeros((base.shape[0], rank), dtype=base.dtype) * scale,
        shardings[b_key],
    )
    return {"base": base, "lora_a": lora_a, "lora_b": lora_b}


def make_params(config: ReproConfig, shardings):
    layers = []
    key = jax.random.PRNGKey(0)
    scale = 1e-2
    for _ in range(config.num_layers):
        key, gate_key, up_key, down_key = jax.random.split(key, 4)
        gate = jax.random.normal(
            gate_key,
            (config.intermediate_size, config.hidden_size),
            dtype=jnp.bfloat16,
        ) * scale
        up = jax.random.normal(
            up_key,
            (config.intermediate_size, config.hidden_size),
            dtype=jnp.bfloat16,
        ) * scale
        down = jax.random.normal(
            down_key,
            (config.hidden_size, config.intermediate_size),
            dtype=jnp.bfloat16,
        ) * scale
        if config.use_lora:
            layers.append(
                {
                    "gate": make_linear_params(
                        gate,
                        shardings,
                        base_key="gate",
                        a_key="gate_lora_a",
                        b_key="gate_lora_b",
                        rank=config.lora_rank,
                    ),
                    "up": make_linear_params(
                        up,
                        shardings,
                        base_key="up",
                        a_key="up_lora_a",
                        b_key="up_lora_b",
                        rank=config.lora_rank,
                    ),
                    "down": make_linear_params(
                        down,
                        shardings,
                        base_key="down",
                        a_key="down_lora_a",
                        b_key="down_lora_b",
                        rank=config.lora_rank,
                    ),
                }
            )
        else:
            layers.append(
                (
                    jax.device_put(gate, shardings["gate"]),
                    jax.device_put(up, shardings["up"]),
                    jax.device_put(down, shardings["down"]),
                )
            )
    key, lm_head_key = jax.random.split(key)
    lm_head = jax.random.normal(
        lm_head_key,
        (config.vocab_size, config.hidden_size),
        dtype=jnp.bfloat16,
    ) * scale
    if config.use_lora:
        lm_head = make_linear_params(
            lm_head,
            shardings,
            base_key="lm_head",
            a_key="lm_head_lora_a",
            b_key="lm_head_lora_b",
            rank=config.lora_rank,
        )
    else:
        lm_head = jax.device_put(lm_head, shardings["lm_head"])
    return (
        tuple(layers),
        lm_head,
    )


def make_train_mask(params, config: ReproConfig):
    if config.use_lora:
        def build_mask(node):
            if isinstance(node, dict):
                mask = {}
                for key, value in node.items():
                    if key == "base":
                        mask[key] = False
                    elif key in {"lora_a", "lora_b"}:
                        mask[key] = True
                    else:
                        mask[key] = build_mask(value)
                return mask
            if isinstance(node, tuple):
                return tuple(build_mask(value) for value in node)
            if isinstance(node, list):
                return [build_mask(value) for value in node]
            return False

        return build_mask(params)

    if config.random_mask is None:
        return None
    if not 0.0 < config.random_mask <= 1.0:
        raise ValueError("`random_mask` must be in the interval (0, 1].")

    leaves, treedef = jtu.tree_flatten(params)
    key = jax.random.PRNGKey(config.random_mask_seed)
    mask_leaves = []
    trainable_count = 0

    for leaf in leaves:
        key, subkey = jax.random.split(key)
        is_trainable = bool(jax.random.bernoulli(subkey, config.random_mask))
        mask_leaves.append(is_trainable)
        trainable_count += int(is_trainable)

    if trainable_count == 0:
        mask_leaves[-1] = True

    return jtu.tree_unflatten(treedef, mask_leaves)


def count_trainable(mask) -> tuple[int, int] | None:
    if mask is None:
        return None
    leaves = jtu.tree_leaves(mask)
    trainable = sum(bool(leaf) for leaf in leaves)
    return trainable, len(leaves)


def make_batch(batch_size: int, config: ReproConfig, batch_sharding):
    inputs = jax.random.normal(
        jax.random.PRNGKey(0),
        (batch_size, config.seq_len, config.hidden_size),
        dtype=jnp.bfloat16,
    )
    labels = jax.random.randint(
        jax.random.PRNGKey(1),
        (batch_size, config.seq_len),
        minval=0,
        maxval=config.vocab_size,
        dtype=jnp.int32,
    )
    labels_sharding = batch_sharding
    if isinstance(batch_sharding, NamedSharding) and len(batch_sharding.spec) != 0:
        labels_sharding = NamedSharding(batch_sharding.mesh, P(batch_sharding.spec[0], None))
    return {
        "inputs": jax.device_put(inputs, batch_sharding),
        "labels": jax.device_put(labels, labels_sharding),
    }


def mlp_forward(params, x):
    layers, _ = params
    h = x
    for layer in layers:
        out_sharding = jax.typeof(h).sharding
        if isinstance(layer, dict):
            gate = einsum("bth,fh->btf", h, layer["gate"]["base"], out_sharding=out_sharding)
            gate_lora = einsum(
                "bth,rh->btr",
                h,
                layer["gate"]["lora_a"],
                out_sharding=out_sharding,
            )
            gate = gate + einsum(
                "btr,fr->btf",
                gate_lora,
                layer["gate"]["lora_b"],
                out_sharding=out_sharding,
            )
            up = einsum("bth,fh->btf", h, layer["up"]["base"], out_sharding=out_sharding)
            up_lora = einsum(
                "bth,rh->btr",
                h,
                layer["up"]["lora_a"],
                out_sharding=out_sharding,
            )
            up = up + einsum(
                "btr,fr->btf",
                up_lora,
                layer["up"]["lora_b"],
                out_sharding=out_sharding,
            )
            down_base = layer["down"]["base"]
            down_a = layer["down"]["lora_a"]
            down_b = layer["down"]["lora_b"]
        else:
            gate_w, up_w, down_w = layer
            gate = einsum("bth,fh->btf", h, gate_w, out_sharding=out_sharding)
            up = einsum("bth,fh->btf", h, up_w, out_sharding=out_sharding)
            down_base = down_w
            down_a = None
            down_b = None
        hidden = jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
        out = einsum(
            "btf,hf->bth",
            hidden.astype(h.dtype),
            down_base,
            out_sharding=out_sharding,
        )
        if down_a is not None:
            down_lora = einsum(
                "btf,rf->btr",
                hidden.astype(h.dtype),
                down_a,
                out_sharding=out_sharding,
            )
            out = out + einsum(
                "btr,hr->bth",
                down_lora,
                down_b,
                out_sharding=out_sharding,
            )
        h = h + out
    return h


def loss_fn(train_weights, frozen_weights, batch, *, loss_impl: str = "base"):
    params = combine(train_weights, frozen_weights)
    hidden = mlp_forward(params, batch["inputs"])
    _, lm_head = params
    if isinstance(lm_head, dict):
        lm_head_base = lm_head["base"]
        lm_head_a = lm_head["lora_a"]
        lm_head_b = lm_head["lora_b"]
    else:
        lm_head_base = lm_head
        lm_head_a = None
        lm_head_b = None
    token_count = jnp.asarray(batch["labels"].size, dtype=jnp.int32)
    if loss_impl == "base":
        logits = einsum(
            "bth,vh->btv",
            hidden,
            lm_head_base,
            out_sharding=jax.typeof(hidden).sharding,
        )
        if lm_head_a is not None:
            logits_lora = einsum(
                "bth,rh->btr",
                hidden,
                lm_head_a,
                out_sharding=jax.typeof(hidden).sharding,
            )
            logits = logits + einsum(
                "btr,vr->btv",
                logits_lora,
                lm_head_b,
                out_sharding=jax.typeof(hidden).sharding,
            )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32),
            batch["labels"],
        )
        loss = jnp.sum(loss)
        return loss, {"loss": (loss, token_count), "token_count": token_count}
    if loss_impl == "reference":
        if cross_entropy_loss is None:
            raise ImportError(
                "`loss_impl=reference` requires `jaxformers.ops.cross_entropy.api`."
            )
        flat_hidden = hidden.reshape((-1, hidden.shape[-1]))
        flat_labels = batch["labels"].reshape(-1)
        flat_lm_head = lm_head_base
        if lm_head_a is not None:
            flat_lm_head = flat_lm_head + einsum(
                "vr,rh->vh",
                lm_head_b,
                lm_head_a,
                out_sharding=jax.typeof(lm_head_base).sharding,
            )
        loss = cross_entropy_loss(
            flat_hidden,
            flat_labels,
            flat_lm_head,
            reduction="sum",
            implementation="reference",
        )
        return loss, {"loss": (loss, token_count), "token_count": token_count}
    raise ValueError(f"Unsupported loss_impl: {loss_impl}")


def reshape_batch_axis(batch, microbatch_size: int, axis: int = 0):
    def reshape_leaf(x):
        new_shape = x.shape[:axis] + (-1, microbatch_size) + x.shape[axis + 1 :]
        if jax.__version__ < "0.7.0":
            return x.reshape(new_shape, order="F")

        sharding = jax.typeof(x).sharding
        if not sharding.mesh.are_all_axes_explicit:
            return x.reshape(new_shape, order="F")

        assert jax.__version__ >= "0.8.1", (
            "microbatching with explicit sharding requires jax version >= 0.8.1."
        )
        spec = sharding.spec
        if len(spec) < axis:
            new_spec = spec
        else:
            new_spec = jax.sharding.PartitionSpec(
                *spec[:axis], None, spec[axis], *spec[axis + 1 :]
            )
        out_sharding = jax.sharding.NamedSharding(sharding.mesh, new_spec)

        local_shape = sharding.shard_shape(x.shape)
        nshards = x.shape[axis] // local_shape[axis]
        if microbatch_size % nshards != 0:
            raise ValueError(f"{nshards=} must evenly divide {microbatch_size=}.")

        return x.reshape(new_shape, order="F", out_sharding=out_sharding)

    return jtu.tree_map(reshape_leaf, batch)


def make_optimizer(config: ReproConfig):
    if config.optimizer == "sgd":
        return optax.sgd(config.learning_rate)
    if config.optimizer == "adam":
        return optax.adam(config.learning_rate)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def index_microbatch(batch, index):
    return jtu.tree_map(
        lambda x: jnp.take(x, index, axis=0),
        batch,
    )

def make_optax_microbatch_train_step(config: ReproConfig, train_mask):
    tx = make_optimizer(config)
    loss_impl_fn = lambda train_weights, frozen_weights, batch: loss_fn(
        train_weights, frozen_weights, batch, loss_impl=config.loss_impl
    )

    def train_step(state, batch):
        params, opt_state = state
        train_weights, frozen_weights = partition(params, train_mask)
        grad_fn = optax.microbatch(
            jax.value_and_grad(loss_impl_fn, has_aux=True),
            argnums=2,
            microbatch_size=config.microbatch_size,
        )
        (_, aux), grads = grad_fn(train_weights, frozen_weights, batch)
        inv_token_count = (1 / aux["token_count"]).astype(jnp.bfloat16)
        grads = jtu.tree_map(lambda g: g * inv_token_count, grads)
        updates, opt_state = tx.update(grads, opt_state, train_weights)
        params = apply_updates(params, updates)
        return (params, opt_state), aux

    return train_step, tx


def make_new_microbatch_train_step(config: ReproConfig, train_mask):
    tx = make_optimizer(config)
    loss_impl_fn = lambda train_weights, frozen_weights, batch: loss_fn(
        train_weights, frozen_weights, batch, loss_impl=config.loss_impl
    )

    def train_step(state, batch):
        params, opt_state = state
        train_weights, frozen_weights = partition(params, train_mask)
        grad_fn = microbatch(
            jax.value_and_grad(loss_impl_fn, has_aux=True),
            argnums=2,
            microbatch_size=config.microbatch_size,
        )
        (_, aux), grads = grad_fn(train_weights, frozen_weights, batch)
        inv_token_count = (1 / aux["token_count"]).astype(jnp.bfloat16)
        grads = jtu.tree_map(lambda g: g * inv_token_count, grads)
        updates, opt_state = tx.update(grads, opt_state, train_weights)
        params = apply_updates(params, updates)
        return (params, opt_state), aux

    return train_step, tx


def make_scan_microbatch_train_step(config: ReproConfig, train_mask):
    tx = make_optimizer(config)
    loss_impl_fn = lambda train_weights, frozen_weights, batch: loss_fn(
        train_weights, frozen_weights, batch, loss_impl=config.loss_impl
    )
    grad_fn = jax.value_and_grad(loss_impl_fn, has_aux=True)

    def train_step(state, batch):
        params, opt_state = state
        train_weights, frozen_weights = partition(params, train_mask)
        scanned_batch = reshape_batch_axis(batch, config.microbatch_size)
        zero_loss = jnp.zeros([], dtype=jnp.float32)
        zero_token_count = jnp.zeros([], dtype=jnp.int32)
        zero_grad = jtu.tree_map(jnp.zeros_like, train_weights)

        def body_fn(carry, minibatch):
            loss_sum, token_count_sum, grad_sum = carry
            (minibatch_loss, minibatch_aux), minibatch_grad = grad_fn(
                train_weights,
                frozen_weights,
                minibatch,
            )
            loss_sum = loss_sum + minibatch_loss
            token_count_sum = token_count_sum + minibatch_aux["token_count"]
            grad_sum = jtu.tree_map(
                lambda acc, grad: acc + grad.astype(acc.dtype),
                grad_sum,
                minibatch_grad,
            )
            return (loss_sum, token_count_sum, grad_sum), None

        (loss, token_count, grads), _ = jax.lax.scan(
            body_fn,
            init=(zero_loss, zero_token_count, zero_grad),
            xs=scanned_batch,
        )
        aux = {"loss": (loss, token_count), "token_count": token_count}
        inv_token_count = (1 / token_count).astype(jnp.bfloat16)
        grads = jtu.tree_map(lambda g: g * inv_token_count, grads)
        updates, opt_state = tx.update(grads, opt_state, train_weights)
        params = apply_updates(params, updates)
        return (params, opt_state), aux

    return train_step, tx


def make_init_outside_microbatch_train_step(config: ReproConfig, train_mask):
    tx = make_optimizer(config)
    loss_impl_fn = lambda train_weights, frozen_weights, batch: loss_fn(
        train_weights, frozen_weights, batch, loss_impl=config.loss_impl
    )
    grad_fn = jax.value_and_grad(loss_impl_fn, has_aux=True)

    def train_step(state, batch):
        params, opt_state = state
        train_weights, frozen_weights = partition(params, train_mask)
        scanned_batch = reshape_batch_axis(batch, config.microbatch_size)
        first_batch = index_microbatch(scanned_batch, 0)
        (first_loss, first_aux), first_grads = grad_fn(
            train_weights,
            frozen_weights,
            first_batch,
        )

        def body_fn(index, carry):
            loss_sum, token_count_sum, grad_sum = carry
            minibatch = index_microbatch(scanned_batch, index)
            (minibatch_loss, minibatch_aux), minibatch_grad = grad_fn(
                train_weights,
                frozen_weights,
                minibatch,
            )
            loss_sum = loss_sum + minibatch_loss
            token_count_sum = token_count_sum + minibatch_aux["token_count"]
            grad_sum = jtu.tree_map(
                lambda acc, grad: acc + grad.astype(acc.dtype),
                grad_sum,
                minibatch_grad,
            )
            return loss_sum, token_count_sum, grad_sum

        loss, token_count, grads = jax.lax.fori_loop(
            1,
            config.accum_steps,
            body_fn,
            (
                first_loss.astype(jnp.float32),
                first_aux["token_count"],
                first_grads,
            ),
        )
        aux = {"loss": (loss, token_count), "token_count": token_count}
        inv_token_count = (1 / token_count).astype(jnp.bfloat16)
        grads = jtu.tree_map(lambda g: g * inv_token_count, grads)
        updates, opt_state = tx.update(grads, opt_state, train_weights)
        params = apply_updates(params, updates)
        return (params, opt_state), aux

    return train_step, tx


def make_multistep_train_step(config: ReproConfig, train_mask):
    tx = optax.MultiSteps(
        make_optimizer(config),
        every_k_schedule=config.accum_steps,
        use_grad_mean=True,
    )
    loss_impl_fn = lambda train_weights, frozen_weights, batch: loss_fn(
        train_weights, frozen_weights, batch, loss_impl=config.loss_impl
    )

    def train_step(state, batch):
        params, opt_state = state
        train_weights, frozen_weights = partition(params, train_mask)
        (_, aux), grads = jax.value_and_grad(loss_impl_fn, has_aux=True)(
            train_weights,
            frozen_weights,
            batch,
        )
        updates, opt_state = tx.update(grads, opt_state, train_weights)
        params = apply_updates(params, updates)
        return (params, opt_state), aux

    return train_step, tx


def compile_and_report(label, train_step, state, batch, dump_path: Path | None):
    print(f"{label}:")
    step_jit = jax.jit(
        train_step,
        donate_argnums=(0,),
    )
    lower = step_jit.lower(state, batch)
    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(lower.as_text())
    compiled = lower.compile()
    print_compiled_memory_stats(compiled.memory_analysis())
    print_flops(compiled.cost_analysis())
    state, metrics = compiled(state, batch)
    metrics = jax.tree.map(lambda x: x.block_until_ready(), metrics)
    return state, float(metrics["loss"][0] / metrics["loss"][1])


def run_new_microbatch(mode: str, config: ReproConfig):
    validate_mode_config(mode, config)
    mesh = make_mesh(mode, config.mesh_devices)
    shardings = make_shardings(mode, mesh)
    with mesh:
        params = make_params(config, shardings)
        train_mask = make_train_mask(params, config)
        train_weights, _ = partition(params, train_mask)
        # tree_pprint(train_weights)
        train_step, tx = make_new_microbatch_train_step(config, train_mask)
        opt_state = tx.init(train_weights)
        state = (params, opt_state)
        # state = jax.device_put((params, opt_state), make_state_shardings(params, opt_state, mesh))
        batch = make_batch(config.logical_batch_size, config, shardings["batch"])
        counts = count_trainable(train_mask)
        if counts is not None:
            print(f"{mode}/new_microbatch trainable_leaves={counts[0]}/{counts[1]}")
        hlo_path = (
            Path("issues/microbatch/hlo") / f"{mode}_microbatch.stablehlo"
            if config.dump_hlo
            else None
        )
        _, loss = compile_and_report(
            f"{mode}/new_microbatch",
            train_step,
            state,
            batch,
            hlo_path,
        )
        print(f"{mode}/microbatch loss={loss:.6f}")

def run_optax_microbatch(mode: str, config: ReproConfig):
    validate_mode_config(mode, config)
    mesh = make_mesh(mode, config.mesh_devices)
    shardings = make_shardings(mode, mesh)
    with mesh:
        params = make_params(config, shardings)
        train_mask = make_train_mask(params, config)
        train_weights, _ = partition(params, train_mask)
        # tree_pprint(train_weights)
        train_step, tx = make_optax_microbatch_train_step(config, train_mask)
        opt_state = tx.init(train_weights)
        state = (params, opt_state)
        # state = jax.device_put((params, opt_state), make_state_shardings(params, opt_state, mesh))
        batch = make_batch(config.logical_batch_size, config, shardings["batch"])
        counts = count_trainable(train_mask)
        if counts is not None:
            print(f"{mode}/microbatch trainable_leaves={counts[0]}/{counts[1]}")
        hlo_path = (
            Path("issues/microbatch/hlo") / f"{mode}_microbatch.stablehlo"
            if config.dump_hlo
            else None
        )
        _, loss = compile_and_report(
            f"{mode}/optax_microbatch",
            train_step,
            state,
            batch,
            hlo_path,
        )
        print(f"{mode}/optax_microbatch loss={loss:.6f}")


def run_scan_microbatch(mode: str, config: ReproConfig):
    validate_mode_config(mode, config)
    mesh = make_mesh(mode, config.mesh_devices)
    shardings = make_shardings(mode, mesh)
    with mesh:
        params = make_params(config, shardings)
        train_mask = make_train_mask(params, config)
        train_weights, _ = partition(params, train_mask)
        train_step, tx = make_scan_microbatch_train_step(config, train_mask)
        opt_state = tx.init(train_weights)
        state = (params, opt_state)
        batch = make_batch(config.logical_batch_size, config, shardings["batch"])
        counts = count_trainable(train_mask)
        if counts is not None:
            print(f"{mode}/scan_microbatch trainable_leaves={counts[0]}/{counts[1]}")
        hlo_path = (
            Path("issues/microbatch/hlo") / f"{mode}_scan_microbatch.stablehlo"
            if config.dump_hlo
            else None
        )
        _, loss = compile_and_report(
            f"{mode}/scan_microbatch",
            train_step,
            state,
            batch,
            hlo_path,
        )
        print(f"{mode}/scan_microbatch loss={loss:.6f}")


def run_init_outside_microbatch(mode: str, config: ReproConfig):
    validate_mode_config(mode, config)
    mesh = make_mesh(mode, config.mesh_devices)
    shardings = make_shardings(mode, mesh)
    with mesh:
        params = make_params(config, shardings)
        train_mask = make_train_mask(params, config)
        train_weights, _ = partition(params, train_mask)
        train_step, tx = make_init_outside_microbatch_train_step(config, train_mask)
        opt_state = tx.init(train_weights)
        state = (params, opt_state)
        batch = make_batch(config.logical_batch_size, config, shardings["batch"])
        counts = count_trainable(train_mask)
        if counts is not None:
            print(f"{mode}/init_outside_microbatch trainable_leaves={counts[0]}/{counts[1]}")
        hlo_path = (
            Path("issues/microbatch/hlo")
            / f"{mode}_init_outside_microbatch.stablehlo"
            if config.dump_hlo
            else None
        )
        _, loss = compile_and_report(
            f"{mode}/init_outside_microbatch",
            train_step,
            state,
            batch,
            hlo_path,
        )
        print(f"{mode}/init_outside_microbatch loss={loss:.6f}")


def run_multistep(mode: str, config: ReproConfig):
    validate_mode_config(mode, config)
    mesh = make_mesh(mode, config.mesh_devices)
    shardings = make_shardings(mode, mesh)
    with mesh:
        params = make_params(config, shardings)
        train_mask = make_train_mask(params, config)
        train_weights, _ = partition(params, train_mask)
        train_step, tx = make_multistep_train_step(config, train_mask)
        opt_state = tx.init(train_weights)
        state = jax.device_put((params, opt_state), make_state_shardings(params, opt_state, mesh))
        batch = make_batch(config.microbatch_size, config, shardings["batch"])
        counts = count_trainable(train_mask)
        if counts is not None:
            print(f"{mode}/multistep trainable_leaves={counts[0]}/{counts[1]}")
        hlo_path = (
            Path("issues/microbatch/hlo") / f"{mode}_multistep.stablehlo"
            if config.dump_hlo
            else None
        )
        _, loss = compile_and_report(
            f"{mode}/multistep",
            train_step,
            state,
            batch,
            hlo_path,
        )
        print(f"{mode}/multistep loss={loss:.6f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["single", "dp", "fsdp"],
        choices=["single", "dp", "fsdp"],
    )
    parser.add_argument("--num-layers", type=int, default=26)
    parser.add_argument("--hidden-size", type=int, default=1152)
    parser.add_argument("--intermediate-size", type=int, default=6912)
    parser.add_argument("--vocab-size", type=int, default=262144)
    parser.add_argument("--loss-impl", choices=["base", "reference"], default="base")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--mesh-devices", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--random-mask", type=float, default=None)
    parser.add_argument("--random-mask-seed", type=int, default=0)
    parser.add_argument("--no-dump-hlo", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ReproConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        vocab_size=args.vocab_size,
        loss_impl=args.loss_impl,
        optimizer=args.optimizer,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        seq_len=args.seq_len,
        microbatch_size=args.microbatch_size,
        accum_steps=args.accum_steps,
        learning_rate=args.learning_rate,
        mesh_devices=args.mesh_devices,
        random_mask=args.random_mask,
        random_mask_seed=args.random_mask_seed,
        dump_hlo=not args.no_dump_hlo,
    )

    print(
        "config:",
        {
            "num_layers": config.num_layers,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size,
            "loss_impl": config.loss_impl,
            "optimizer": config.optimizer,
            "use_lora": config.use_lora,
            "lora_rank": config.lora_rank,
            "seq_len": config.seq_len,
            "microbatch_size": config.microbatch_size,
            "accum_steps": config.accum_steps,
            "logical_batch_size": config.logical_batch_size,
            "mesh_devices": config.mesh_devices,
            "random_mask": config.random_mask,
            "random_mask_seed": config.random_mask_seed,
            "available_devices": len(jax.devices()),
        },
    )

    for mode in args.modes:
        print(f"\n=== mode={mode} ===")
        run_new_microbatch(mode, config)
        run_optax_microbatch(mode, config)
        run_multistep(mode, config)
        # run_scan_microbatch(mode, config)
        # run_init_outside_microbatch(mode, config)


if __name__ == "__main__":
    main()
