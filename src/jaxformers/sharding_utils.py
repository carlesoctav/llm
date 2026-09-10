from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypeAlias, TypedDict

import jax
from jax.sharding import AxisType, PartitionSpec as P


BATCH = ("dp_replicate", "dp_shard")
FSDP = ("dp_shard", "cp")
MODEL = ("tp",)
SEQ = ("tp", "cp")
CONTEXT = ("cp",)
AxisName = str | tuple[str, ...] | None
LogicalRules: TypeAlias = dict[str, AxisName]

DEFAULT_LOGICAL_AXIS_RULES: LogicalRules = {
    "none": None,
    "batch": BATCH,
    "fsdp": FSDP,
    "model": MODEL,
    "sequence": SEQ,
    "context": CONTEXT,
}
_CURRENT_LOGICAL_AXIS_RULES = ContextVar("logical_axis_rules", default={})


class ParallelDims(TypedDict):
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int


DEFAULT_PARALLEL_DIMS = {"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1}


class SparseParallelDims(TypedDict):
    dp_replicate: int
    dp_shard: int
    cp: int
    ep: int
    etp: int


def drop_axis(mesh_axes: AxisName, axis_name: str) -> AxisName:
    if mesh_axes is None:
        return None
    if isinstance(mesh_axes, str):
        return None if mesh_axes == axis_name else mesh_axes

    filtered_axes = tuple(axis for axis in mesh_axes if axis != axis_name)
    return filtered_axes if filtered_axes else None


def mutate_sharding_rule_parallel_dims(
    sharding_rules: LogicalRules,
    parallel_dims: ParallelDims,
    sequence_parallelism: bool = True,
):
    rules = dict(sharding_rules)
    if not sequence_parallelism:
        rules["sequence"] = drop_axis(rules["sequence"], "tp")

    # if parallel_dims["cp"] == 1:
    #     rules["context"] = drop_axis(rules["context"], "cp")
    #     rules["sequence"] = drop_axis(rules["sequence"], "cp")

    # # Avoid explicit singleton mesh axes in PartitionSpecs. These can trigger sharding
    # # mismatches in transposes/VJPs (e.g. reductions) on newer JAX versions.
    # for axis_name, axis_size in parallel_dims.items():
    #     if axis_size != 1:
    #         continue
    #     for rule_key, rule_val in list(rules.items()):
    #         rules[rule_key] = drop_axis(rule_val, axis_name)

    return rules


def make_logical_axis_rules(
    parallel_dims: ParallelDims,
    *,
    sequence_parallelism: bool = True,
    **kwargs,
) -> LogicalRules:
    return mutate_sharding_rule_parallel_dims(
        DEFAULT_LOGICAL_AXIS_RULES,
        parallel_dims,
        sequence_parallelism=sequence_parallelism,
    )


def make_mesh(
    parallel_dims: ParallelDims,
    devices: list | None = None,
    *,
    multihost: bool = False,
    **kwargs,
):
    if multihost:
        jax.distributed.initialize()

    axis_shapes = tuple(parallel_dims.values())
    axis_names = tuple(parallel_dims.keys())
    axis_types = tuple(AxisType.Explicit for _ in axis_names)
    return jax.make_mesh(
        axis_shapes,
        axis_names,
        axis_types=axis_types,
        devices=devices,
    )


@contextmanager
def with_logical_axis(rules: LogicalRules):
    token = _CURRENT_LOGICAL_AXIS_RULES.set(rules)
    try:
        yield
    finally:
        _CURRENT_LOGICAL_AXIS_RULES.reset(token)


def get_logical_axis_rules() -> LogicalRules:
    return _CURRENT_LOGICAL_AXIS_RULES.get()


def remove_size_one_mesh_axis(spec, mesh):
    if spec is None:
        return None

    new_spec = []
    for s in spec:
        if s is None or s == P.UNCONSTRAINED:
            new_spec.append(s)
        elif isinstance(s, tuple):
            new_spec.append(tuple(i for i in s if mesh.shape.get(i, 1) != 1))
        else:
            new_spec.append(None if mesh.shape.get(s, 1) == 1 else s)

    return P(*new_spec, unreduced=spec.unreduced, reduced=spec.reduced)


def from_logical_rules(
    sharding: tuple[str | None, ...],
    sharding_rules: LogicalRules | None = None,
) -> P:
    if get_logical_axis_rules() or sharding_rules:
        context_rules = get_logical_axis_rules()
        rules = sharding_rules or context_rules
        spec = P(
            *tuple(rules[str(s)] if (s and str(s) in rules) else s for s in sharding)
        )
    else:
        spec = P(*sharding)
    try:
        mesh = jax.sharding.get_abstract_mesh()
    except Exception:
        return spec
    return remove_size_one_mesh_axis(spec, mesh)


def check_mesh_axis_for_inference(parallel_dims: ParallelDims):
    # if parallel_dims["dp_shard"] > 1:
    #     raise ValueError("not recommended to shard across dp_shard for inference")
    if parallel_dims["cp"] > 1:
        raise ValueError("context parallelism not supported for inference")
