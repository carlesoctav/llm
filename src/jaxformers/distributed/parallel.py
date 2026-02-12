from dataclasses import dataclass
from typing import TypedDict

BATCH = ("dp_replicate", "dp_shard")
FSDP = ("dp_shard", "cp")
MODEL = ("tp",)
SEQ = ("tp", "cp")
CONTEXT = ("cp",)
AxisName = str | tuple[str, ...] | None


@dataclass
class ParallelDims(TypedDict):
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int

@dataclass
class SparseParallelDims(TypedDict):
    dp_replicate: int
    dp_shard: int
    cp: int
    ep: int
    etp: int

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
    rules: dict[str, AxisName],
    parallel_dims: ParallelDims,
    sequence_parallelism: bool = True,
):
    if not sequence_parallelism:
        rules["context"] = drop_axis(rules.get("context"), "tp")
        rules["sequence"] = drop_axis(rules.get("sequence"), "tp")

    if parallel_dims["cp"] == 1:
        rules["context"] = drop_axis(rules.get("context"), "cp")

    return rules


def check_mesh_axis_for_inference(parallel_dims: ParallelDims):
    # if parallel_dims["dp_shard"] > 1:
    #     raise ValueError("not recommended to shard across dp_shard for inference")
    if parallel_dims["cp"] > 1:
        raise ValueError("context parallelism not supported for inference")
