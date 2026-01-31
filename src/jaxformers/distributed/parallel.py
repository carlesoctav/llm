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

@dataclass
class SparseParallelDims(TypedDict):
    dp_replicate: int
    dp_shard: int
    cp: int
    ep: int
    etp: int

    ep: int
    etp: int

def change_sharding_rule_parallel_dims(rules: dict[str, AxisName], parallel_dims: ParallelDims):
    if parallel_dims["cp"] == 1:
        rules["context"] = None
        # rules["sequence"] = ("tp",)
        # rules["fsdp"] = ("dp_shard", )
    return rules
