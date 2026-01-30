from dataclasses import dataclass
from typing import TypedDict

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
