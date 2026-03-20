from typing import Protocol, runtime_checkable

import grain
from grain import (
    transforms as grain_transforms,
    IterDataset
)


@runtime_checkable
class DatasetTransforms(Protocol):
    def __call__(self, dataset: grain.IterDataset) -> grain.IterDataset: ...


TransformFn = grain_transforms.Map | grain_transforms.RandomMap | DatasetTransforms

def transform_ds(ds, *transforms: TransformFn) -> IterDataset:
    for op in transforms:
        if isinstance(op, DatasetTransforms):
            ds = op(ds)
        elif isinstance(op, grain_transforms.RandomMap):
            ds = ds.random_map(op)
        elif isinstance(op, grain_transforms.Map):
            ds = ds.map(op)
        else:
            raise TypeError(f"Unsupported operation type: {type(op)}")
    return ds
