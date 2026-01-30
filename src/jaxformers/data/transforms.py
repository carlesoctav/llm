from typing import Protocol, runtime_checkable
import grain


@runtime_checkable
class DatasetTransforms(Protocol):
    def __call__(self, dataset: grain.IterDataset) -> grain.IterDataset: ...
