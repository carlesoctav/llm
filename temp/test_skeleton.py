import pytest
from datasets import Dataset, IterableDataset
from jaxformers.data.huggingface_dataset import (
    HuggingFaceSourceIterDataset,
    HuggingFaceSourceMapDataset,
)
def test_iter_state_roundtrip():
    ds = IterableDataset.from_generator(lambda: ({"x": i} for i in range(10)))
    wrapped = HuggingFaceSourceIterDataset(ds)
    it = iter(wrapped)
    assert next(it)["x"] == 0
    assert next(it)["x"] == 1
    assert next(it)["x"] == 2
    state = it.get_state()
    assert next(it)["x"] == 3
    it.set_state(state)
    assert next(it)["x"] == 3  # resumes
def test_map_slice_even_odd_branch():
    base = Dataset.from_dict({"x": list(range(10))})
    wrapped = HuggingFaceSourceMapDataset(base)
    evens = [row["x"] for row in wrapped[0:len(base):2]._source]
    odds  = [row["x"] for row in wrapped[1:len(base):2]._source]
    assert evens == [0,2,4,6,8]
    assert odds  == [1,3,5,7,9]
