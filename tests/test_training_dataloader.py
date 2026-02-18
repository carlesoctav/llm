import dataclasses as dc

import jax
import numpy as np
from datasets import Dataset, IterableDataset
from grain import transforms as grain_transforms
from jax.sharding import Mesh, PartitionSpec

from jaxformers.data.training import make_dataloader


@dc.dataclass
class SimpleTokenize(grain_transforms.Map):
    column: str = "text"
    max_length: int = 4
    pad_id: int = 0

    def map(self, features: dict) -> dict:
        text = features[self.column]
        ids = np.full((self.max_length,), self.pad_id, dtype=np.int32)
        encoded = [ord(c) for c in text][: self.max_length]
        if encoded:
            ids[: len(encoded)] = np.asarray(encoded, dtype=np.int32)

        out = {
            "id": np.int32(features["id"]),
            "input_ids": ids,
        }
        if "source" in features:
            out["source"] = np.int32(features["source"])
        return out


def text_iterable_dataset(
    num_examples: int,
    *,
    source: int | None = None,
    id_offset: int = 0,
    num_shards: int = 1,
) -> IterableDataset:
    ids = list(range(id_offset, id_offset + num_examples))
    data = {
        "id": ids,
        "text": [f"t{i}" for i in ids],
    }
    if source is not None:
        data["source"] = [source] * num_examples

    return Dataset.from_dict(data).to_iterable_dataset(num_shards=num_shards)


def _flatten_batches(dataset, key: str) -> list[int]:
    out: list[int] = []
    for batch in dataset:
        out.extend(np.asarray(batch[key]).reshape(-1).tolist())
    return out


def test_training_cpu_only():
    ds = text_iterable_dataset(8)
    dl = make_dataloader(
        datasets=ds,
        transforms=[SimpleTokenize(max_length=4)],
        global_batch_size=4,
        dataloading_host_index=0,
        dataloading_host_count=1,
        shuffle=False,
        worker_count=0,
        drop_remainder=True,
    )

    batch0 = next(iter(dl))
    assert batch0["id"].tolist() == [0, 1, 2, 3]
    assert batch0["input_ids"].shape == (4, 4)
    assert batch0["input_ids"].dtype == np.int32
    assert batch0["input_ids"][:, 0].tolist() == [ord("t")] * 4
    assert batch0["input_ids"][:, 1].tolist() == [
        ord("0"),
        ord("1"),
        ord("2"),
        ord("3"),
    ]
    assert batch0["input_ids"][:, 2:].tolist() == [[0, 0]] * 4


def test_training_single_host_tpu():
    ds = text_iterable_dataset(4)

    mesh = Mesh(np.array([jax.devices()[0]]), ("data",))
    pspec = PartitionSpec("data")

    dl = make_dataloader(
        datasets=ds,
        transforms=[SimpleTokenize(max_length=4)],
        global_batch_size=4,
        dataloading_host_index=0,
        dataloading_host_count=1,
        shuffle=False,
        worker_count=0,
        drop_remainder=True,
        mesh=mesh,
        pspec=pspec,
    )

    batch0 = next(iter(dl))
    assert isinstance(batch0["input_ids"], jax.Array)
    assert batch0["input_ids"].shape == (4, 4)
    assert batch0["input_ids"].sharding.spec == pspec


# def test_training_multihost_tpu():
#     ds = _text_iterable_dataset(8, num_shards=2)

#     dl0 = make_dataloader_from_huggingface(
#         datasets=ds,
#         transforms=[SimpleTokenize(max_length=4)],
#         global_batch_size=4,
#         dataloading_host_index=0,
#         dataloading_host_count=2,
#         shuffle=False,
#         worker_count=0,
#         drop_remainder=True,
#     )
#     dl1 = make_dataloader_from_huggingface(
#         datasets=ds,
#         transforms=[SimpleTokenize(max_length=4)],
#         global_batch_size=4,
#         dataloading_host_index=1,
#         dataloading_host_count=2,
#         shuffle=False,
#         worker_count=0,
#         drop_remainder=True,
#     )

#     ids0 = _flatten_batches(dl0, "id")
#     ids1 = _flatten_batches(dl1, "id")

# assert ids0 == [0, 1, 2, 3]
# assert ids1 == [4, 5, 6, 7]
# assert set(ids0).isdisjoint(ids1)
# assert sorted(ids0 + ids1) == list(range(8))


def test_training_mix_two_datasets_cpu_only():
    ds_a = text_iterable_dataset(4, source=0)
    ds_b = text_iterable_dataset(4, source=1)

    dl = make_dataloader(
        datasets=[ds_a, ds_b],
        transforms=[SimpleTokenize(max_length=4)],
        global_batch_size=2,
        dataloading_host_index=0,
        dataloading_host_count=1,
        shuffle=False,
        worker_count=0,
        drop_remainder=True,
    )

    it = iter(dl)
    batch0 = next(it)
    batch1 = next(it)

    assert batch0["source"].tolist() == [0, 1]
    assert batch0["id"].tolist() == [0, 0]
    assert batch1["source"].tolist() == [0, 1]
    assert batch1["id"].tolist() == [1, 1]
