import dataclasses as dc

import numpy as np
from datasets import Dataset
from grain import transforms as grain_transforms

from jaxformers.data import make_data, make_loader, make_transforms
from jaxformers.data.source.huggingface import HuggingFaceSourceIterDataset


@dc.dataclass
class IdentityMap(grain_transforms.Map):
    def map(self, features: dict) -> dict:
        return features


def test_make_loader_simple_uses_named_loader():
    dataset = HuggingFaceSourceIterDataset(
        Dataset.from_dict({"id": [0, 1]}).to_iterable_dataset()
    )
    loader = make_loader(
        "simple",
        dataset,
        [IdentityMap()],
        {
            "batch_size": 2,
            "shard": False,
            "shuffle": False,
            "num_workers": 0,
        },
    )

    batch = next(iter(loader))
    assert batch["id"].tolist() == [0, 1]


class DummyTokenizer:
    def __call__(
        self,
        text,
        *,
        truncation,
        padding,
        max_length,
        return_tensors,
        return_attention_mask,
        return_token_type_ids,
    ):
        del text, truncation, padding, return_tensors, return_attention_mask
        del return_token_type_ids
        return {
            "input_ids": np.ones((1, max_length), dtype=np.int32),
            "attention_mask": np.ones((1, max_length), dtype=np.int32),
        }


def test_make_transforms_ntp_supports_new_data_type():
    transforms = make_transforms(
        "ntp",
        {
            "column": "text",
            "max_length": 4,
            "tokenizer": DummyTokenizer(),
            "data_type": "text",
            "packing": False,
        },
    )

    assert len(transforms) == 2


def test_make_data_uses_single_group_loader(monkeypatch):
    dataset = HuggingFaceSourceIterDataset(
        Dataset.from_dict({"id": [0, 1]}).to_iterable_dataset()
    )

    def fake_make_source(source_name, source_config, *, streaming=False):
        del source_name, source_config, streaming
        return [dataset]

    monkeypatch.setattr("jaxformers.data.make_source", fake_make_source)

    loader = make_data(
        {
            "streaming": False,
            "loader": {"num_workers": 0, "shard": False},
            "train": {
                "source": {"load_kwargs": []},
                "transforms": [IdentityMap()],
                "loader": {"batch_size": 2, "shuffle": False},
            },
        }
    )

    batch = next(iter(loader))
    assert batch["id"].tolist() == [0, 1]


def test_make_data_uses_zip_loader(monkeypatch):
    datasets = {
        "sft": HuggingFaceSourceIterDataset(
            Dataset.from_dict({"id": [0, 1]}).to_iterable_dataset()
        ),
        "kl": HuggingFaceSourceIterDataset(
            Dataset.from_dict({"id": [10, 11]}).to_iterable_dataset()
        ),
    }

    def fake_make_source(source_name, source_config, *, streaming=False):
        del source_name, streaming
        return [datasets[source_config["name"]]]

    monkeypatch.setattr("jaxformers.data.make_source", fake_make_source)

    loader = make_data(
        {
            "streaming": False,
            "loader": {"combine": "zip", "num_workers": 0, "shard": False},
            "sft": {
                "source": {"name": "sft"},
                "transforms": [IdentityMap()],
                "loader": {"batch_size": 2, "shuffle": False},
            },
            "kl": {
                "source": {"name": "kl"},
                "transforms": [IdentityMap()],
                "loader": {"batch_size": 1, "shuffle": False},
            },
        }
    )

    sft_batch, kl_batch = next(iter(loader))
    assert sft_batch["id"].tolist() == [0, 1]
    assert kl_batch["id"].tolist() == [10]


def test_make_data_uses_mix_loader(monkeypatch):
    datasets = {
        "a": HuggingFaceSourceIterDataset(
            Dataset.from_dict({"id": [0, 1]}).to_iterable_dataset()
        ),
        "b": HuggingFaceSourceIterDataset(
            Dataset.from_dict({"id": [10, 11]}).to_iterable_dataset()
        ),
    }

    def fake_make_source(source_name, source_config, *, streaming=False):
        del source_name, streaming
        return [datasets[source_config["name"]]]

    monkeypatch.setattr("jaxformers.data.make_source", fake_make_source)

    loader = make_data(
        {
            "streaming": False,
            "loader": {"combine": "mix", "num_workers": 0, "shard": False},
            "a": {
                "source": {"name": "a"},
                "transforms": [IdentityMap()],
                "loader": {"batch_size": 2, "shuffle": False},
            },
            "b": {
                "source": {"name": "b"},
                "transforms": [IdentityMap()],
                "loader": {"batch_size": 1, "shuffle": False},
            },
        }
    )

    it = iter(loader)
    batch0 = next(it)
    batch1 = next(it)
    batch_sizes = sorted([len(batch0["id"]), len(batch1["id"])])
    assert batch_sizes == [1, 2]
