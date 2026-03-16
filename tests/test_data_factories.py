import dataclasses as dc

import numpy as np
from datasets import Dataset
from grain import transforms as grain_transforms

from jaxformers.data import make_loader, make_transforms


@dc.dataclass
class IdentityMap(grain_transforms.Map):
    def map(self, features: dict) -> dict:
        return features


def test_make_loader_simple_uses_named_loader():
    dataset = Dataset.from_dict({"id": [0, 1]}).to_iterable_dataset()
    loader = make_loader(
        "simple",
        dataset,
        [IdentityMap()],
        {
            "global_batch_size": 2,
            "dataloading_host_index": 0,
            "dataloading_host_count": 1,
            "shuffle": False,
            "worker_count": 0,
            "drop_remainder": True,
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
