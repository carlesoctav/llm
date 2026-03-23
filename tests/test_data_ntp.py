import numpy as np
from datasets import Dataset

from jaxformers.data import make_loader, make_transforms
from jaxformers.data.source.huggingface import HuggingFaceSourceIterDataset


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


def test_ntp_cpu():
    hf_data = HuggingFaceSourceIterDataset(
        Dataset.from_list(
            [{"text": "saya makan nasi"}, {"text": "tinggal di indonesia"}]
        ).to_iterable_dataset()
    )
    ntp_transforms = make_transforms(
        "ntp",
        {
            "column": "text",
            "max_length": 8,
            "tokenizer": DummyTokenizer(),
            "data_type": "text",
            "packing": False,
        },
    )
    dataset = make_loader(
        "simple",
        hf_data,
        ntp_transforms,
        {
            "batch_size": 2,
            "shard": False,
            "shuffle": False,
            "num_workers": 0,
        },
    )
    data = next(iter(dataset))
    assert data["inputs"]["input_ids"].shape == (2, 8)


def test_ntp_single_host_sharded():
    pass
