import numpy as np

from datasets import Dataset

from jaxformers.data.dummy import DummyNtpMapDataset
from jaxformers.data.next_token_prediction import transforms as ntp_transforms
from jaxformers.data.training import make_dataloader


def test_dummy_ntp_dataloader_accepts_no_transforms():
    ds = DummyNtpMapDataset(seq_len=8, vocab_size=32, num_examples=4, seed=0)
    dl = make_dataloader(
        datasets=ds,
        transforms=None,
        global_batch_size=2,
        dataloading_host_index=0,
        dataloading_host_count=1,
        shuffle=False,
        worker_count=0,
        drop_remainder=True,
    )

    batch = next(iter(dl))
    assert batch["inputs"]["input_ids"].shape == (2, 8)
    assert batch["inputs"]["input_ids"].dtype == np.int32
    assert batch["labels"].shape == (2, 8)
    assert batch["labels"].dtype == np.int32


def test_ntp_nest_inputs_transform_on_tokenized_hf_dataset():
    hf_data = Dataset.from_dict(
        {
            "input_ids": [np.asarray([1, 2, 3, 4], dtype=np.int32)] * 4,
            "attention_mask": [np.asarray([1, 1, 1, 1], dtype=np.int32)] * 4,
            "labels": [np.asarray([2, 3, 4, 5], dtype=np.int32)] * 4,
        }
    )

    ops = ntp_transforms(
        column="text",
        max_length=4,
        tokenizer=None,
        is_tokenized=True,
        is_chat=False,
        packing=False,
    )

    dl = make_dataloader(
        datasets=hf_data,
        transforms=ops,
        global_batch_size=2,
        dataloading_host_index=0,
        dataloading_host_count=1,
        shuffle=False,
        worker_count=0,
        drop_remainder=True,
    )

    batch = next(iter(dl))
    input_ids = np.asarray(batch["inputs"]["input_ids"])
    attention_mask = np.asarray(batch["inputs"]["attention_mask"])
    labels = np.asarray(batch["labels"])
    assert input_ids.shape == (2, 4)
    assert attention_mask.shape == (2, 4)
    assert labels.shape == (2, 4)
