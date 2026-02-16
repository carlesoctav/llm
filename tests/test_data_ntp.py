from jaxformers.data.next_token_prediction import transforms
from jaxformers.data import huggingface_dataset, make_dataloader
from datasets import Dataset

def test_ntp_cpu():
    texts = ["saya makan nasi", "tinggal di indonesia"]
    hf_data = Dataset.from_list(texts)
    ntp_transforms = transforms()
    dataset = make_dataloader(hf_data, ntp_transforms)
    data = next(iter(dataset))


def test_ntp_single_host_sharded():
    pass
