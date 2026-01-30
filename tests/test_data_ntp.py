from jaxformers.data.next_token_prediction import next_token_prediction_transforms
from jaxformers.data import huggingface_dataset, make_dataloader_from_huggingface
from datasets import Dataset

def test_ntp_cpu():
    texts = ["saya makan nasi", "tinggal di indonesia"]
    hf_data = Dataset.from_list(texts)
    ntp_transforms = next_token_prediction_transforms()
    dataset = make_dataloader_from_huggingface(hf_data, ntp_transforms)
    data = next(iter(dataset))


def test_ntp_single_host_sharded():
    pass
