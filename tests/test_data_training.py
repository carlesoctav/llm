from jaxformers.data import huggingface_dataset
from datasets import Dataset

def test_ntp_single():
    texts = ["saya makan nasi", "tinggal di indonesia"]
    hf_data = Dataset.from_list(texts)
    pass
