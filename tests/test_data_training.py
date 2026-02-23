from datasets import Dataset

def test_ntp_single():
    texts = ["saya makan nasi", "tinggal di indonesia"]
    hf_data = Dataset.from_list([{"text": t} for t in texts])
    assert len(hf_data) == 2
