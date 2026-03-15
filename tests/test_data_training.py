from datasets import Dataset


def test_ntp_single():
    hf_data = Dataset.from_list(
        [{"text": "saya makan nasi"}, {"text": "tinggal di indonesia"}]
    )
    assert len(hf_data) == 2
