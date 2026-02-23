import grain
import numpy as np


class DummyNtpMapDataset(grain.MapDataset):
    """Simple synthetic next-token prediction dataset.

    Emits already-tokenized examples:
      - inputs.input_ids: int32[T]
      - inputs.attention_mask: int32[T]
      - labels: int32[T]
    """

    def __init__(
        self,
        *,
        seq_len: int,
        vocab_size: int,
        num_examples: int = 1024,
        seed: int = 0,
    ):
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if num_examples <= 0:
            raise ValueError("num_examples must be positive")
        self._seq_len = int(seq_len)
        self._vocab_size = int(vocab_size)
        self._num_examples = int(num_examples)
        self._seed = int(seed)

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.slice(index)

        rng = np.random.default_rng(self._seed + int(index))
        input_ids = rng.integers(
            0, self._vocab_size, size=(self._seq_len,), dtype=np.int32
        )
        labels = rng.integers(
            0, self._vocab_size, size=(self._seq_len,), dtype=np.int32
        )
        attention_mask = np.ones((self._seq_len,), dtype=np.int32)
        return {
            "inputs": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            "labels": labels,
        }


def load(load_kwargs: list[dict]):
    return [DummyNtpMapDataset(**kw) for kw in load_kwargs]

