from jaxformers.data.transforms import DatasetTransforms
import dataclasses as dc
import typing as tp

import numpy as np

import grain
from grain import transforms as grain_transforms
from jaxtyping import Array
from transformers import PreTrainedTokenizerBase
import jax.tree_util as jtu
from dataclasses import dataclass

@dataclass
class ApplyFirstFitPacking(DatasetTransforms):
    """Apply Grain first-fit packing transformation."""

    length_struct: dict[str, int]
    num_packing_bins: int | None = None
    shuffle_bins: bool = True

    def __call__(
        self, dataset: grain.IterDataset
    ) -> grain.IterDataset:
        bins = self.num_packing_bins or max(self.length_struct.values())
        packed = grain.experimental.FirstFitPackIterDataset(
            dataset,
            length_struct=self.length_struct,
            num_packing_bins=bins,
            shuffle_bins=self.shuffle_bins,
        )
        return packed

@dc.dataclass
class TokenizeText(grain_transforms.Map):
    """Tokenize raw text coming from a column."""

    column: str
    tokenizer: PreTrainedTokenizerBase
    packing: bool
    max_length: int | None = None

    def map(self, features: dict[str, tp.Any]) -> dict[str, Array]:
        if self.column not in features:
            raise KeyError(f"Column {self.column!r} not found in element")
        text = features[self.column]
        encoded = self.tokenizer(
            text,
            truncation=self.max_length is not None,
            padding="max_length" if not self.packing else None,
            max_length=self.max_length,
            return_tensors = "np",
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        encoded["input_ids"] = encoded["input_ids"].squeeze(0)
        encoded["attention_mask"] = encoded["attention_mask"].squeeze(0)
        return encoded

def next_token_prediction_transforms(
    dataset_type: str,
    column: str,
    max_length: int,
    tokenizer: PreTrainedTokenizerBase,
    is_tokenized: bool,
    packing: bool = False,
    packing_bins: int | None = None,
) ->  list[grain_transforms.Map | grain_transforms.RandomMap | DatasetTransforms]:
    """Build the list of transforms required for next-token prediction."""

    transforms  = []
    if not is_tokenized:
        transforms.append(
            TokenizeText(
                column=column,
                tokenizer=tokenizer,
                max_length=max_length,
                packing=packing,
            )
        )
    if packing:
        length_struct = {"input_ids": max_length}
        transforms.append(
            ApplyFirstFitPacking(
                length_struct=length_struct, num_packing_bins=packing_bins
            )
        )
    return transforms
