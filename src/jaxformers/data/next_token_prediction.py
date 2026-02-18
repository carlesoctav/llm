import dataclasses as dc
import typing as tp
from dataclasses import dataclass

import grain
import jax.tree_util as jtu
import numpy as np
from grain import transforms as grain_transforms
from jaxtyping import Array
from transformers import PreTrainedTokenizerBase

from jaxformers.data.transforms import DatasetTransforms


@dataclass
class ApplyFirstFitPacking(DatasetTransforms):
    """Apply Grain first-fit packing transformation."""

    length_struct: dict[str, int]
    num_packing_bins: int | None = None
    shuffle_bins: bool = True
    meta_features: tp.Sequence[str] = ()

    def __call__(self, dataset: grain.IterDataset) -> grain.IterDataset:
        bins = self.num_packing_bins or max(self.length_struct.values())
        packed = grain.experimental.FirstFitPackIterDataset(
            dataset,
            length_struct=self.length_struct,
            num_packing_bins=bins,
            shuffle_bins=self.shuffle_bins,
            meta_features=self.meta_features,
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
        output = {}
        encoded = self.tokenizer(
            text,
            truncation=self.max_length is not None,
            padding="max_length" if not self.packing else "do_not_pad",
            max_length=self.max_length,
            return_tensors="np",
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        output["input_ids"] = encoded["input_ids"].squeeze(0)[:-1]
        output["attention_mask"] = encoded["attention_mask"].squeeze(0)[:-1]
        output["labels"] = encoded["input_ids"].squeeze(0)[1:]
        return output


@dc.dataclass
class NestInputs(grain_transforms.Map):
    """Nest token arrays under an `inputs` dict for model consumption."""

    def map(self, features: dict[str, tp.Any]) -> dict[str, tp.Any]:
        if "inputs" in features:
            return features

        inputs: dict[str, tp.Any] = {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"],
        }
        if "input_ids_segment_ids" in features:
            inputs["segment_ids"] = features["input_ids_segment_ids"]
        return {"inputs": inputs, "labels": features["labels"]}


def transforms(
    column: str,
    max_length: int,
    tokenizer: PreTrainedTokenizerBase,
    is_tokenized: bool,
    packing: bool = False,
    packing_bins: int | None = None,
) -> list[grain_transforms.Map | grain_transforms.RandomMap | DatasetTransforms]:
    """Build the list of transforms required for next-token prediction."""

    transforms = []
    if not is_tokenized:
        transforms.append(
            TokenizeText(
                column=column,
                tokenizer=tokenizer,
                max_length=max_length + 1,
                packing=packing,
            )
        )
    if packing:
        length_struct = {
            "input_ids": max_length,
            "attention_mask": max_length,
            "labels": max_length,
        }
        transforms.append(
            ApplyFirstFitPacking(
                length_struct=length_struct,
                num_packing_bins=packing_bins,
                # These are redundant with `input_ids_segment_ids` /
                # `input_ids_positions` and just bloat each batch.
                meta_features=("attention_mask", "labels"),
            )
        )
    transforms.append(NestInputs())
    return transforms
