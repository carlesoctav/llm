import dataclasses as dc
import typing as tp
from dataclasses import dataclass
from enum import auto, StrEnum

import grain
import jax
import numpy as np
from grain import transforms as grain_transforms
from jaxtyping import Array
from transformers import PreTrainedTokenizerBase

from .base import DatasetTransforms


class DataType(StrEnum):
    CHAT = auto()
    TEXT = auto()
    TOKEN = auto()


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
    data_type: str
    chat_template: str | None = None
    assistant_loss: bool = False
    max_length: int | None = None

    def map(self, features: dict[str, tp.Any]) -> dict[str, Array]:
        if self.column not in features:
            raise KeyError(f"Column {self.column!r} not found in element")
        text = features[self.column]
        if self.data_type == DataType.CHAT:
            encoded = self.tokenizer.apply_chat_template(
                text,
                truncation=self.max_length is not None,
                padding="max_length" if not self.packing else "do_not_pad",
                max_length=self.max_length + 1,
                chat_template=self.chat_template,
                return_tensors="np",
                return_assistant_tokens_mask=self.assistant_loss,
            )
        else:
            encoded = self.tokenizer(
                text,
                truncation=self.max_length is not None,
                padding="max_length" if not self.packing else "do_not_pad",
                max_length=self.max_length + 1,
                return_tensors="np",
                return_attention_mask=True,
                return_token_type_ids=False,
            )
        output = {k: v.squeeze(0)[:-1] for k, v in encoded.items()}
        output["labels"] = encoded["input_ids"].squeeze(0)[1:]
        output["_mask"] = output["attention_mask"]
        if "assistant_masks" in output:
            if not np.any(output["assistant_masks"]) and jax.process_index() == 0:
                raise RuntimeError(
                    "assistant_loss=True was requested but no assistant token was found. "
                    "This can occur if the chat template does not distinguish assistant vs user tokens "
                    "or if truncation (max_length) removed the assistant token. "
                    "please fix this issue before proceeding"
                )
            output["_mask"] = output["_mask"] * output["assistant_masks"]
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
        if "assistant_masks" in features:
            inputs["assistant_masks"] = features["assistant_masks"]
        mask = features["_mask"] if "_mask" in features else features["attention_mask"]
        return {
            "inputs": inputs,
            "labels": features["labels"],
            "_mask": mask,
        }


def make(
    column: str,
    max_length: int,
    data_type: str | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    chat_template_path: str | None = None,
    assistant_loss: bool = False,
    packing: bool = False,
    packing_bins: int | None = 64,
) -> list[grain_transforms.Map | grain_transforms.RandomMap | DatasetTransforms]:
    """Build the list of transforms required for next-token prediction."""

    if data_type is None:
        raise ValueError("data_type is required")
    if data_type not in tuple(DataType):
        raise ValueError(
            f"Unsupported data_type {data_type!r}. Expected one of {tuple(DataType)!r}."
        )
    if data_type != DataType.TOKEN and tokenizer is None:
        raise ValueError(f"tokenizer is required unless data_type={DataType.TOKEN!r}")

    transforms = []
    chat_template = None
    if data_type == DataType.CHAT and chat_template_path:
        with open(chat_template_path) as f:
            chat_template = f.read()

    if data_type == DataType.CHAT and chat_template:
        preview = chat_template.replace("\n", "\\n")
        preview_short = (preview[:120] + "...") if len(preview) > 120 else preview
        print(f"Using custom chat_template (preview): '{preview_short}'")
    elif data_type == DataType.CHAT:
        print("No custom chat_template provided; using default chat formatting.")

    if data_type != DataType.TOKEN:
        transforms.append(
            TokenizeText(
                column=column,
                tokenizer=tokenizer,
                max_length=max_length,
                data_type=data_type,
                chat_template=chat_template,
                assistant_loss=assistant_loss,
                packing=packing,
            )
        )
    if packing:
        length_struct = {
            "input_ids": max_length,
            "attention_mask": max_length,
            "labels": max_length,
            **({"_mask": max_length} if data_type != DataType.TOKEN else {}),
            **({"assistant_masks": max_length} if assistant_loss else {}),
        }
        transforms.append(
            ApplyFirstFitPacking(
                length_struct=length_struct,
                num_packing_bins=packing_bins,
                meta_features=("attention_mask", "labels")
                + (("_mask",) if data_type != DataType.TOKEN else ())
                + (("assistant_masks",) if assistant_loss else ()),
            )
        )
    transforms.append(NestInputs())
    return transforms
