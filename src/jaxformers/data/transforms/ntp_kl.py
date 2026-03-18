import dataclasses as dc
import typing as tp
from dataclasses import dataclass
from enum import auto, StrEnum

import grain
from grain import transforms as grain_transforms
from jaxtyping import Array
from transformers import PreTrainedTokenizerBase

from .base import DatasetTransforms


class DataType(StrEnum):
    CHAT = auto()


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
    kl_column: str
    tokenizer: PreTrainedTokenizerBase
    packing: bool
    data_type: str
    chat_template: str | None = None
    kl_chat_template: str | None = None
    assistant_loss: bool = False
    max_length: int | None = None
    kl_max_length: int | None = None

    def map(self, features: dict[str, tp.Any]) -> dict[str, Array]:
        if self.column not in features:
            raise KeyError(f"Column {self.column!r} not found in element")
        text = features[self.column]
        kl_text = features[self.kl_column]
        encoded = self.tokenizer.apply_chat_template(
            text,
            truncation=self.max_length is not None,
            padding="max_length" if not self.packing else "do_not_pad",
            max_length=self.max_length + 1,
            chat_template=self.chat_template,
            return_tensors="np",
            return_assistant_tokens_mask=self.assistant_loss,
        )
        kl_encoded = self.tokenizer.apply_chat_template(
            kl_text,
            truncation=self.kl_max_length is not None,
            padding="max_length" if not self.packing else "do_not_pad",
            max_length=self.kl_max_length,
            chat_template=self.kl_chat_template,
            return_tensors="np",
            return_assistant_tokens_mask=self.assistant_loss,
        )

        output = {k: v.squeeze(0)[:-1] for k, v in encoded.items()}
        output.update({f"kl_{k}": v.squeeze(0) for k, v in kl_encoded.items()})
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

        kl_inputs: dict[str, tp.Any] = {
            "input_ids": features["kl_input_ids"],
            "attention_mask": features["kl_attention_mask"],
        }

        if "input_ids_segment_ids" in features:
            inputs["segment_ids"] = features["input_ids_segment_ids"]
        if "input_ids_segment_positions" in features:
            inputs["segment_positions"] = features["input_ids_segment_positions"]
        if "assistant_masks" in features:
            inputs["assistant_masks"] = features["assistant_masks"]

        if "kl_input_ids_segment_ids" in features:
            kl_inputs["segment_ids"] = features["kl_input_ids_segment_ids"]
        if "kl_input_ids_segment_positions" in features:
            kl_inputs["segment_positions"] = features["kl_input_ids_segment_positions"]
        if "kl_assistant_masks" in features:
            kl_inputs["assistant_masks"] = features["kl_assistant_masks"]
        return {"inputs": inputs, "labels": features["labels"], "kl_inputs": kl_inputs}


def make(
    column: str,
    kl_column: str,
    max_length: int,
    kl_max_length: int,
    data_type: str = "chat",
    tokenizer: PreTrainedTokenizerBase | None = None,
    chat_template_path: str | None = None,
    kl_chat_template_path: str | None = None,
    assistant_loss: bool = False,
    packing: bool = False,
    packing_bins: int | None = None,
) -> list[grain_transforms.Map | grain_transforms.RandomMap | DatasetTransforms]:
    """Build the list of transforms required for next-token prediction."""

    if data_type is None:
        raise ValueError("data_type is required")
    if data_type not in tuple(DataType):
        raise ValueError(
            f"Unsupported data_type {data_type!r}. Expected one of {tuple(DataType)!r}."
        )
    if tokenizer is None:
        raise ValueError("tokenizer is required")

    transforms = []
    chat_template = None
    kl_chat_template = None
    if chat_template_path:
        with open(chat_template_path) as f:
            chat_template = f.read()
    if kl_chat_template_path:
        with open(kl_chat_template_path) as f:
            kl_chat_template = f.read()

    if chat_template:
        preview = chat_template.replace("\n", "\\n")
        preview_short = (preview[:120] + "...") if len(preview) > 120 else preview
        print(f"Using custom chat_template (preview): '{preview_short}'")
    if kl_chat_template:
        preview = kl_chat_template.replace("\n", "\\n")
        preview_short = (preview[:120] + "...") if len(preview) > 120 else preview
        print(f"Using custom kl_chat_template (preview): '{preview_short}'")
    if not chat_template and not kl_chat_template:
        print(
            "No custom chat_template and kl_chat_template provided; using default chat formatting."
        )

    transforms.append(
        TokenizeText(
            column=column,
            kl_column=kl_column,
            tokenizer=tokenizer,
            max_length=max_length,
            kl_max_length=kl_max_length,
            data_type=data_type,
            chat_template=chat_template,
            kl_chat_template=kl_chat_template or chat_template,
            assistant_loss=assistant_loss,
            packing=packing,
        )
    )

    if packing:
        length_struct = {
            "input_ids": max_length,
            "attention_mask": max_length,
            "labels": max_length,
            **({"assistant_masks": max_length} if assistant_loss else {}),
            "kl_input_ids": max_length,
            "kl_attention_mask": max_length,
            **({"kl_assistant_masks": max_length} if assistant_loss else {}),
        }
        transforms.append(
            ApplyFirstFitPacking(
                length_struct=length_struct,
                num_packing_bins=packing_bins,
                meta_features=("attention_mask", "labels", "kl_attention_mask")
                + (("assistant_masks", "kl_assistant_mask") if assistant_loss else ()),
            )
        )
    transforms.append(NestInputs())
    return transforms
