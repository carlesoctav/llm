import dataclasses as dc
import typing as tp

from grain import transforms as grain_transforms
from jaxtyping import Array
from transformers import PreTrainedTokenizerBase

from .base import DatasetTransforms
from .ntp import ApplyFirstFitPacking, DataType


@dc.dataclass
class TokenizeText(grain_transforms.Map):
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
                max_length=self.max_length,
                chat_template=self.chat_template,
                return_tensors="np",
                return_assistant_tokens_mask=self.assistant_loss,
            )
        else:
            encoded = self.tokenizer(
                text,
                truncation=self.max_length is not None,
                padding="max_length" if not self.packing else "do_not_pad",
                max_length=self.max_length,
                return_tensors="np",
                return_attention_mask=True,
                return_token_type_ids=False,
            )

        return {k: v.squeeze(0) for k, v in encoded.items()}


@dc.dataclass
class NestInputs(grain_transforms.Map):
    def map(self, features: dict[str, tp.Any]) -> dict[str, tp.Any]:
        if "inputs" in features:
            return features

        inputs: dict[str, tp.Any] = {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"],
        }
        if "input_ids_segment_ids" in features:
            inputs["segment_ids"] = features["input_ids_segment_ids"]
        if "input_ids_segment_positions" in features:
            inputs["segment_positions"] = features["input_ids_segment_positions"]
        if "assistant_masks" in features:
            inputs["assistant_masks"] = features["assistant_masks"]
        return {"inputs": inputs}


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
            **({"assistant_masks": max_length} if assistant_loss else {}),
        }
        transforms.append(
            ApplyFirstFitPacking(
                length_struct=length_struct,
                num_packing_bins=packing_bins,
                meta_features=("attention_mask",)
                + (("assistant_masks",) if assistant_loss else ()),
            )
        )
    transforms.append(NestInputs())
    return transforms
