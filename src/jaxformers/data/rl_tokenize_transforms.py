from __future__ import annotations

import dataclasses as dc
from enum import StrEnum
from pathlib import Path
from typing import Any

from grain import transforms as grain_transforms
from transformers import PreTrainedTokenizerBase


class RLInputType(StrEnum):
    CHAT = "chat"
    COLUMN = "column"


def _fallback_special_id(tokenizer: PreTrainedTokenizerBase) -> int:
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    return 0


@dc.dataclass
class TokenizeRLInput(grain_transforms.Map):
    tokenizer: PreTrainedTokenizerBase
    chat_template: str | None = None
    add_generation_prompt: bool = True

    def map(self, features: dict[str, Any]) -> dict[str, Any]:
        if "type" not in features:
            raise KeyError("RL examples must include a 'type' field.")
        if "content" not in features:
            raise KeyError("RL examples must include a 'content' field.")

        example_type = RLInputType(str(features["type"]).lower())
        content = features["content"]

        if example_type is RLInputType.CHAT:
            rendered_content = self.tokenizer.apply_chat_template(
                content,
                tokenize=False,
                chat_template=self.chat_template,
                add_generation_prompt=self.add_generation_prompt,
            )
            encoded = self.tokenizer(
                rendered_content,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
        else:
            rendered_content = str(content)
            encoded = self.tokenizer(
                rendered_content,
                add_special_tokens=True,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

        input_ids = [int(token_id) for token_id in encoded["input_ids"]]
        if not input_ids:
            input_ids = [_fallback_special_id(self.tokenizer)]

        output = dict(features)
        output["type"] = example_type.value
        output["content"] = rendered_content
        output["input_ids"] = input_ids
        return output


def make_tokenize_rl_input(
    *,
    tokenizer: PreTrainedTokenizerBase,
    chat_template_path: str | None = None,
    add_generation_prompt: bool = True,
) -> TokenizeRLInput:
    chat_template = None
    if chat_template_path:
        chat_template = Path(chat_template_path).read_text()

    return TokenizeRLInput(
        tokenizer=tokenizer,
        chat_template=chat_template,
        add_generation_prompt=add_generation_prompt,
    )
