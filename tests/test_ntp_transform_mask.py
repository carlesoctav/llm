import numpy as np
import pytest

from jaxformers.data.transforms.ntp import NestInputs, TokenizeText


class DummyTextTokenizer:
    def __call__(
        self,
        text,
        *,
        truncation,
        padding,
        max_length,
        return_tensors,
        return_attention_mask,
        return_token_type_ids,
    ):
        del text, truncation, padding, return_tensors, return_attention_mask
        del return_token_type_ids
        return {
            "input_ids": np.array([[1, 2, 3, 4]], dtype=np.int32),
            "attention_mask": np.array([[1, 1, 1, 0]], dtype=np.int32),
        }


class DummyChatTokenizer:
    def __init__(self, assistant_masks):
        self.assistant_masks = np.array([assistant_masks], dtype=np.int32)

    def apply_chat_template(
        self,
        text,
        *,
        truncation,
        padding,
        max_length,
        chat_template,
        return_tensors,
        return_assistant_tokens_mask,
    ):
        del text, truncation, padding, max_length, chat_template
        del return_tensors, return_assistant_tokens_mask
        return {
            "input_ids": np.array([[1, 2, 3, 4]], dtype=np.int32),
            "attention_mask": np.array([[1, 1, 1, 0]], dtype=np.int32),
            "assistant_masks": self.assistant_masks,
        }


def test_tokenize_text_sets_mask_from_attention_mask():
    output = TokenizeText(
        column="text",
        tokenizer=DummyTextTokenizer(),
        packing=False,
        data_type="text",
        max_length=3,
    ).map({"text": "hello"})

    np.testing.assert_array_equal(output["loss_mask"], output["attention_mask"])


def test_tokenize_text_sets_mask_from_assistant_mask():
    output = TokenizeText(
        column="text",
        tokenizer=DummyChatTokenizer([0, 1, 1, 0]),
        packing=False,
        data_type="chat",
        assistant_loss=True,
        max_length=3,
    ).map({"text": [{"role": "user", "content": "hi"}]})

    np.testing.assert_array_equal(
        output["loss_mask"], output["attention_mask"] * output["assistant_masks"]
    )


def test_tokenize_text_raises_without_assistant_tokens():
    with pytest.raises(RuntimeError, match="assistant_loss=True was requested"):
        TokenizeText(
            column="text",
            tokenizer=DummyChatTokenizer([0, 0, 0, 0]),
            packing=False,
            data_type="chat",
            assistant_loss=True,
            max_length=3,
        ).map({"text": [{"role": "user", "content": "hi"}]})


def test_nest_inputs_keeps_loss_mask_at_batch_root():
    features = {
        "input_ids": np.array([1, 2, 3], dtype=np.int32),
        "attention_mask": np.array([1, 1, 0], dtype=np.int32),
        "labels": np.array([2, 3, 4], dtype=np.int32),
        "loss_mask": np.array([1, 1, 0], dtype=np.int32),
    }

    output = NestInputs().map(features)

    np.testing.assert_array_equal(output["loss_mask"], features["loss_mask"])
    np.testing.assert_array_equal(
        output["inputs"]["attention_mask"], features["attention_mask"]
    )


def test_nest_inputs_uses_attention_mask_when_mask_is_missing():
    features = {
        "input_ids": np.array([1, 2, 3], dtype=np.int32),
        "attention_mask": np.array([1, 1, 0], dtype=np.int32),
        "labels": np.array([2, 3, 4], dtype=np.int32),
    }

    output = NestInputs().map(features)

    np.testing.assert_array_equal(output["loss_mask"], features["attention_mask"])
