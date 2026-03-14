from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from .base import DatasetWithReward, load_huggingface_dataset


_HASH_ANSWER_RE = re.compile(r"####\s*([^\n]+)")
_BOXED_ANSWER_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


class InputType(StrEnum):
    CHAT = "chat"
    COLUMN = "column"


def _extract_final_answer(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    boxed_answers = _BOXED_ANSWER_RE.findall(text)
    if boxed_answers:
        return boxed_answers[-1].strip()

    hash_answers = _HASH_ANSWER_RE.findall(text)
    if hash_answers:
        return hash_answers[-1].strip()

    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return " ".join(text.split())


def make(
    *,
    load_kwargs: dict[str, Any] | None = None,
    dataset_name: str = "gsm8k",
    question_column: str = "question",
    answer_column: str = "answer",
    prompt_template: str = "Question: {question}\nAnswer:",
    input_type: str = InputType.CHAT,
) -> DatasetWithReward:
    load_kwargs = load_kwargs or {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "train",
    }
    dataset = load_huggingface_dataset(load_kwargs)

    def format_example(example: dict[str, Any]) -> dict[str, Any]:
        question = str(example[question_column])
        prompt = prompt_template.format(question=question)
        normalized_input_type = InputType(str(input_type).lower())
        if normalized_input_type is InputType.CHAT:
            content: str | list[dict[str, str]] = [
                {"role": "user", "content": prompt},
            ]
        else:
            content = prompt
        return {
            "type": normalized_input_type.value,
            "content": content,
            "name": dataset_name,
        }

    def reward_fn(example: dict[str, Any], rollout_sample) -> dict[str, Any]:
        prediction = _extract_final_answer(rollout_sample.output_text)
        gold = _extract_final_answer(str(example[answer_column]))
        correct = bool(prediction) and prediction == gold
        return {
            "reward": float(correct),
            "correct": correct,
            "prediction": prediction,
            "target": gold,
        }

    return DatasetWithReward(
        name=dataset_name,
        dataset=dataset,
        format_example_fn=format_example,
        reward_fn=reward_fn,
    )
