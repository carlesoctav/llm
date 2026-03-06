from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from datasets import load_dataset

from jaxformers.inference import LLM


ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def load(load_kwargs: list[dict[str, Any]]) -> list[Iterable[dict[str, Any]]]:
    return [load_dataset(**kwargs) for kwargs in load_kwargs]


def _normalize_answer(text: str) -> str:
    answer = text.strip()
    if "####" in answer:
        answer = answer.split("####")[-1]
    tag_match = ANSWER_TAG_RE.search(answer)
    if tag_match is not None:
        answer = tag_match.group(1)
    answer = answer.strip()
    numbers = NUMBER_RE.findall(answer)
    if numbers:
        answer = numbers[-1]
    return answer.replace(",", "").strip()


def _score_gsm8k(completion: str, answer: str) -> float:
    predicted = _normalize_answer(completion)
    target = _normalize_answer(answer)
    score = 1.0 if predicted and predicted == target else 0.0
    if ANSWER_TAG_RE.search(completion) is not None:
        score += 0.25
    return score


def compute_advantages(rewards: np.ndarray, num_generations: int) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32)
    grouped = rewards.reshape(-1, num_generations)
    means = grouped.mean(axis=-1, keepdims=True)
    stds = grouped.std(axis=-1, keepdims=True, ddof=1)
    return ((grouped - means) / (stds + 1e-4)).reshape(-1).astype(np.float32)


@dataclass
class GRPORolloutDataset:
    datasets: list[Iterable[dict[str, Any]]]
    llm: LLM
    pad_token_id: int
    prompt_batch_size: int
    num_generations: int
    max_tokens: int
    max_model_len: int
    prompt_column: str | None = None
    messages_column: str | None = None
    answer_column: str | None = None
    prompt_template: str | None = None
    system_prompt: str | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    drop_last: bool = True

    def __post_init__(self) -> None:
        if self.num_generations <= 1:
            raise ValueError("GRPO requires num_generations > 1.")
        if self.temperature <= 0.0:
            raise ValueError(
                "GRPO rollout requires temperature > 0 when num_generations > 1."
            )
        if self.max_model_len <= self.max_tokens:
            raise ValueError("max_model_len must be larger than max_tokens.")
        if self.prompt_column is None and self.messages_column is None:
            raise ValueError("Either prompt_column or messages_column must be set.")

    def __iter__(self):
        batch: list[dict[str, Any]] = []
        for example in itertools.chain.from_iterable(self.datasets):
            batch.append(example)
            if len(batch) == self.prompt_batch_size:
                yield self._rollout_batch(batch)
                batch = []
        if batch and not self.drop_last:
            yield self._rollout_batch(batch)

    def _build_prompt(self, example: dict[str, Any]) -> str | list[dict[str, str]]:
        if self.messages_column is not None:
            messages = [dict(message) for message in example[self.messages_column]]
            if self.system_prompt and not any(m.get("role") == "system" for m in messages):
                messages = [{"role": "system", "content": self.system_prompt}, *messages]
            return messages

        prompt = str(example[self.prompt_column])
        if self.prompt_template is not None:
            prompt = self.prompt_template.format(prompt=prompt, **example)
        if self.system_prompt:
            prompt = f"{self.system_prompt}\n\n{prompt}"
        return prompt

    def _rollout_batch(self, examples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        prompts = [self._build_prompt(example) for example in examples]
        outputs = self.llm.generate(
            prompts,
            max_tokens=self.max_tokens,
            num_generations=self.num_generations,
            truncate_prompt_tokens=self.max_model_len - self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            detokenize=True,
            use_tqdm=False,
        )

        expanded_examples = [
            example
            for example in examples
            for _ in range(self.num_generations)
        ]

        input_ids = []
        labels = []
        attention_mask = []
        loss_mask = []
        generation_mask = []
        completion_ids = []
        rewards = []
        completion_lengths = []
        prompt_lengths = []

        for example, output in zip(expanded_examples, outputs, strict=True):
            prompt_token_ids = list(output.prompt_token_ids or [])
            raw_completion_ids = list(output.completion_token_ids or [])[: self.max_tokens]
            available_completion = max(0, self.max_model_len - len(prompt_token_ids))
            completion_token_ids = raw_completion_ids[:available_completion]

            completion_buffer = np.full((self.max_tokens,), self.pad_token_id, dtype=np.int32)
            completion_buffer[: len(completion_token_ids)] = np.asarray(
                completion_token_ids,
                dtype=np.int32,
            )
            generation_buffer = np.zeros((self.max_tokens,), dtype=np.float32)
            generation_buffer[: len(completion_token_ids)] = 1.0

            combined = prompt_token_ids + completion_token_ids
            combined_buffer = np.full((self.max_model_len,), self.pad_token_id, dtype=np.int32)
            combined_buffer[: len(combined)] = np.asarray(combined, dtype=np.int32)

            valid = np.zeros((self.max_model_len,), dtype=np.float32)
            valid[: len(combined)] = 1.0
            generated = np.zeros((self.max_model_len,), dtype=np.float32)
            generated[len(prompt_token_ids) : len(combined)] = 1.0

            input_ids.append(combined_buffer[:-1])
            labels.append(combined_buffer[1:])
            attention_mask.append(valid[:-1])
            loss_mask.append(valid[1:] * generated[1:])
            generation_mask.append(generation_buffer)
            completion_ids.append(completion_buffer)
            prompt_lengths.append(len(prompt_token_ids))
            completion_lengths.append(len(completion_token_ids))

            answer = str(example.get(self.answer_column, "")) if self.answer_column else ""
            rewards.append(_score_gsm8k(output.text or "", answer))

        rewards_array = np.asarray(rewards, dtype=np.float32)
        return {
            "inputs": {
                "input_ids": np.asarray(input_ids, dtype=np.int32),
                "attention_mask": np.asarray(attention_mask, dtype=np.float32),
            },
            "labels": np.asarray(labels, dtype=np.int32),
            "loss_mask": np.asarray(loss_mask, dtype=np.float32),
            "completion_ids": np.asarray(completion_ids, dtype=np.int32),
            "generation_mask": np.asarray(generation_mask, dtype=np.float32),
            "advantages": compute_advantages(rewards_array, self.num_generations),
            "rewards": rewards_array,
            "prompt_lengths": np.asarray(prompt_lengths, dtype=np.int32),
            "completion_lengths": np.asarray(completion_lengths, dtype=np.int32),
            "old_logps": np.zeros((len(outputs), self.max_model_len - 1), dtype=np.float32),
            "ref_logps": np.zeros((len(outputs), self.max_model_len - 1), dtype=np.float32),
        }
