from __future__ import annotations

from dataclasses import dataclass

from jaxformers.inference.input_batch import SamplingParams
from jaxformers.inference.model_runner import ModelRunner
from jaxformers.models.qwen3 import InferenceModel


@dataclass
class LLMEngine:
    model: InferenceModel
    seed: int = 0

    def __post_init__(self):
        self.runner = ModelRunner(self.model, seed=self.seed)
        self.tokenizer = self.model.tokenizer

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        req_id: str | None = None,
    ) -> str:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
        return self.runner.add_request(prompt, sampling_params=sampling_params, req_id=req_id)

    def step(self) -> list[tuple[str, list[int]]]:
        return self.runner.step()

    def is_finished(self) -> bool:
        return not self.runner.has_pending()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[dict]:
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)

        req_ids: list[str] = []
        for idx, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            req_id = self.add_request(prompt, sampling_params=sp, req_id=f"req-{idx}")
            req_ids.append(req_id)

        outputs: dict[str, list[int]] = {}
        while not self.is_finished():
            for req_id, token_ids in self.step():
                outputs[req_id] = token_ids

        return [
            {
                "request_id": req_id,
                "token_ids": outputs.get(req_id, []),
                "text": self.tokenizer.decode(outputs.get(req_id, [])),
            }
            for req_id in req_ids
        ]
