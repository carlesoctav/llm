from __future__ import annotations

from dataclasses import dataclass

from jaxformers.inference.input_batch import InputBatch, SamplingParams
from jaxformers.inference.uni_worker import UniWorker
from jaxformers.models.qwen3 import InferenceModel


@dataclass
class ModelRunner:
    model: InferenceModel
    seed: int = 0

    def __post_init__(self):
        additional_config = self.model.config["additional_config"]
        max_num_seqs = additional_config.get("max_num_seqs")
        if max_num_seqs is None:
            max_num_seqs = additional_config.get("max_num_request")
        if max_num_seqs is None:
            raise KeyError("Model additional_config must include `max_num_seqs`.")
        self.input_batch = InputBatch(
            max_model_len=additional_config["max_model_len"],
            max_num_batched_token=additional_config["max_num_batched_token"],
            page_size=additional_config["page_size"],
            num_pages=additional_config["num_pages"],
            max_num_seqs=int(max_num_seqs),
        )
        self.worker = UniWorker(self.model, seed=self.seed)
        self.kv_cache = self.model.init_kv()
        self.kv_cache = self.worker.precompile(self.kv_cache)

    def add_request(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        req_id: str | None = None,
    ) -> str:
        return self.input_batch.add_request(
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            req_id=req_id,
        )

    def has_pending(self) -> bool:
        return not self.input_batch.is_finished()

    def step(self) -> list[tuple[str, list[int]]]:
        batch = self.input_batch.schedule()
        if batch is None:
            return []

        if not batch.should_sample:
            self.kv_cache = self.worker.run_backbone(batch, self.kv_cache)
            self.input_batch.commit_model_step(batch)
            return []

        _, sampled_token_ids, self.kv_cache = self.worker.run(batch, self.kv_cache)
        self.input_batch.commit_model_step(batch)
        self.input_batch.commit_sampled_tokens(
            batch,
            sampled_token_ids=[int(x) for x in sampled_token_ids.tolist()],
            eos_token_id=self.model.config["eos_token_id"],
        )
        return self.input_batch.pop_finished()
