from __future__ import annotations

from dataclasses import dataclass

from jaxformers.inference.scheduler import Request
from jaxformers.inference.worker import EngineConfig, JaxWorker
from jaxformers.modeling_utils import Model


@dataclass(frozen=True)
class GenerateOutput:
    token_ids: list[int]
    generation_mask: list[bool]


class LLM:
    def __init__(
        self,
        model: Model,
        *,
        max_num_batched_tokens: int,
        max_num_seqs: int,
        max_model_len: int,
        page_size: int = 32,
        dtype=None,
        seed: int = 0,
        generation_mask_modes: list[str] | None = None,
    ) -> None:
        if dtype is None:
            dtype = model.weights["model.embed_tokens.weight"].dtype
        if generation_mask_modes is None:
            generation_mask_modes = ["last", "assistant"]

        self.model = model
        self.generation_mask_modes = generation_mask_modes
        self.worker = JaxWorker(
            model=model,
            config=EngineConfig(
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                page_size=page_size,
                dtype=dtype,
                seed=seed,
            ),
        )

    def _encode_str_prompt(self, prompt: str) -> tuple[list[int], list[bool]]:
        token_ids = self.model.tokenizer.encode(prompt)
        return token_ids, [False for _ in range(len(token_ids))]

    def _encode_chat_prompt(
        self, messages: list[dict], *, generation_mask_modes: list[str]
    ) -> tuple[list[int], list[bool]]:
        tok = self.model.tokenizer

        encoded_full = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="np",
            return_dict=True,
        )
        prompt_token_ids = encoded_full["input_ids"].squeeze(0).tolist()
        prompt_mask = [False for _ in range(len(prompt_token_ids))]

        if "assistant" not in generation_mask_modes:
            return prompt_token_ids, prompt_mask

        # Incremental encoding to attribute spans to each message.
        prev_len = 0
        for i in range(len(messages)):
            encoded_i = tok.apply_chat_template(
                messages[: i + 1],
                add_generation_prompt=False,
                return_tensors="np",
                return_dict=True,
            )
            token_ids_i = encoded_i["input_ids"].squeeze(0).tolist()
            cur_len = len(token_ids_i)

            role = messages[i]["role"]
            if role == "assistant":
                for j in range(prev_len, cur_len):
                    prompt_mask[j] = True

            prev_len = cur_len

        return prompt_token_ids, prompt_mask

    def generate(
        self,
        prompts: list[str] | list[list[int]] | list[list[dict]],
        *,
        max_tokens: int,
        ignore_eos: bool = True,
        temperature: float = 0.0,
    ) -> list[GenerateOutput]:
        eos_token_id = self.model.tokenizer.eos_token_id
        if eos_token_id is None:
            if not ignore_eos:
                raise ValueError("tokenizer.eos_token_id is None but ignore_eos=False")
            eos_token_id = -1
        eos_token_id = int(eos_token_id)

        requests: list[Request] = []
        if prompts:
            if isinstance(prompts[0], str):
                for i, prompt in enumerate(prompts):
                    prompt_token_ids, prompt_mask = self._encode_str_prompt(prompt)
                    requests.append(
                        Request(
                            request_id=i,
                            prompt_token_ids=prompt_token_ids,
                            prompt_generation_mask=prompt_mask,
                            max_tokens=max_tokens,
                            eos_token_id=eos_token_id,
                            ignore_eos=ignore_eos,
                        )
                    )
            elif isinstance(prompts[0], list) and prompts[0] and isinstance(prompts[0][0], int):
                for i, token_ids in enumerate(prompts):
                    requests.append(
                        Request(
                            request_id=i,
                            prompt_token_ids=token_ids,
                            prompt_generation_mask=[False for _ in range(len(token_ids))],
                            max_tokens=max_tokens,
                            eos_token_id=eos_token_id,
                            ignore_eos=ignore_eos,
                        )
                    )
            else:
                for i, messages in enumerate(prompts):
                    prompt_token_ids, prompt_mask = self._encode_chat_prompt(
                        messages, generation_mask_modes=self.generation_mask_modes
                    )
                    requests.append(
                        Request(
                            request_id=i,
                            prompt_token_ids=prompt_token_ids,
                            prompt_generation_mask=prompt_mask,
                            max_tokens=max_tokens,
                            eos_token_id=eos_token_id,
                            ignore_eos=ignore_eos,
                        )
                    )

        for req in requests:
            self.worker.add_request(req)

        finished = self.worker.run(temperature=temperature)

        outputs: list[GenerateOutput] = []
        for i in range(len(requests)):
            seq = finished[i]
            token_ids = seq.request.prompt_token_ids + seq.generated_token_ids
            generation_mask = seq.request.prompt_generation_mask + [
                True for _ in range(len(seq.generated_token_ids))
            ]
            outputs.append(
                GenerateOutput(
                    token_ids=token_ids,
                    generation_mask=generation_mask,
                )
            )
        return outputs
