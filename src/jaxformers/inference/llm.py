from __future__ import annotations

from contextlib import contextmanager
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from jaxformers.modeling_utils import Model


@dataclass(frozen=True)
class GenerateOutput:
    token_ids: list[int]
    generation_mask: list[bool]
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    text: str | None = None


class JaxformersLLM:
    def __init__(
        self,
        model: Model,
        *,
        max_num_batched_tokens: int,
        max_num_seqs: int,
        max_model_len: int,
        prefill_chunk_size: int | None = None,
        page_size: int = 64,
        dtype=None,
        seed: int = 0,
        generation_mask_modes: list[str] | None = None,
    ) -> None:
        from jaxformers.inference.worker import EngineConfig, JaxWorker

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
                prefill_chunk_size=prefill_chunk_size,
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
        from jaxformers.inference.sampling import SamplingParams
        from jaxformers.inference.scheduler import Request

        eos_token_id = self.model.tokenizer.eos_token_id
        if eos_token_id is None:
            if not ignore_eos:
                raise ValueError("tokenizer.eos_token_id is None but ignore_eos=False")
            eos_token_id = -1
        eos_token_id = int(eos_token_id)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=float(temperature),
            top_p=1.0,
            top_k=0,
            ignore_eos=ignore_eos,
        )

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
                            eos_token_id=eos_token_id,
                            sampling_params=sampling_params,
                        )
                    )
            elif isinstance(prompts[0], list) and prompts[0] and isinstance(prompts[0][0], int):
                for i, token_ids in enumerate(prompts):
                    requests.append(
                        Request(
                            request_id=i,
                            prompt_token_ids=token_ids,
                            prompt_generation_mask=[False for _ in range(len(token_ids))],
                            eos_token_id=eos_token_id,
                            sampling_params=sampling_params,
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
                            eos_token_id=eos_token_id,
                            sampling_params=sampling_params,
                        )
                    )

        for req in requests:
            self.worker.add_request(req)

        outputs: list[GenerateOutput] = []
        finished = self.worker.run(temperature=float(temperature))
        for i in range(len(requests)):
            token_ids, generation_mask = finished[i]
            prompt_token_ids = requests[i].prompt_token_ids
            completion_token_ids = [t for t, m in zip(token_ids, generation_mask) if m]
            outputs.append(
                GenerateOutput(
                    token_ids=token_ids,
                    generation_mask=generation_mask,
                    prompt_token_ids=prompt_token_ids,
                    completion_token_ids=completion_token_ids,
                )
            )
        return outputs


class LLM:
    """vLLM-TPU backed LLM wrapper that returns token ids + a generation mask.

    This uses vLLM's Python API (not the OpenAI server) so we can directly
    capture `prompt_token_ids` and generated `token_ids` without retokenizing.
    """

    def __init__(
        self,
        model: str,
        *,
        tokenizer: str | None = None,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
        tpu_memory_utilization: float | None = None,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        seed: int = 0,
        trust_remote_code: bool = False,
        download_dir: str | None = None,
        model_impl: str = "vllm",
        vllm_model_impl: str | None = "vllm",
        tpu_backend_type: str | None = None,
        jax_devices: list[int] | tuple[int, ...] | None = None,
        # vLLM's `LLM` exposes many more knobs via `**kwargs` (EngineArgs).
        **kwargs: Any,
    ) -> None:
        if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") is None:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from vllm import LLM as VllmLLM
        from vllm import envs as vllm_envs

        vllm_envs.VLLM_ENABLE_V1_MULTIPROCESSING = (
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] != "0"
        )

        if vllm_model_impl is not None:
            os.environ["MODEL_IMPL_TYPE"] = str(vllm_model_impl)
        if tpu_backend_type is not None:
            os.environ["TPU_BACKEND_TYPE"] = str(tpu_backend_type)

        self.tensor_parallel_size = int(tensor_parallel_size)
        engine_kwargs: dict[str, Any] = dict(kwargs)
        additional_config = dict(engine_kwargs.pop("additional_config", {}) or {})
        if max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
        if max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = int(max_num_seqs)
        if max_model_len is not None:
            engine_kwargs["max_model_len"] = int(max_model_len)
        if gpu_memory_utilization is None:
            gpu_memory_utilization = tpu_memory_utilization
        if gpu_memory_utilization is not None:
            engine_kwargs["gpu_memory_utilization"] = float(gpu_memory_utilization)
        engine_kwargs.setdefault("async_scheduling", False)
        if jax_devices:
            sharding_config = dict(additional_config.get("sharding", {}) or {})
            sharding_strategy = dict(sharding_config.get("sharding_strategy", {}) or {})
            sharding_strategy["device_indexes"] = [int(device) for device in jax_devices]
            sharding_config["sharding_strategy"] = sharding_strategy
            additional_config["sharding"] = sharding_config
        if additional_config:
            engine_kwargs["additional_config"] = additional_config
        engine_kwargs["model_impl"] = model_impl

        with self._suspend_global_jax_mesh():
            self._llm = VllmLLM(
                model=model,
                tokenizer=tokenizer,
                tensor_parallel_size=self.tensor_parallel_size,
                dtype=dtype,
                seed=int(seed),
                trust_remote_code=trust_remote_code,
                download_dir=download_dir,
                **engine_kwargs,
            )

    def sync_jaxformers_weights(self, model: Model | dict[str, Any]) -> None:
        from jaxformers.inference.vllm_sync import (
            build_gemma3_vllm_sync_payload,
            reshard_like_vllm_state,
            sync_vllm_state_in_place,
        )

        payload = build_gemma3_vllm_sync_payload(
            model,
            tensor_parallel_size=self.tensor_parallel_size,
        )
        model_runner = self._try_get_inprocess_model_runner()
        if model_runner is not None and getattr(model_runner, "state", None) is not None:
            with self._suspend_global_jax_mesh():
                model_runner.state = sync_vllm_state_in_place(
                    payload.updated_weights,
                    model_runner.state,
                    payload.mappings,
                    payload.transpose_keys,
                )
                if hasattr(model_runner, "execute_model_state"):
                    model_runner.execute_model_state = None
                if hasattr(model_runner, "_pre_async_results"):
                    model_runner._pre_async_results = None
                self._reset_vllm_caches()
            return
        with self._suspend_global_jax_mesh():
            self._llm.collective_rpc(
                "sync_weights",
                args=(
                    payload.updated_weights,
                    payload.mappings,
                    payload.transpose_keys,
                    reshard_like_vllm_state,
                ),
            )
            self._reset_vllm_caches()

    def _try_get_inprocess_model_runner(self) -> Any | None:
        candidates = (
            ("llm_engine", "engine_core", "model_executor", "driver_worker", "model_runner"),
            (
                "llm_engine",
                "engine_core",
                "engine_core",
                "model_executor",
                "driver_worker",
                "model_runner",
            ),
        )
        for chain in candidates:
            obj = self._llm
            for attr in chain:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        return None

    def _reset_vllm_caches(self) -> None:
        reset_prefix_cache = getattr(self._llm, "reset_prefix_cache", None)
        if callable(reset_prefix_cache):
            reset_prefix_cache(reset_running_requests=True, reset_connector=True)

    @contextmanager
    def _suspend_global_jax_mesh(self):
        try:
            from jax import sharding as jax_sharding
        except Exception:
            yield
            return

        previous_mesh = jax_sharding.get_mesh()
        empty_mesh = jax_sharding.Mesh(np.empty((), dtype=object), ())
        if previous_mesh.axis_names:
            jax_sharding.set_mesh(empty_mesh)
        try:
            yield
        finally:
            if previous_mesh.axis_names:
                jax_sharding.set_mesh(previous_mesh)

    def generate(
        self,
        prompts: list[str] | list[list[int]] | list[list[dict]],
        *,
        max_tokens: int,
        ignore_eos: bool = True,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        num_generations: int = 1,
        truncate_prompt_tokens: int | None = None,
        detokenize: bool = False,
        use_tqdm: bool = False,
        chat_template: str | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[GenerateOutput]:
        from vllm import SamplingParams

        if not prompts:
            return []

        sampling_params = SamplingParams(
            n=int(num_generations),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            ignore_eos=bool(ignore_eos),
            truncate_prompt_tokens=(
                int(truncate_prompt_tokens) if truncate_prompt_tokens is not None else None
            ),
            detokenize=bool(detokenize),
        )

        request_outputs: list[Any]
        if isinstance(prompts[0], str):
            with self._suspend_global_jax_mesh():
                request_outputs = self._llm.generate(
                    prompts,
                    sampling_params=sampling_params,
                    use_tqdm=use_tqdm,
                    tokenization_kwargs=tokenization_kwargs,
                )
        elif isinstance(prompts[0], list) and prompts[0] and isinstance(prompts[0][0], int):
            vllm_prompts = [{"prompt_token_ids": [int(x) for x in token_ids]} for token_ids in prompts]
            with self._suspend_global_jax_mesh():
                request_outputs = self._llm.generate(
                    vllm_prompts,
                    sampling_params=sampling_params,
                    use_tqdm=use_tqdm,
                )
        else:
            with self._suspend_global_jax_mesh():
                request_outputs = self._llm.chat(
                    prompts,
                    sampling_params=sampling_params,
                    use_tqdm=use_tqdm,
                    chat_template=chat_template,
                    tokenization_kwargs=tokenization_kwargs,
                )

        outputs: list[GenerateOutput] = []
        for req_out in request_outputs:
            prompt_token_ids = [int(x) for x in (req_out.prompt_token_ids or [])]
            for completion in req_out.outputs:
                completion_token_ids = [int(x) for x in completion.token_ids]
                if len(completion_token_ids) > max_tokens:
                    completion_token_ids = completion_token_ids[:max_tokens]
                token_ids = prompt_token_ids + completion_token_ids
                generation_mask = [False] * len(prompt_token_ids) + [True] * len(
                    completion_token_ids
                )
                outputs.append(
                    GenerateOutput(
                        token_ids=token_ids,
                        generation_mask=generation_mask,
                        prompt_token_ids=prompt_token_ids,
                        completion_token_ids=completion_token_ids,
                        text=completion.text if detokenize else None,
                    )
                )
        return outputs
