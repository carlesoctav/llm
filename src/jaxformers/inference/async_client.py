from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from collections.abc import Mapping
from typing import Any, cast, TypeAlias

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import AsyncEngineArgs
from vllm.v1.executor import Executor

from jaxformers.async_utils import AsyncLoopThread
from jaxformers.module_utils import ToVllmMappingAbstract, VllmMapping


try:
    from verifiers.clients.client import Client as VerifiersClient
    from verifiers.errors import EmptyModelResponseError, InvalidModelResponseError
    from verifiers.types import (
        ClientConfig,
        Messages,
        Response,
        ResponseMessage,
        ResponseTokens,
        SamplingArgs,
        Tool,
        Usage,
    )
except ImportError:  # pragma: no cover
    VerifiersClient = object
    ClientConfig = Any
    Messages: TypeAlias = list[dict[str, Any]]
    SamplingArgs: TypeAlias = dict[str, Any]
    Tool: TypeAlias = dict[str, Any]

    class _FallbackModel(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as err:
                raise AttributeError(key) from err

    Response = _FallbackModel
    ResponseMessage = _FallbackModel
    ResponseTokens = _FallbackModel
    Usage = _FallbackModel
    EmptyModelResponseError = RuntimeError
    InvalidModelResponseError = RuntimeError


def _extract_logprobs(token_ids: list[int], sample_logprobs) -> list[float]:
    if sample_logprobs is None:
        return [0.0] * len(token_ids)

    values = []
    for token_id, position_logprobs in zip(token_ids, sample_logprobs):
        if not position_logprobs:
            values.append(0.0)
            continue
        if token_id in position_logprobs:
            values.append(position_logprobs[token_id].logprob)
            continue
        values.append(next(iter(position_logprobs.values())).logprob)
    if len(values) < len(token_ids):
        values.extend([0.0] * (len(token_ids) - len(values)))
    return values


def _flatten_vllm_mapping(mapping: VllmMapping) -> dict[str, jax.Array]:
    return {".".join(path): leaf.value for path, leaf in mapping.state.flat_state()}


def _coerce_prompt_ids(prompt_ids) -> list[int]:
    if isinstance(prompt_ids, Mapping):
        return list(prompt_ids["input_ids"])
    return list(prompt_ids)


def _sync_dict_state(
    target_state: dict[str, jax.Array],
    mapping: VllmMapping,
) -> dict[str, jax.Array]:
    updated_state = dict(target_state)
    for source_key, value in _flatten_vllm_mapping(mapping).items():
        target_key = mapping.mappings[source_key][0]
        if target_key not in target_state:
            prefixed_target_key = f"vllm_model.{target_key}"
            if prefixed_target_key not in target_state:
                raise KeyError(target_key)
            target_key = prefixed_target_key
        if source_key in mapping.transpose_keys:
            value = jnp.transpose(value, mapping.transpose_keys[source_key])
        target_value = target_state[target_key]
        if value.shape != target_value.shape:
            raise ValueError(
                f"Weight shape mismatch for {target_key}: "
                f"expected {target_value.shape}, got {value.shape}"
            )
        updated_state[target_key] = jax.device_put(
            jnp.array(value, dtype=target_value.dtype, copy=True),
            target_value.sharding,
        )
    return updated_state


@contextlib.contextmanager
def _mute_stdio():
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    with open(os.devnull, "w") as devnull:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


_mute_stdio = contextlib.nullcontext


class AsyncSameProcessTPUInferenceClient(VerifiersClient):
    def __init__(
        self,
        model: str,
        *,
        tokenizer: PreTrainedTokenizerBase | str | None = None,
        vllm_config: dict[str, Any],
        dummy=True,
    ) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        os.environ["MODEL_IMPL_TYPE"] = "flax_nnx"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
        # os.environ.setdefault("GLOG_minloglevel", "2")
        # os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

        llm_config = dict(vllm_config)
        llm_config["model"] = model
        # llm_config["load_format"] = "dummy"
        if "tokenizer" not in llm_config and isinstance(tokenizer, str):
            llm_config["tokenizer"] = tokenizer

        self.model_name = model
        with _mute_stdio():
            from vllm.v1.engine.async_llm import AsyncLLM

            engine_args = AsyncEngineArgs(**llm_config)
            vllm_config = engine_args.create_engine_config()
            executor_class = Executor.get_class(vllm_config)

            logging.getLogger("absl").setLevel(logging.ERROR)
            logging.getLogger("tpu_inference").setLevel(logging.ERROR)
            logging.getLogger("vllm").setLevel(logging.ERROR)

            self.tokenizer = (
                tokenizer
                if isinstance(tokenizer, PreTrainedTokenizerBase)
                else AutoTokenizer.from_pretrained(tokenizer or model)
            )
            self.llm = AsyncLLM(
                vllm_config,
                executor_class=executor_class,
                log_stats=True,
                use_uniproc_engine_core=True,
            )

        self._client = self.llm
        self._config = None
        self._count = 0
        self._update_w_counter = 0
        self._executor = AsyncLoopThread()

    def setup_client(self, config: ClientConfig):  # pragma: no cover
        del config
        raise TypeError(
            "SameProcessTPUInferenceClient must be constructed with make(...), "
            "not from a verifiers ClientConfig."
        )

    async def close(self) -> None:
        self._executor.close()
        self._client = None
        self.llm = None

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> Response:
        native_prompt, extra_kwargs = await self.to_native_prompt(prompt)
        native_tools = None
        if tools is not None:
            native_tools = [await self.to_native_tool(tool) for tool in tools]
        native_response = await self.get_native_response(
            native_prompt,
            model,
            sampling_args,
            native_tools,
            **extra_kwargs,
            **kwargs,
        )
        await self.raise_from_native_response(native_response)
        return await self.from_native_response(native_response)

    async def to_native_prompt(
        self,
        messages: Messages,
    ) -> tuple[list[dict[str, Any]], dict]:
        native_messages = []
        for message in messages:
            if hasattr(message, "model_dump"):
                native_messages.append(message.model_dump(mode="python"))
            elif isinstance(message, Mapping):
                native_messages.append(dict(message))
            else:
                native_messages.append(
                    {
                        "role": getattr(message, "role"),
                        "content": getattr(message, "content"),
                    }
                )
        return native_messages, {}

    async def to_native_tool(self, tool: Tool) -> dict[str, Any]:
        if hasattr(tool, "model_dump"):
            return tool.model_dump(mode="python")
        return dict(tool)

    def _render_prompt_ids(
        self,
        prompt: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> list[int]:
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return _coerce_prompt_ids(prompt_ids)

    def _make_sampling_params(self, sampling_args: SamplingArgs):
        from vllm import SamplingParams

        params = dict(sampling_args)
        if "n" in params and params["n"] != 1:
            raise ValueError("SameProcessTPUInferenceClient supports only n=1.")

        params["logprobs"] = params["logprobs"] if "logprobs" in params else 1
        params["prompt_logprobs"] = None
        params["detokenize"] = True
        params["skip_special_tokens"] = False
        return SamplingParams(**params)

    async def _generate(
        self,
        prompt: list[dict[str, Any]],
        sampling_args: SamplingArgs,
        tools: list[dict[str, Any]] | None,
    ):
        prompt_ids = self._render_prompt_ids(prompt, tools)
        sampling_params = self._make_sampling_params(sampling_args)
        request_id = str(self._count)
        self._count += 1
        final_output = None
        async for output in self.llm.generate(
            prompt=prompt_ids,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            final_output = output
        return final_output

    async def get_native_response(
        self,
        prompt: list[dict[str, Any]],
        model: str,
        sampling_args: SamplingArgs,
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        del model, kwargs
        return await self._generate(prompt, sampling_args, tools)

    async def raise_from_native_response(self, response) -> None:
        if not response.outputs:
            raise EmptyModelResponseError("Empty response from vLLM generate().")
        if len(response.outputs) != 1:
            raise InvalidModelResponseError(
                f"Expected exactly one completion, got {len(response.outputs)}."
            )

    async def from_native_response(self, response) -> Response:
        output = response.outputs[0]
        prompt_ids = list(response.prompt_token_ids or [])
        completion_ids = list(output.token_ids)
        completion_logprobs = _extract_logprobs(completion_ids, output.logprobs)
        finish_reason = cast(str | None, output.finish_reason)
        is_truncated = finish_reason == "length"
        tokens = ResponseTokens(
            prompt_ids=prompt_ids,
            prompt_mask=[0] * len(prompt_ids),
            completion_ids=completion_ids,
            completion_mask=[1] * len(completion_ids),
            completion_logprobs=completion_logprobs,
            routed_experts=None,
        )
        usage = Usage(
            prompt_tokens=len(prompt_ids),
            reasoning_tokens=0,
            completion_tokens=len(completion_ids),
            total_tokens=len(prompt_ids) + len(completion_ids),
        )
        message = ResponseMessage(
            role="assistant",
            content=output.text,
            reasoning_content=None,
            thinking_blocks=None,
            tool_calls=None,
            finish_reason=finish_reason,
            is_truncated=is_truncated,
            tokens=tokens,
        )
        return Response(
            id=response.request_id,
            created=int(time.time()),
            model=self.model_name,
            usage=usage,
            message=message,
        )

    def sync_weights(self, model: ToVllmMappingAbstract) -> None:
        if not isinstance(model, ToVllmMappingAbstract):
            raise TypeError(
                "sync_weights expects a model implementing ToVllmMappingAbstract."
            )

        mapping = model.to_vllm()
        self._executor.submit(
            self.llm.pause_generation(mode="keep", clear_cache=False)
        ).result()
        self._executor.submit(
            self.llm.collective_rpc(
                "sync_weights",
                args=(
                    mapping.state,
                    mapping.mappings,
                    mapping.transpose_keys,
                    None,
                ),
            )
        ).result()
        self._executor.submit(self.llm.resume_generation()).result()
        self._update_w_counter += 1
