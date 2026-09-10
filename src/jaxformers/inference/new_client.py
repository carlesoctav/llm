from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import socket
import threading
import time
import urllib.request
from typing import Any

from transformers import PreTrainedTokenizerBase

from jaxformers.async_utils import AsyncLoopThread
from jaxformers.module_utils import ToVllmMappingAbstract


try:
    from verifiers.types import ClientConfig
except ImportError:  # pragma: no cover
    ClientConfig = Any


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_health(port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"vLLM OpenAI server did not become healthy at {url}")


class NewClient:
    def __init__(
        self,
        model: str,
        *,
        tokenizer: PreTrainedTokenizerBase | str | None = None,
        vllm_config: dict[str, Any],
        host: str = "127.0.0.1",
        port: int | None = None,
        health_timeout_s: float = 60.0,
    ) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        os.environ["MODEL_IMPL_TYPE"] = "flax_nnx"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        llm_config = dict(vllm_config)
        llm_config["model"] = model
        if "tokenizer" not in llm_config and isinstance(tokenizer, str):
            llm_config["tokenizer"] = tokenizer

        self.host = host
        self.port = port if port is not None else _find_free_port()
        self.model_name = model
        self.llm = None
        self._server = None
        self._server_task = None
        self._runtime_lock = threading.RLock()
        self._executor = AsyncLoopThread()
        self._update_w_counter = 0

        self.llm = self._executor.submit(
            self._init_engine_and_server(llm_config)
        ).result()
        self.model_name = self.llm.vllm_config.model_config.model
        self.client_config = ClientConfig(
            client_idx=0,
            client_type="openai_chat_completions_token",
            api_key_var="",
            api_base_url=f"http://127.0.0.1:{self.port}/v1",
        )
        _wait_for_health(self.port, health_timeout_s)

    async def _init_engine_and_server(self, llm_config: dict[str, Any]):
        import uvicorn
        from vllm import AsyncEngineArgs
        from vllm.entrypoints.openai.cli_args import make_arg_parser
        from vllm.entrypoints.openai.api_server import build_app, init_app_state
        from vllm.utils.argparse_utils import FlexibleArgumentParser
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.v1.executor import Executor

        args =  mean is this accesigble for the env to evalshow me the code of this rubrvrishow me make_arg_parser(FlexibleArgumentParser()).parse_args([])
        for key, value in llm_config.items():
            setattr(args, key, value)
        args.disable_fastapi_docs = True
        engine_args = AsyncEngineArgs.from_cli_args(args)
        vllm_config = engine_args.create_engine_config()
        args.structured_outputs_config = engine_args.structured_outputs_config
        executor_class = Executor.get_class(vllm_config)

        logging.getLogger("absl").setLevel(logging.ERROR)
        logging.getLogger("tpu_inference").setLevel(logging.ERROR)
        logging.getLogger("vllm").setLevel(logging.ERROR)

        self.llm = AsyncLLM(
            vllm_config,
            executor_class=executor_class,
            log_stats=True,
            use_uniproc_engine_core=True,
        )
        app = build_app(args)
        await init_app_state(self.llm, app.state, args)

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())
        await asyncio.sleep(0.1)
        return self.llm

    @contextlib.contextmanager
    def runtime_lock(self):
        with self._runtime_lock:
            yield

    def update_weights(self, model: ToVllmMappingAbstract) -> None:
        self.sync_weights(model)

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

    async def _close_server(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._server_task is not None:
            await self._server_task
        if self.llm is not None:
            self.llm.shutdown()
            self.llm = None

    async def close(self) -> None:
        await asyncio.wrap_future(self._executor.submit(self._close_server()))
        self._executor.close()
