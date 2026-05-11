from __future__ import annotations

import asyncio
import logging
import sws
from typing import Any


def get_config():
    c = sws.Config()
    c.model_name =


FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['content'] }}{% endfor %}"
)


def _create_server_args(model_path: str, has_chat_template: bool) -> argparse.Namespace:
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)
    cli_args = ["--model", model_path]
    if not has_chat_template:
        cli_args.extend(["--chat-template", FALLBACK_CHAT_TEMPLATE])
    args = parser.parse_args(cli_args)
    args.disable_fastapi_docs = True
    return args


async def _init_engine_and_server(self, llm_config: dict[str, Any]):
    import uvicorn
    from vllm import AsyncEngineArgs
    from vllm.entrypoints.openai.api_server import build_app, init_app_state
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.executor import Executor

    engine_args = AsyncEngineArgs(**llm_config)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)

    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("tpu_inference").setLevel(logging.ERROR)
    logging.getLogger("vllm").setLevel(logging.ERROR)

    llm = AsyncLLM(
        vllm_config,
        executor_class=executor_class,
        log_stats=True,
        use_uniproc_engine_core=True,
    )
    args = _create_server_args(
        llm.vllm_config.model_config.model,
        True,
    )
    app = build_app(args)
    await init_app_state(llm, app.state, args)

    config = uvicorn.Config(
        app,
        log_level="warning",
        access_log=False,
    )
    _server = uvicorn.Server(config)
    _server_task = asyncio.create_task(_server.serve())
    await asyncio.sleep(0.1)
    return llm


async def main(config):
    self._env_names = env_names
    self._datasets = [_get_env_dataset(env) for env in envs]
    pass


if __name__ == "__main__":
    asyncio.run(main())
