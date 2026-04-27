# Async vLLM Client Plan

Goal: make concurrent Verifiers rollout calls actually overlap inside vLLM.

The current client is async only at the Python API boundary:

```python
async def get_native_response(...):
    return await asyncio.to_thread(self._generate, prompt, sampling_args, tools)
```

But `_generate()` calls sync `LLM.generate()` under `_runtime_lock`, so concurrent
rollout coroutines still serialize at generation time.

The fix is to use `vllm.v1.engine.async_llm.AsyncLLM` inside
`SameProcessTPUInferenceClient`.

## Construction

Replace:

```python
from vllm import LLM

self.llm = LLM(**llm_config)
self._client = self.llm
```

with:

```python
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

self.llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**llm_config))
self._client = self.llm
```

Keep a lock, but use it only for operations that must not race with generation,
mainly weight sync. Do not hold it around every generation request.

## Sampling Params

Current logic can mostly stay:

```python
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
```

Optionally set final-only output if we do not want streaming chunks:

```python
from vllm.sampling_params import RequestOutputKind

sampling_params.output_kind = RequestOutputKind.FINAL_ONLY
```

## Async Generation

Replace `_generate()` and `asyncio.to_thread(...)` with direct async generation:

```python
async def _generate(
    self,
    prompt: list[dict[str, Any]],
    sampling_args: SamplingArgs,
    tools: list[dict[str, Any]] | None,
):
    from vllm.inputs import TokensPrompt

    prompt_ids = self._render_prompt_ids(prompt, tools)
    sampling_params = self._make_sampling_params(sampling_args)
    request_id = str(next(self._request_counter))

    final_output = None
    async for output in self.llm.generate(
        TokensPrompt(prompt_token_ids=prompt_ids),
        sampling_params,
        request_id=request_id,
    ):
        final_output = output

    if final_output is None:
        raise EmptyModelResponseError("Empty response from vLLM generate().")
    return final_output
```

Then:

```python
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
```

Now if Verifiers runs multiple rollout tasks concurrently, each task awaits its
own `AsyncLLM.generate(...)` stream. vLLM sees multiple live requests and can
schedule/continuous-batch them.

## Request Counter

Add in `__init__`:

```python
from itertools import count

self._request_counter = count()
```

## Weight Sync

Weight sync must not race with active requests.

Simple first version:

```python
self._active_generations = 0
self._no_active_generations = asyncio.Condition()
self._sync_lock = asyncio.Lock()
```

Generation path:

```python
async with self._sync_lock:
    async with self._no_active_generations:
        self._active_generations += 1

try:
    ...
finally:
    async with self._no_active_generations:
        self._active_generations -= 1
        if self._active_generations == 0:
            self._no_active_generations.notify_all()
```

Async sync path:

```python
async def _sync_weights_async(self, mapping: VllmMapping) -> None:
    async with self._sync_lock:
        async with self._no_active_generations:
            while self._active_generations:
                await self._no_active_generations.wait()

        with _mute_stdio():
            self.llm.collective_rpc(
                "sync_weights",
                args=(
                    mapping.state,
                    mapping.mappings,
                    mapping.transpose_keys,
                    None,
                ),
            )
```

If the trainer calls sync `update_weights(...)`, bridge it into the client loop
with the same mechanism used by the caller, or make the training path call an
async update method.

For a first local experiment, it is also acceptable to keep the old sync
`update_weights()` and only call it between rollout batches, while no async
rollouts are active.

## Close

```python
async def close(self) -> None:
    self._client = None
    self.llm.shutdown()
    self.llm = None
```

## Test

Use a fake `AsyncLLM.generate()`:

1. Start 8 concurrent `get_native_response()` calls.
2. Assert fake `generate()` saw 8 distinct request IDs.
3. Assert calls overlapped before returning.
4. Assert each coroutine got its own final output.

Then test with GSM8K + `async_verifiers.py` and compare throughput against the
old `asyncio.to_thread(LLM.generate)` path.
