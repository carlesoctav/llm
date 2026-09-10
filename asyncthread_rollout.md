# Async Thread Rollout Loader

Use a background asyncio event loop for rollout generation while keeping the
trainer and Grain dataset interface synchronous.

This avoids rewriting the trainer to `asyncio.run(main())`, but still lets
rollout requests progress while the main thread is doing JAX training work.

## Design

The iterator has three parts:

- sync consumer API: `__next__`
- background asyncio loop thread
- bounded set of in-flight rollout group futures

The main thread owns sampling from the dataset/RNG. The async loop only runs the
expensive coroutine work.

## Event Loop Thread

```python
import asyncio
import threading
from concurrent.futures import Future


class AsyncLoopThread:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="rollout_async_loop",
        )
        self._thread.start()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro) -> Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def close(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()
```

## Iterator Shape

```python
from concurrent.futures import FIRST_COMPLETED, Future, wait


class VerifiersIterator(grain.DatasetIterator):
    def __init__(
        self,
        envs,
        env_names,
        client,
        rollouts_per_example,
        sampling_args,
        env_weights,
        seed,
        max_retries,
        max_inflight_groups,
    ):
        super().__init__()
        self._envs = envs
        self._env_names = env_names
        self._datasets = [_get_env_dataset(env) for env in envs]
        self._client = client
        self._rollouts_per_example = rollouts_per_example
        self._sampling_args = sampling_args
        self._env_weights = env_weights
        self._rng = np.random.default_rng(seed)
        self._max_retries = max_retries
        self._max_inflight_groups = max_inflight_groups
        self._loop_thread = AsyncLoopThread()
        self._futures: set[Future] = set()
        self._buffer = []

        for _ in range(max_inflight_groups):
            self._submit_next_group()

    def _submit_next_group(self):
        env, example = self._sample_group_inputs()
        future = self._loop_thread.submit(self._fetch_group(env, example))
        self._futures.add(future)

    async def _fetch_group(self, env, example):
        group_inputs = [
            copy.deepcopy(example) for _ in range(self._rollouts_per_example)
        ]
        states = await env.run_group(
            group_inputs,
            self._client,
            self._client.model_name,
            self._sampling_args,
            max_retries=self._max_retries,
            state_columns=["trajectory"],
        )
        rewards = [float(state["reward"] or 0.0) for state in states]
        advantages = _compute_advantages(rewards)
        outputs = []
        for state, advantage in zip(states, advantages):
            outputs.append(
                {
                    "trajectory": state["trajectory"],
                    "reward": np.float32(state["reward"] or 0.0),
                    "advantage": np.float32(advantage),
                    "example_id": state["example_id"],
                    "task": state["task"],
                    "is_truncated": state["is_truncated"],
                    "stop_condition": state.get("stop_condition"),
                }
            )
        return outputs

    def __next__(self):
        if not self._buffer:
            done, pending = wait(self._futures, return_when=FIRST_COMPLETED)
            self._futures = pending
            future = done.pop()
            self._buffer = future.result()
            self._submit_next_group()
        return self._buffer.pop(0)
```

## Why This Beats Main-Thread Async

With a main-thread event loop, rollout tasks only progress while the event loop
is running. If the training step is synchronous, the loop stops during training.

With a background loop thread, rollout coroutines continue while the main thread
is doing JAX work.

This matters when `env.run_group(...)` is mostly network/model-server IO.

## Lifecycle

Cancel futures and stop the loop when the iterator closes.

```python
def close(self):
    for future in self._futures:
        future.cancel()
    self._futures.clear()
    self._loop_thread.close()
    super().close()

def __del__(self):
    self.close()
```

If close can race with `__next__`, add a `_closed` flag and raise `ValueError`
from `__next__` after close.

## Checkpointing

Strict checkpointing is harder with multiple in-flight groups.

Pragmatic first version:

- save RNG state
- save the current local output buffer
- cancel all in-flight futures on restore
- refill the in-flight window from restored RNG state

This can replay or skip rollout work around checkpoint boundaries, but online RL
rollout generation usually tolerates that.

Strict version:

- sample examples before submission
- store pending sampled examples in iterator state
- on restore, resubmit those exact pending examples

Do not let worker coroutines sample from `self._rng`, otherwise restore and
determinism become much harder.
