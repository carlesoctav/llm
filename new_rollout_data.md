# Rollout Data Iterator Plan

The current `VerifiersIterator` only keeps one rollout group in flight. Increasing
`ThreadPoolExecutor(max_workers=...)` alone does not help if the iterator submits
only one future and then blocks on it from `__next__`.

## Goal

Mimic PRIME-RL's bounded in-flight rollout scheduling inside the local data
iterator:

- keep `N` rollout groups in flight
- return one completed rollout at a time from `__next__`
- replace completed futures immediately so the in-flight window stays full

## Shape

Use a persistent executor and a pending-future set or deque.

At iterator init:

```python
self._executor = ThreadPoolExecutor(max_workers=self._max_inflight_groups)
self._futures = set()
for _ in range(self._max_inflight_groups):
    self._submit_next_group()
```

Submission should sample on the iterator thread, then run only the expensive
rollout work in the executor:

```python
def _submit_next_group(self):
    env, example = self._sample_group_inputs()
    future = self._executor.submit(self._fetch_group, env, example)
    self._futures.add(future)
```

Do not submit `_fetch_next_group` if it samples internally. Worker threads should
not touch `self._rng`.

`__next__` should drain a local rollout buffer first. If empty, wait for one
future to finish, submit a replacement, then return one rollout:

```python
def __next__(self):
    if not self._buffer:
        done, self._futures = wait(
            self._futures, return_when=FIRST_COMPLETED
        )
        future = done.pop()
        self._buffer = future.result()
        self._submit_next_group()
    return self._buffer.pop(0)
```

## Why Not `ThreadPrefetchIterDataset`

`ThreadPrefetchIterDataset` only calls `next(parent_iter)` in the background. It
cannot make `VerifiersIterator` produce more than one rollout group ahead if the
parent iterator itself only maintains one pending future.

The fix belongs inside `VerifiersIterator`, where rollout group scheduling is
known.

## Checkpoint Caveat

Multiple in-flight groups make exact iterator checkpointing harder. The current
single-future version only saves RNG state and the current output buffer. With
`N` in-flight groups, a strict restore should also save pending sampled examples.

Acceptable first version:

- save RNG state and current output buffer
- cancel pending futures on restore
- refill the in-flight window from the restored RNG state

This may replay or skip work around checkpoint boundaries, but it is usually
fine for online RL rollout generation.

## Complete Sketch: Wait-In-Iterator

This is the simplest version. There is no extra producer thread. `__next__`
drives the future window whenever the local output buffer is empty.

```python
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait


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
        self._executor = ThreadPoolExecutor(
            max_workers=max_inflight_groups,
            thread_name_prefix="rollout_iterator",
        )
        self._buffer = []
        self._futures: set[Future] = set()
        for _ in range(max_inflight_groups):
            self._submit_next_group()

    def _submit_next_group(self):
        env, example = self._sample_group_inputs()
        future = self._executor.submit(self._fetch_group, env, example)
        self._futures.add(future)

    def _fetch_group(self, env, example):
        group_inputs = [
            copy.deepcopy(example) for _ in range(self._rollouts_per_example)
        ]
        states = asyncio.run(
            env.run_group(
                group_inputs,
                self._client,
                self._client.model_name,
                self._sampling_args,
                max_retries=self._max_retries,
                state_columns=["trajectory"],
            )
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

    def get_state(self):
        return {
            "bit_generator_state": copy.deepcopy(self._rng.bit_generator.state),
            "buffer": copy.deepcopy(self._buffer),
        }

    def set_state(self, state):
        self._rng.bit_generator.state = state["bit_generator_state"]
        self._buffer = copy.deepcopy(state["buffer"])
        for future in self._futures:
            future.cancel()
        self._futures.clear()
        for _ in range(self._max_inflight_groups):
            self._submit_next_group()

    def close(self):
        for future in self._futures:
            future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().close()

    def __del__(self):
        self.close()
```

## Complete Sketch: Producer Thread + Queue

This version has a dedicated producer thread that owns sampling and future
replacement. `__next__` only blocks on a queue of completed rollout items.

```python
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from queue import Queue
import threading


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
        output_buffer_size,
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
        self._executor = ThreadPoolExecutor(
            max_workers=max_inflight_groups,
            thread_name_prefix="rollout_iterator",
        )
        self._futures: set[Future] = set()
        self._output_queue = Queue(maxsize=output_buffer_size)
        self._stop = threading.Event()
        self._producer = threading.Thread(
            target=self._producer_loop,
            daemon=True,
            name="rollout_iterator_producer",
        )
        self._producer.start()

    def _submit_next_group(self):
        env, example = self._sample_group_inputs()
        future = self._executor.submit(self._fetch_group, env, example)
        self._futures.add(future)

    def _producer_loop(self):
        for _ in range(self._max_inflight_groups):
            self._submit_next_group()

        while not self._stop.is_set():
            done, pending = wait(
                self._futures,
                timeout=0.1,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                continue

            self._futures = pending
            for future in done:
                outputs = future.result()
                self._submit_next_group()
                for output in outputs:
                    self._output_queue.put(output)

    def _fetch_group(self, env, example):
        group_inputs = [
            copy.deepcopy(example) for _ in range(self._rollouts_per_example)
        ]
        states = asyncio.run(
            env.run_group(
                group_inputs,
                self._client,
                self._client.model_name,
                self._sampling_args,
                max_retries=self._max_retries,
                state_columns=["trajectory"],
            )
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
        return self._output_queue.get()

    def get_state(self):
        return {
            "bit_generator_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    def set_state(self, state):
        self._rng.bit_generator.state = state["bit_generator_state"]

    def close(self):
        self._stop.set()
        for future in self._futures:
            future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().close()

    def __del__(self):
        self.close()
```

Use the first version unless `__next__` itself must stay extremely cheap. It is
easier to reason about and easier to checkpoint.
