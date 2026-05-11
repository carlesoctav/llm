from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future as ConFuture


class AsyncLoopThread:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="rollout_async_loop"
        )
        self._thread.start()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro) -> ConFuture:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=1)
        self._loop.close()
