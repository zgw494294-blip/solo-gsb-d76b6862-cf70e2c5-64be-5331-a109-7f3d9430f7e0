"""Shared bounded process executor for the audio endpoints.

All CPU-heavy audio work runs in a ``ProcessPoolExecutor`` so the event
loop (and ``/health``) stays responsive while computations are in flight.
Admission is bounded: at most ``AUDIO_WORKERS`` jobs run concurrently and
at most ``AUDIO_QUEUE`` further jobs wait FIFO for a slot. Anything beyond
that is rejected immediately with :class:`QueueFullError` (mapped to
``503`` + ``Retry-After: 1`` by the HTTP layer) instead of piling up
inside the process pool.

Slot lifecycle (all bookkeeping runs on the event loop, so no locks):

* a request that finds ``running < workers`` takes a slot immediately;
* otherwise it joins a FIFO queue while ``len(queue) < AUDIO_QUEUE``;
* when a running job finishes — success, business exception, or a client
  that disconnected mid-computation — its slot is handed to the oldest
  live waiter (never to a new arrival) exactly once, via the future's
  done callback;
* a queued waiter whose client disconnects is removed from the queue,
  freeing the queue slot; if the disconnect races with the slot hand-off,
  the slot is passed on to the next waiter instead of leaking;
* a running job whose client disconnects keeps its slot until the worker
  process finishes; only the result is discarded.
"""
from __future__ import annotations

import asyncio
import multiprocessing
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable

WORKERS_DEFAULT, WORKERS_MIN, WORKERS_MAX = 2, 1, 8
QUEUE_DEFAULT, QUEUE_MIN, QUEUE_MAX = 4, 0, 32


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def worker_count_from_env() -> int:
    return _env_int("AUDIO_WORKERS", WORKERS_DEFAULT, WORKERS_MIN, WORKERS_MAX)


def queue_size_from_env() -> int:
    return _env_int("AUDIO_QUEUE", QUEUE_DEFAULT, QUEUE_MIN, QUEUE_MAX)


class QueueFullError(Exception):
    """Executor is saturated: every worker is busy and the queue is full."""


class ClientDisconnectedError(Exception):
    """The HTTP client went away before the result was ready."""


# --- test hook -------------------------------------------------------------
# Optional gate, installed by tests, that pauses every job inside the worker
# process until the test releases it. The gate must be installed before the
# pool spawns its workers: forked workers inherit the shared semaphores.
# Always None in production.
_TEST_GATE: "ComputeGate | None" = None


def install_test_gate(gate: "ComputeGate") -> None:
    global _TEST_GATE
    _TEST_GATE = gate


def uninstall_test_gate() -> None:
    global _TEST_GATE
    _TEST_GATE = None


class ComputeGate:
    """Blocks each job inside the worker process until released (test hook).

    The shared semaphores are created before the pool forks its workers, so
    the child processes and the test driver share the same primitives.
    """

    def __init__(self) -> None:
        self._entered = multiprocessing.Semaphore(0)
        self._release = multiprocessing.Semaphore(0)
        self._enabled = multiprocessing.Value("i", 1)

    def _child_wait(self) -> None:
        """Runs inside the worker process, before the real computation."""
        if not self._enabled.value:
            return
        self._entered.release()
        while self._enabled.value:
            if self._release.acquire(timeout=0.05):
                return

    def wait_entered(self, timeout: float = 10.0) -> None:
        """Block the caller until one job has reached the gate."""
        if not self._entered.acquire(timeout=timeout):
            raise TimeoutError("no job reached the compute gate")

    def release_one(self) -> None:
        """Let one parked job proceed with its computation."""
        self._release.release()

    def disable(self) -> None:
        """Let all current and future jobs pass without blocking."""
        self._enabled.value = 0


def _execute_job(fn: Callable[..., Any], args: tuple) -> Any:
    """Pool entry point: applies the test gate, then runs the real job."""
    gate = _TEST_GATE
    if gate is not None:
        gate._child_wait()
    return fn(*args)


def _mp_context() -> multiprocessing.context.BaseContext:
    # fork so the test gate (and the already-imported app modules) are
    # inherited by the workers; fall back to the platform default elsewhere.
    try:
        return multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - non-Unix platforms
        return multiprocessing.get_context()


class AudioExecutor:
    """Bounded FIFO scheduler in front of a process pool.

    The pool itself never holds more than ``max_workers`` submitted jobs,
    so nothing accumulates inside it: excess load is either queued here
    (bounded) or rejected.
    """

    def __init__(self, max_workers: int, max_queue: int) -> None:
        self._max_workers = max_workers
        self._max_queue = max_queue
        self._running = 0
        self._pending: deque[asyncio.Future] = deque()
        self._pool: ProcessPoolExecutor | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def start(self) -> None:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self._max_workers, mp_context=_mp_context()
            )

    async def run(
        self,
        fn: Callable[..., Any],
        *args: Any,
        cancel_event: asyncio.Event | None = None,
    ) -> Any:
        """Run ``fn(*args)`` in a worker process, bounded and FIFO.

        Raises ``QueueFullError`` when saturated, ``ClientDisconnectedError``
        when ``cancel_event`` fires before the result is ready, and whatever
        ``fn`` raised otherwise.
        """
        if self._closed or self._pool is None:
            raise RuntimeError("audio executor is not running")
        if cancel_event is not None and cancel_event.is_set():
            raise ClientDisconnectedError
        if self._running < self._max_workers:
            self._running += 1
        elif len(self._pending) < self._max_queue:
            await self._wait_for_slot(cancel_event)
        else:
            raise QueueFullError
        return await self._run_in_pool(fn, args, cancel_event)

    async def _wait_for_slot(
        self, cancel_event: asyncio.Event | None
    ) -> None:
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        self._pending.append(waiter)
        disconnect = (
            asyncio.ensure_future(cancel_event.wait())
            if cancel_event is not None
            else None
        )
        try:
            if disconnect is not None:
                await asyncio.wait(
                    {waiter, disconnect}, return_when=asyncio.FIRST_COMPLETED
                )
            else:
                await waiter
        except BaseException:
            # Cancelled from the outside (e.g. server shutdown).
            self._abort_wait(waiter)
            raise
        finally:
            if disconnect is not None:
                disconnect.cancel()
        if (
            waiter.done()
            and not waiter.cancelled()
            and not (disconnect is not None and disconnect.done())
        ):
            return  # slot granted and the client is still connected
        self._abort_wait(waiter)
        raise ClientDisconnectedError

    def _abort_wait(self, waiter: asyncio.Future) -> None:
        if waiter.done() and not waiter.cancelled():
            # The slot was already handed to this waiter: pass it on so it
            # is released exactly once.
            self._release()
        else:
            waiter.cancel()
            try:
                self._pending.remove(waiter)
            except ValueError:
                pass

    def _release(self) -> None:
        """Free one running slot, handing it to the oldest live waiter."""
        while self._pending:
            waiter = self._pending.popleft()
            if not waiter.done():
                waiter.set_result(None)
                return
        self._running -= 1

    async def _run_in_pool(
        self,
        fn: Callable[..., Any],
        args: tuple,
        cancel_event: asyncio.Event | None,
    ) -> Any:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._pool, _execute_job, fn, args)
        # The slot is released exactly once, when the computation actually
        # finishes — even if the client has already gone away.
        future.add_done_callback(lambda _done: self._release())
        if cancel_event is None:
            return await future
        disconnect = asyncio.ensure_future(cancel_event.wait())
        try:
            await asyncio.wait(
                {future, disconnect}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            disconnect.cancel()
        if future.done():
            return future.result()
        # Disconnected mid-computation: the slot stays held until the worker
        # finishes (done callback above); only the result is discarded.
        raise ClientDisconnectedError

    async def shutdown(self) -> None:
        """Stop accepting work and reclaim the process pool."""
        self._closed = True
        while self._pending:
            self._pending.popleft().cancel()
        pool, self._pool = self._pool, None
        if pool is not None:
            # Never more than max_workers jobs are inside the pool, so there
            # is nothing queued to cancel; running jobs finish on their own.
            await asyncio.to_thread(pool.shutdown, False, cancel_futures=True)
