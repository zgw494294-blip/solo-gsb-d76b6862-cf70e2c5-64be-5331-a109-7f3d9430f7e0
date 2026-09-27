"""Concurrency tests for the shared bounded process executor.

The ComputeGate test hook parks every job inside the worker process until
the test releases it, giving deterministic control over what is running,
what is queued and what gets rejected.
"""
import asyncio
import base64
import io
import threading
import time
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.executor import (
    AudioExecutor,
    ClientDisconnectedError,
    ComputeGate,
    QueueFullError,
    install_test_gate,
    queue_size_from_env,
    uninstall_test_gate,
    worker_count_from_env,
)
from app.loudness import AudioValidationError
from app.main import app

RATE = 48000


# --- jobs executed inside the worker processes (must be module-level) ------


def _job_echo(value):
    return value


def _job_fail():
    raise AudioValidationError("simulated business failure")


# --- helpers ---------------------------------------------------------------


async def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not met in time")
        await asyncio.sleep(0.005)


def _wait_executor_state(running, pending, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        executor = getattr(app.state, "audio_executor", None)
        if executor is not None and (
            executor._running,
            len(executor._pending),
        ) == (running, pending):
            return
        time.sleep(0.01)
    raise TimeoutError(f"executor state not ({running}, {pending})")


def make_wav_base64(freq=440.0, duration=1.0, amp=0.2, channels=1) -> str:
    t = np.arange(int(RATE * duration)) / RATE
    sig = amp * np.sin(2 * np.pi * freq * t)
    if channels == 2:
        sig = np.stack([sig, sig], axis=1)
    pcm = np.clip(np.round(sig * 32768), -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def compress_payload():
    return {
        "wav_base64": make_wav_base64(),
        "threshold_dbfs": -20.0,
        "ratio": 4.0,
        "attack_ms": 5.0,
        "release_ms": 100.0,
        "knee_db": 6.0,
    }


# --- executor-level tests ----------------------------------------------------


def test_env_config_defaults_and_clamping(monkeypatch):
    monkeypatch.delenv("AUDIO_WORKERS", raising=False)
    monkeypatch.delenv("AUDIO_QUEUE", raising=False)
    assert worker_count_from_env() == 2
    assert queue_size_from_env() == 4

    monkeypatch.setenv("AUDIO_WORKERS", "5")
    monkeypatch.setenv("AUDIO_QUEUE", "0")
    assert worker_count_from_env() == 5
    assert queue_size_from_env() == 0

    monkeypatch.setenv("AUDIO_WORKERS", "0")
    monkeypatch.setenv("AUDIO_QUEUE", "33")
    assert worker_count_from_env() == 1  # clamped to [1, 8]
    assert queue_size_from_env() == 32  # clamped to [0, 32]

    monkeypatch.setenv("AUDIO_WORKERS", "99")
    monkeypatch.setenv("AUDIO_QUEUE", "-3")
    assert worker_count_from_env() == 8
    assert queue_size_from_env() == 0

    monkeypatch.setenv("AUDIO_WORKERS", "not-a-number")
    monkeypatch.setenv("AUDIO_QUEUE", "")
    assert worker_count_from_env() == 2  # unparseable -> default
    assert queue_size_from_env() == 4


def test_workers_run_in_parallel():
    async def main():
        gate = ComputeGate()
        install_test_gate(gate)
        executor = AudioExecutor(max_workers=2, max_queue=0)
        executor.start()
        try:
            first = asyncio.create_task(executor.run(_job_echo, "a"))
            second = asyncio.create_task(executor.run(_job_echo, "b"))
            # Both jobs are parked inside the workers at the same time.
            await asyncio.to_thread(gate.wait_entered)
            await asyncio.to_thread(gate.wait_entered)
            gate.release_one()
            gate.release_one()
            assert await asyncio.wait_for(first, 5) == "a"
            assert await asyncio.wait_for(second, 5) == "b"
            assert executor._running == 0
        finally:
            gate.disable()
            uninstall_test_gate()
            await executor.shutdown()

    asyncio.run(main())


def test_full_queue_rejects_immediately_and_runs_fifo():
    async def main():
        gate = ComputeGate()
        install_test_gate(gate)
        executor = AudioExecutor(max_workers=1, max_queue=3)
        executor.start()
        try:
            first = asyncio.create_task(executor.run(_job_echo, "a"))
            await asyncio.to_thread(gate.wait_entered)  # a: running, parked
            queued = [
                asyncio.create_task(executor.run(_job_echo, tag))
                for tag in ("b", "c", "d")
            ]
            await _wait_until(lambda: len(executor._pending) == 3)

            # Queue full: rejected at once, nothing piles up.
            with pytest.raises(QueueFullError):
                await executor.run(_job_echo, "e")
            assert len(executor._pending) == 3

            # Drain one slot at a time: results must come back in FIFO order.
            gate.release_one()
            assert await asyncio.wait_for(first, 5) == "a"
            for task, tag in zip(queued, ("b", "c", "d")):
                await asyncio.to_thread(gate.wait_entered)
                gate.release_one()
                assert await asyncio.wait_for(task, 5) == tag
            assert executor._running == 0
            assert not executor._pending
        finally:
            gate.disable()
            uninstall_test_gate()
            await executor.shutdown()

    asyncio.run(main())


def test_zero_queue_accepts_no_waiters():
    async def main():
        gate = ComputeGate()
        install_test_gate(gate)
        executor = AudioExecutor(max_workers=1, max_queue=0)
        executor.start()
        try:
            first = asyncio.create_task(executor.run(_job_echo, "a"))
            await asyncio.to_thread(gate.wait_entered)
            with pytest.raises(QueueFullError):
                await executor.run(_job_echo, "b")
            gate.release_one()
            assert await asyncio.wait_for(first, 5) == "a"
            assert executor._running == 0
        finally:
            gate.disable()
            uninstall_test_gate()
            await executor.shutdown()

    asyncio.run(main())


def test_disconnect_while_queued_and_while_running():
    """Two-phase disconnect: a queued waiter frees its queue slot on
    disconnect; a running job keeps its worker slot until the computation
    finishes, and only its result is discarded."""
    async def main():
        gate = ComputeGate()
        install_test_gate(gate)
        executor = AudioExecutor(max_workers=1, max_queue=1)
        executor.start()
        try:
            cancel_a, cancel_b = asyncio.Event(), asyncio.Event()
            task_a = asyncio.create_task(
                executor.run(_job_echo, "a", cancel_event=cancel_a)
            )
            await asyncio.to_thread(gate.wait_entered)  # a: running, parked

            # Phase 1: b waits in the queue, then its client disconnects.
            task_b = asyncio.create_task(
                executor.run(_job_echo, "b", cancel_event=cancel_b)
            )
            await _wait_until(lambda: len(executor._pending) == 1)
            cancel_b.set()
            with pytest.raises(ClientDisconnectedError):
                await task_b
            await _wait_until(lambda: len(executor._pending) == 0)

            # The freed queue slot is available again: c may wait.
            task_c = asyncio.create_task(executor.run(_job_echo, "c"))
            await _wait_until(lambda: len(executor._pending) == 1)

            # Phase 2: a's client disconnects mid-computation.
            cancel_a.set()
            with pytest.raises(ClientDisconnectedError):
                await task_a
            # The worker slot is still held: c stays queued, d is rejected.
            assert executor._running == 1
            assert len(executor._pending) == 1
            with pytest.raises(QueueFullError):
                await executor.run(_job_echo, "d")

            # Only when a's computation finishes is the slot released; the
            # queued c then runs (FIFO) and completes normally.
            gate.release_one()
            await asyncio.to_thread(gate.wait_entered)  # c: now running
            gate.release_one()
            assert await asyncio.wait_for(task_c, 5) == "c"
            assert executor._running == 0
            assert not executor._pending
        finally:
            gate.disable()
            uninstall_test_gate()
            await executor.shutdown()

    asyncio.run(main())


def test_business_exception_releases_capacity():
    async def main():
        gate = ComputeGate()
        install_test_gate(gate)
        executor = AudioExecutor(max_workers=1, max_queue=0)
        executor.start()
        try:
            failing = asyncio.create_task(executor.run(_job_fail))
            await asyncio.to_thread(gate.wait_entered)
            gate.release_one()
            with pytest.raises(AudioValidationError):
                await failing
            # The slot was released exactly once: the executor is idle and
            # the next job runs to completion.
            assert executor._running == 0
            ok = asyncio.create_task(executor.run(_job_echo, "recovered"))
            await asyncio.to_thread(gate.wait_entered)
            gate.release_one()
            assert await asyncio.wait_for(ok, 5) == "recovered"
            assert executor._running == 0
        finally:
            gate.disable()
            uninstall_test_gate()
            await executor.shutdown()

    asyncio.run(main())


def test_cancel_race_releases_slot_exactly_once():
    """A queued waiter that disconnects while a running job finishes must
    neither leak the handed-off slot nor release it twice."""
    async def main():
        gate = ComputeGate()
        install_test_gate(gate)
        executor = AudioExecutor(max_workers=1, max_queue=1)
        executor.start()
        try:
            for _ in range(20):
                cancel_a, cancel_b = asyncio.Event(), asyncio.Event()
                task_a = asyncio.create_task(
                    executor.run(_job_echo, "a", cancel_event=cancel_a)
                )
                await asyncio.to_thread(gate.wait_entered)
                task_b = asyncio.create_task(
                    executor.run(_job_echo, "b", cancel_event=cancel_b)
                )
                await _wait_until(lambda: len(executor._pending) == 1)
                # Disconnect b at the same moment a's computation finishes.
                cancel_b.set()
                gate.release_one()
                with pytest.raises(ClientDisconnectedError):
                    await task_b
                assert await asyncio.wait_for(task_a, 5) == "a"
                # Whatever the interleaving, the executor drains back to
                # idle (no leaked or double-released slot) ...
                await _wait_until(
                    lambda: executor._running == 0 and not executor._pending
                )
                # ... and accepts new work.
                again = asyncio.create_task(executor.run(_job_echo, "again"))
                await asyncio.to_thread(gate.wait_entered)
                gate.release_one()
                assert await asyncio.wait_for(again, 5) == "again"
                assert executor._running == 0
        finally:
            gate.disable()
            uninstall_test_gate()
            await executor.shutdown()

    asyncio.run(main())


def test_slot_handoff_race_whitebox():
    """Deterministically drive the hand-off/disconnect race branch."""
    async def main():
        executor = AudioExecutor(max_workers=1, max_queue=1)
        loop = asyncio.get_running_loop()
        executor._running = 1  # one job "running"
        waiter = loop.create_future()
        executor._pending.append(waiter)
        executor._release()  # slot handed to the waiter, still counted
        assert waiter.done()
        assert executor._running == 1
        # The waiter's client disconnected right as the slot arrived: the
        # slot must be passed on (here: back to idle) exactly once.
        executor._abort_wait(waiter)
        assert executor._running == 0
        assert not executor._pending

    asyncio.run(main())


# --- HTTP-level tests --------------------------------------------------------


def test_http_saturated_returns_503_and_health_still_ok(monkeypatch):
    gate = ComputeGate()
    install_test_gate(gate)
    monkeypatch.setenv("AUDIO_WORKERS", "1")
    monkeypatch.setenv("AUDIO_QUEUE", "0")
    try:
        with TestClient(app) as client:
            first_result = {}

            def blocked_post():
                first_result["response"] = client.post(
                    "/v1/compress", json=compress_payload()
                )

            thread = threading.Thread(target=blocked_post)
            thread.start()
            gate.wait_entered(timeout=15)  # the only worker is now busy

            # Saturated: immediate 503 with Retry-After, no pile-up.
            resp = client.post("/v1/compress", json=compress_payload())
            assert resp.status_code == 503
            assert resp.headers["Retry-After"] == "1"

            # /health does not go through the queue.
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}

            gate.release_one()
            thread.join(timeout=30)
            assert not thread.is_alive()
            response = first_result["response"]
            assert response.status_code == 200
            assert "wav_base64" in response.json()
        assert app.state.audio_executor is None  # reclaimed on shutdown
    finally:
        gate.disable()
        uninstall_test_gate()


def test_http_queued_requests_complete_in_order(monkeypatch):
    gate = ComputeGate()
    install_test_gate(gate)
    monkeypatch.setenv("AUDIO_WORKERS", "1")
    monkeypatch.setenv("AUDIO_QUEUE", "2")
    try:
        with TestClient(app) as client:
            results = {}

            def post(tag):
                results[tag] = client.post("/v1/compress", json=compress_payload())

            threads = [
                threading.Thread(target=post, args=(tag,)) for tag in "abc"
            ]
            for thread in threads:
                thread.start()
            gate.wait_entered(timeout=15)  # one running
            # Both remaining requests must actually be queued before the
            # fourth one can be rejected.
            _wait_executor_state(1, 2)

            # One running + two queued = capacity exhausted.
            resp = client.post("/v1/compress", json=compress_payload())
            assert resp.status_code == 503
            assert resp.headers["Retry-After"] == "1"

            for tag in "abc":
                gate.release_one()
                if tag != "c":
                    gate.wait_entered(timeout=15)
            for thread in threads:
                thread.join(timeout=30)
                assert not thread.is_alive()
            assert {tag: r.status_code for tag, r in results.items()} == {
                "a": 200,
                "b": 200,
                "c": 200,
            }
            for response in results.values():
                assert "wav_base64" in response.json()
    finally:
        gate.disable()
        uninstall_test_gate()


def test_http_validation_still_applies_with_executor():
    with TestClient(app) as client:
        resp = client.post("/v1/compress", json=compress_payload())
        assert resp.status_code == 200

        bad = compress_payload()
        bad["wav_base64"] = "!!!not-base64!!!"
        resp = client.post("/v1/compress", json=bad)
        assert resp.status_code == 422

        bad = compress_payload()
        bad["ratio"] = 100.0
        resp = client.post("/v1/compress", json=bad)
        assert resp.status_code == 422
