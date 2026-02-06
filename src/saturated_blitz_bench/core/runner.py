"""Main benchmark orchestrator — the saturated concurrency loop."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from saturated_blitz_bench.config import BenchmarkConfig
from saturated_blitz_bench.core.scheduler import Scheduler
from saturated_blitz_bench.core.streaming_client import StreamingClient
from saturated_blitz_bench.core.token_counter import count_text_tokens
from saturated_blitz_bench.metrics.collector import RequestRecord
from saturated_blitz_bench.metrics.concurrency_tracker import ConcurrencyTracker
from saturated_blitz_bench.utils.logging import LiveProgress
from saturated_blitz_bench.workload.loader import Prompt, PromptPool
from saturated_blitz_bench.workload.tool_validator import (
    extract_tool_calls_from_chunks,
    validate_tool_call,
)

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs the saturated-concurrency benchmark loop."""

    def __init__(self, config: BenchmarkConfig, pool: PromptPool) -> None:
        self.config = config
        self.pool = pool
        self.scheduler = Scheduler(
            pool,
            weights={
                k: v
                for k, v in config.workload.distribution.model_dump().items()
            },
        )
        self.client = StreamingClient(
            base_url=config.endpoint.base_url,
            api_key=config.endpoint.api_key,
            timeout=config.test.request_timeout,
        )
        self.concurrency_tracker = ConcurrencyTracker()
        self.results: list[RequestRecord] = []
        self._results_lock = asyncio.Lock()
        self._total_output_tokens = 0
        self._completed = 0
        self._errors = 0
        self._start_time = 0.0

    async def run(self) -> list[RequestRecord]:
        """Execute the benchmark and return all request records."""
        cfg = self.config.test
        semaphore = asyncio.Semaphore(cfg.max_concurrency)
        self._start_time = time.time()
        end_time = self._start_time + cfg.duration_seconds

        self.concurrency_tracker.start()

        progress = LiveProgress(cfg.max_concurrency, cfg.duration_seconds)
        progress.start()

        tasks: list[asyncio.Task] = []

        # Concurrency sampler
        sampler_task = asyncio.create_task(
            self._sample_concurrency(cfg.duration_seconds)
        )

        # Progress updater
        progress_task = asyncio.create_task(
            self._update_progress(progress, cfg.duration_seconds)
        )

        try:
            while time.time() < end_time:
                await semaphore.acquire()
                if time.time() >= end_time:
                    semaphore.release()
                    break
                prompt = self.scheduler.select()
                task = asyncio.create_task(
                    self._execute_request(prompt, semaphore)
                )
                tasks.append(task)

            # Drain: wait for in-flight requests with generous timeout
            if tasks:
                drain_timeout = cfg.request_timeout * 2
                logger.info(
                    "Draining %d in-flight requests (timeout=%ds)...",
                    sum(1 for t in tasks if not t.done()),
                    drain_timeout,
                )
                await asyncio.wait(tasks, timeout=drain_timeout)
        finally:
            sampler_task.cancel()
            progress_task.cancel()
            progress.stop()
            await self.client.close()

        logger.info(
            "Benchmark complete: %d requests, %d errors",
            self._completed,
            self._errors,
        )
        return self.results

    async def _execute_request(
        self,
        prompt: Prompt,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Execute a single streaming request and record metrics."""
        record = RequestRecord(
            prompt_id=prompt.id,
            category=prompt.category,
            input_tokens=prompt.input_token_count,
        )
        record.request_sent_at = time.time()
        self.concurrency_tracker.increment()

        try:
            chunks: list[dict] = []
            async for chunk in self.client.stream_chat_completion(
                messages=prompt.messages,
                model=self.config.endpoint.model,
                max_tokens=prompt.max_tokens,
                temperature=self.config.test.temperature,
                tools=prompt.tools,
            ):
                now = time.time()

                # Check for content tokens
                has_content = False
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content") or delta.get("tool_calls"):
                        has_content = True
                        break

                if has_content:
                    if record.first_token_at is None:
                        record.first_token_at = now
                    record.token_timestamps.append(now)

                chunks.append(chunk)

                # Extract usage from final chunk if provided
                usage = chunk.get("usage")
                if usage:
                    if usage.get("prompt_tokens"):
                        record.input_tokens = usage["prompt_tokens"]
                    if usage.get("completion_tokens"):
                        record.output_tokens = usage["completion_tokens"]

            record.completed_at = time.time()

            # Fallback token counting: count generated content with tiktoken
            if record.output_tokens == 0:
                generated_text = _extract_generated_text(chunks)
                if generated_text:
                    record.output_tokens = count_text_tokens(generated_text)

            # Tool call validation
            if prompt.category == "tool_call" and prompt.expected_tool:
                tool_calls = extract_tool_calls_from_chunks(chunks)
                record.tool_call_correct = validate_tool_call(
                    _extract_generated_text(chunks),
                    tool_calls,
                    prompt.expected_tool,
                )

            record.status = "success"

        except asyncio.TimeoutError:
            record.status = "timeout"
            record.completed_at = time.time()
            record.error = "Request timed out"
        except httpx.HTTPStatusError as e:
            record.status = "error"
            record.completed_at = time.time()
            record.error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            record.status = "error"
            record.completed_at = time.time()
            record.error = str(e)[:300]
        finally:
            record.finalize()
            self.concurrency_tracker.decrement()

            async with self._results_lock:
                self.results.append(record)
                if record.status == "success":
                    self._completed += 1
                    self._total_output_tokens += record.output_tokens
                else:
                    self._errors += 1

            semaphore.release()

    async def _sample_concurrency(self, duration: float) -> None:
        """Sample concurrency every second for the tracker."""
        try:
            end = time.time() + duration + 30  # extra buffer for drain
            while time.time() < end:
                self.concurrency_tracker.sample()
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass

    async def _update_progress(self, progress: LiveProgress, duration: float) -> None:
        """Update the live progress display every 0.5s."""
        try:
            end = time.time() + duration + 30
            while time.time() < end:
                progress.update(
                    active=self.concurrency_tracker.current,
                    completed=self._completed,
                    errors=self._errors,
                    total_output_tokens=self._total_output_tokens,
                )
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass


def _extract_generated_text(chunks: list[dict]) -> str:
    """Concatenate content deltas from SSE chunks."""
    parts: list[str] = []
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            content = choice.get("delta", {}).get("content")
            if content:
                parts.append(content)
    return "".join(parts)
