"""Simulated streaming client for dry-run benchmarks (no network calls)."""

from __future__ import annotations

import asyncio
import random
from typing import Any, AsyncIterator

import httpx


class DryRunClient:
    """Drop-in replacement for StreamingClient that simulates LLM responses.

    Produces realistic timing behavior (TTFT, ITL, variable token counts)
    and configurable error/timeout rates for load testing without a real endpoint.

    Default parameters simulate a mid-range GPU inference server:
    - TTFT: 100-500ms (time for prefill/prompt processing)
    - Output tokens: 50-500 (realistic completion lengths)
    - ITL: 15-40ms per token (~25-65 tok/s per request)
    - Error rate: 2%, Timeout rate: 1%
    """

    def __init__(
        self,
        ttft_range: tuple[float, float] = (0.1, 0.5),
        tokens_range: tuple[int, int] = (50, 500),
        itl_range: tuple[float, float] = (0.015, 0.04),
        error_rate: float = 0.02,
        timeout_rate: float = 0.01,
    ) -> None:
        self.ttft_range = ttft_range
        self.tokens_range = tokens_range
        self.itl_range = itl_range
        self.error_rate = error_rate
        self.timeout_rate = timeout_rate

    async def stream_chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Simulate a streaming chat completion with realistic timing."""
        # Decide if this request should fail
        roll = random.random()
        if roll < self.timeout_rate:
            # Simulate a timeout after some delay
            await asyncio.sleep(random.uniform(1.0, 5.0))
            raise asyncio.TimeoutError("Simulated request timeout")

        if roll < self.timeout_rate + self.error_rate:
            # Simulate an HTTP error after brief delay
            await asyncio.sleep(random.uniform(0.01, 0.1))
            resp = httpx.Response(
                500,
                request=httpx.Request("POST", "http://dry-run/v1/chat/completions"),
                text="Simulated server error",
            )
            raise httpx.HTTPStatusError(
                "Simulated 500",
                request=resp.request,
                response=resp,
            )

        # Simulate TTFT (prefill latency)
        ttft = random.uniform(*self.ttft_range)
        await asyncio.sleep(ttft)

        # Decide how many tokens to generate (capped by max_tokens)
        num_tokens = random.randint(*self.tokens_range)
        num_tokens = min(num_tokens, max_tokens)

        # Yield token chunks with ITL delays
        for i in range(num_tokens):
            itl = random.uniform(*self.itl_range)
            await asyncio.sleep(itl)

            chunk: dict[str, Any] = {
                "id": "chatcmpl-dryrun",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"t{i} "},
                        "finish_reason": None,
                    }
                ],
            }
            yield chunk

        # Final chunk with finish_reason and usage.
        # Only include completion_tokens — do NOT include prompt_tokens here
        # so the runner keeps the accurate input_token_count from the prompt pool.
        final: dict[str, Any] = {
            "id": "chatcmpl-dryrun",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "completion_tokens": num_tokens,
            },
        }
        yield final

    async def close(self) -> None:
        """No-op — no real connections to close."""
