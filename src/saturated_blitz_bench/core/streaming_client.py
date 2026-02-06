"""Async OpenAI-compatible streaming HTTP client using httpx."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)


class StreamingClient:
    """Async client for OpenAI-compatible /v1/chat/completions streaming."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout: float = 180.0,
    ) -> None:
        self.completions_url = f"{base_url.rstrip('/')}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=30.0),
            http2=True,
            limits=httpx.Limits(
                max_connections=500,
                max_keepalive_connections=200,
            ),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def stream_chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a chat completion request, yielding parsed SSE data chunks.

        Each yielded dict is the parsed JSON from a `data: {...}` SSE line.
        The final chunk typically has choices[0].finish_reason set.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools

        async with self._client.stream(
            "POST",
            self.completions_url,
            json=payload,
        ) as response:
            response.raise_for_status()
            buffer = ""
            async for raw_bytes in response.aiter_bytes():
                buffer += raw_bytes.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line == "data: [DONE]":
                        return
                    if line.startswith("data: "):
                        json_str = line[6:]
                        try:
                            chunk = json.loads(json_str)
                            yield chunk
                        except json.JSONDecodeError:
                            logger.debug("Failed to parse SSE chunk: %s", json_str)
