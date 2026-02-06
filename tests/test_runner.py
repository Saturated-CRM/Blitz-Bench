"""Tests for the benchmark runner (unit-level, not integration)."""

import asyncio
import json

import httpx
import pytest
import respx

from saturated_blitz_bench.config import BenchmarkConfig
from saturated_blitz_bench.core.runner import BenchmarkRunner, _extract_generated_text
from saturated_blitz_bench.workload.loader import Prompt, PromptPool


def _make_test_pool() -> PromptPool:
    """Create a minimal prompt pool for testing."""
    prompts = [
        Prompt(
            {
                "id": "test_1",
                "category": "short_chat",
                "source": "test",
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "max_tokens": 64,
                "input_token_count": 10,
            }
        )
    ]
    return PromptPool(prompts)


def _sse_body(content_parts: list[str], usage: dict | None = None) -> bytes:
    """Build SSE response body."""
    lines = []
    for part in content_parts:
        chunk = {
            "id": "chatcmpl-test",
            "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
        }
        lines.append(f"data: {json.dumps(chunk)}\n\n")

    # Final chunk
    final: dict = {
        "id": "chatcmpl-test",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    if usage:
        final["usage"] = usage
    lines.append(f"data: {json.dumps(final)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def test_extract_generated_text():
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
        {"choices": [{"delta": {}}]},
    ]
    assert _extract_generated_text(chunks) == "Hello world"


@pytest.mark.asyncio
async def test_runner_single_request():
    """Test runner executes a single request correctly with mocked endpoint."""
    config = BenchmarkConfig()
    config.endpoint.base_url = "https://mock-api.test/v1"
    config.endpoint.model = "test-model"
    config.test.max_concurrency = 1
    config.test.duration_seconds = 3
    config.test.warmup_seconds = 0
    config.test.request_timeout = 10

    pool = _make_test_pool()

    body = _sse_body(
        ["Hi", " there", "!"],
        usage={"prompt_tokens": 10, "completion_tokens": 3},
    )

    with respx.mock:
        respx.post("https://mock-api.test/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=body,
                headers={"content-type": "text/event-stream"},
            )
        )

        runner = BenchmarkRunner(config, pool)
        records = await runner.run()

    assert len(records) > 0
    success = [r for r in records if r.status == "success"]
    assert len(success) > 0
    assert success[0].output_tokens == 3
    assert success[0].ttft is not None
    assert success[0].ttft > 0


@pytest.mark.asyncio
async def test_runner_handles_http_error():
    """Runner records error status when endpoint returns HTTP 500."""
    config = BenchmarkConfig()
    config.endpoint.base_url = "https://mock-api.test/v1"
    config.endpoint.model = "test-model"
    config.test.max_concurrency = 1
    config.test.duration_seconds = 2
    config.test.warmup_seconds = 0
    config.test.request_timeout = 5

    pool = _make_test_pool()

    with respx.mock:
        respx.post("https://mock-api.test/v1/chat/completions").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        runner = BenchmarkRunner(config, pool)
        records = await runner.run()

    errors = [r for r in records if r.status == "error"]
    assert len(errors) > 0
    assert "500" in errors[0].error


@pytest.mark.asyncio
async def test_runner_handles_generic_exception():
    """Runner records error when streaming client raises unexpected exception."""
    config = BenchmarkConfig()
    config.endpoint.base_url = "https://mock-api.test/v1"
    config.endpoint.model = "test-model"
    config.test.max_concurrency = 1
    config.test.duration_seconds = 2
    config.test.warmup_seconds = 0
    config.test.request_timeout = 5

    pool = _make_test_pool()

    with respx.mock:
        respx.post("https://mock-api.test/v1/chat/completions").mock(
            side_effect=RuntimeError("Connection reset")
        )

        runner = BenchmarkRunner(config, pool)
        records = await runner.run()

    errors = [r for r in records if r.status == "error"]
    assert len(errors) > 0
    assert "Connection reset" in errors[0].error


def test_extract_generated_text_empty_chunks():
    assert _extract_generated_text([]) == ""


def test_extract_generated_text_no_content():
    chunks = [
        {"choices": [{"delta": {}}]},
        {"choices": [{"delta": {"role": "assistant"}}]},
    ]
    assert _extract_generated_text(chunks) == ""
