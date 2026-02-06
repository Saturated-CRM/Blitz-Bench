"""Tests for the streaming client SSE parsing."""

import asyncio
import json

import httpx
import pytest
import respx

from saturated_blitz_bench.core.streaming_client import StreamingClient


def _make_sse_response(chunks: list[dict]) -> str:
    """Build SSE response string from chunk dicts."""
    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines)


@pytest.mark.asyncio
async def test_stream_parses_sse_chunks():
    """Test that the client correctly parses SSE data lines."""
    chunks = [
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"content": " world"}, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    ]

    sse_body = _make_sse_response(chunks)

    with respx.mock:
        respx.post("https://test.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=sse_body.encode(),
                headers={"content-type": "text/event-stream"},
            )
        )

        client = StreamingClient(
            base_url="https://test.example.com/v1",
            timeout=10.0,
        )
        received = []
        async for chunk in client.stream_chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model="test-model",
        ):
            received.append(chunk)

        await client.close()

    assert len(received) == 3
    assert received[0]["choices"][0]["delta"]["content"] == "Hello"
    assert received[1]["choices"][0]["delta"]["content"] == " world"
    assert received[2]["usage"]["completion_tokens"] == 2


@pytest.mark.asyncio
async def test_stream_handles_empty_lines():
    """SSE can have empty lines between data lines."""
    sse_body = (
        "\n\n"
        'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        "\n"
        "data: [DONE]\n\n"
    )

    with respx.mock:
        respx.post("https://test.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=sse_body.encode(),
                headers={"content-type": "text/event-stream"},
            )
        )

        client = StreamingClient(base_url="https://test.example.com/v1", timeout=10.0)
        received = []
        async for chunk in client.stream_chat_completion(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        ):
            received.append(chunk)

        await client.close()

    assert len(received) == 1


@pytest.mark.asyncio
async def test_stream_handles_http_error():
    """Streaming client raises HTTPStatusError on 500 response."""
    with respx.mock:
        respx.post("https://test.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        client = StreamingClient(base_url="https://test.example.com/v1", timeout=10.0)
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in client.stream_chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="test-model",
            ):
                pass
        await client.close()


@pytest.mark.asyncio
async def test_stream_handles_leftover_buffer():
    """Data at end of stream without trailing newline is still yielded."""
    # No trailing newline after [DONE], and final data line has no \n
    sse_body = (
        'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}'
    )

    with respx.mock:
        respx.post("https://test.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=sse_body.encode(),
                headers={"content-type": "text/event-stream"},
            )
        )

        client = StreamingClient(base_url="https://test.example.com/v1", timeout=10.0)
        received = []
        async for chunk in client.stream_chat_completion(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        ):
            received.append(chunk)
        await client.close()

    # Both chunks should be received (first from normal parsing, second from leftover buffer)
    assert len(received) == 2
    assert received[1]["usage"]["completion_tokens"] == 1


@pytest.mark.asyncio
async def test_stream_handles_malformed_json():
    """Malformed JSON in SSE line is skipped, valid chunks still yielded."""
    sse_body = (
        'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        "data: {not valid json}\n\n"
        'data: {"choices":[{"delta":{"content":"!!"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    with respx.mock:
        respx.post("https://test.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=sse_body.encode(),
                headers={"content-type": "text/event-stream"},
            )
        )

        client = StreamingClient(base_url="https://test.example.com/v1", timeout=10.0)
        received = []
        async for chunk in client.stream_chat_completion(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        ):
            received.append(chunk)
        await client.close()

    # Only the two valid chunks should be received
    assert len(received) == 2
