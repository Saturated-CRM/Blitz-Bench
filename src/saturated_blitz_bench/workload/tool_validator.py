"""Validate tool call responses against expected tool metadata."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_tool_call(
    response_content: str,
    tool_calls_parsed: list[dict[str, Any]] | None,
    expected_tool: dict[str, Any] | None,
) -> bool | None:
    """Check if the model's tool call matches expected.

    Returns True if correct, False if wrong, None if not a tool-call prompt.
    """
    if expected_tool is None:
        return None

    if not tool_calls_parsed:
        logger.debug("Expected tool call but got none in response")
        return False

    expected_name = expected_tool.get("name", "")
    required_args = expected_tool.get("required_args", [])

    for tc in tool_calls_parsed:
        func = tc.get("function", {})
        name = func.get("name", "")
        if name != expected_name:
            continue

        # Check args are valid JSON and contain required keys
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except (json.JSONDecodeError, TypeError):
            return False

        if not isinstance(args, dict):
            return False

        for req in required_args:
            if req not in args:
                return False

        return True

    logger.debug(
        "Tool call name mismatch: expected=%s, got=%s",
        expected_name,
        [tc.get("function", {}).get("name") for tc in tool_calls_parsed],
    )
    return False


def extract_tool_calls_from_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reassemble tool_calls from streaming SSE delta chunks.

    OpenAI streaming sends tool calls as incremental deltas across chunks.
    We accumulate them into complete tool_call objects.
    """
    tool_calls: dict[int, dict[str, Any]] = {}

    for chunk in chunks:
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            for tc_delta in delta.get("tool_calls", []):
                idx = tc_delta.get("index", 0)
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": tc_delta.get("id", ""),
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                tc = tool_calls[idx]
                if tc_delta.get("id"):
                    tc["id"] = tc_delta["id"]
                func_delta = tc_delta.get("function", {})
                if func_delta.get("name"):
                    tc["function"]["name"] += func_delta["name"]
                if func_delta.get("arguments"):
                    tc["function"]["arguments"] += func_delta["arguments"]

    return [tool_calls[i] for i in sorted(tool_calls)]
