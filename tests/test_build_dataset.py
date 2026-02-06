"""Tests for the dataset build pipeline (unit tests, no HuggingFace downloads)."""

import json
import uuid

import tiktoken

# Test the helper functions directly
from saturated_blitz_bench.dataset_builder import (
    _count_messages_tokens,
    _count_tokens,
    _get_encoding,
    _make_entry,
    _sample_balanced,
    CATEGORIES,
)


def test_token_counting():
    enc = _get_encoding("cl100k_base")
    count = _count_tokens("Hello, world!", enc)
    assert count > 0
    assert isinstance(count, int)


def test_messages_token_counting():
    enc = _get_encoding("cl100k_base")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    count = _count_messages_tokens(messages, enc)
    assert count > 0
    # Should include overhead (4 per message + 2 reply priming)
    assert count > 10


def test_make_entry():
    entry = _make_entry(
        category="short_chat",
        source="test",
        messages=[{"role": "user", "content": "Hello"}],
        token_count=5,
        max_tokens=1024,
    )
    assert entry["category"] == "short_chat"
    assert entry["source"] == "test"
    assert entry["input_token_count"] == 5
    assert entry["max_tokens"] == 1024
    assert len(entry["id"]) == 12
    assert len(entry["messages"]) == 1


def test_make_entry_with_tools():
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    expected_tool = {"name": "get_weather", "required_args": ["location"]}
    entry = _make_entry(
        category="tool_call",
        source="test",
        messages=[{"role": "user", "content": "Weather?"}],
        token_count=100,
        max_tokens=4096,
        tools=tools,
        expected_tool=expected_tool,
    )
    assert entry["tools"] == tools
    assert entry["expected_tool"] == expected_tool


def test_sample_balanced():
    """Test that balanced sampling covers token buckets."""
    # Create prompts across the short_chat range
    prompts = []
    for tok in [60, 80, 120, 180, 250, 300, 400, 450]:
        prompts.append(
            _make_entry(
                "short_chat", "test",
                [{"role": "user", "content": "x"}],
                token_count=tok, max_tokens=1024,
            )
        )

    sampled = _sample_balanced(prompts, "short_chat", 4)
    assert len(sampled) == 4

    # Should have spread across buckets
    token_counts = {p["input_token_count"] for p in sampled}
    assert len(token_counts) >= 2  # at least 2 different token ranges


def test_categories_weights_sum():
    total = sum(spec["weight"] for spec in CATEGORIES.values())
    assert abs(total - 1.0) < 0.001


def test_all_categories_have_buckets():
    for cat, spec in CATEGORIES.items():
        assert len(spec["buckets"]) >= 2, f"{cat} needs at least 2 buckets"
        for lo, hi in spec["buckets"]:
            assert lo < hi, f"{cat} bucket ({lo}, {hi}) is invalid"
