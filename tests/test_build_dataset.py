"""Tests for the dataset build pipeline (unit tests, no HuggingFace downloads)."""

import json
from unittest.mock import patch, MagicMock

import tiktoken

from saturated_blitz_bench.dataset_builder import (
    _count_messages_tokens,
    _count_tokens,
    _get_encoding,
    _make_entry,
    _normalize_tools,
    _truncate_text,
    _process_hermes_fc,
    _process_toolace,
    _process_scrolls,
    _process_govreport,
    _process_pg19,
    _process_longbench,
    _process_longbench_v2,
    _process_gsm8k,
    _process_code_contests,
    _process_wildchat,
    _sample_balanced,
    CATEGORIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ENC = _get_encoding("cl100k_base")

def _long_text(target_tokens: int) -> str:
    """Generate text with exactly target_tokens tokens."""
    base = "the quick brown fox jumps over the lazy dog in the park near the river "
    text = base * ((target_tokens // 10) + 50)
    tokens = ENC.encode(text, disallowed_special=())
    return ENC.decode(tokens[:target_tokens])


# ---------------------------------------------------------------------------
# Token counting & entry creation
# ---------------------------------------------------------------------------

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


def test_make_entry_no_tools_field_when_none():
    entry = _make_entry(
        category="short_chat",
        source="test",
        messages=[{"role": "user", "content": "Hi"}],
        token_count=5,
        max_tokens=1024,
    )
    assert "tools" not in entry
    assert "expected_tool" not in entry


# ---------------------------------------------------------------------------
# Tool normalization
# ---------------------------------------------------------------------------

def test_normalize_tools_already_correct():
    tools = [{"type": "function", "function": {"name": "foo", "parameters": {}}}]
    result = _normalize_tools(tools)
    assert result == tools


def test_normalize_tools_bare_function():
    tools = [{"name": "get_weather", "description": "Get weather", "parameters": {}}]
    result = _normalize_tools(tools)
    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "get_weather"


def test_normalize_tools_missing_type():
    tools = [{"function": {"name": "search", "parameters": {}}}]
    result = _normalize_tools(tools)
    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "search"


def test_normalize_tools_mixed():
    tools = [
        {"type": "function", "function": {"name": "a"}},
        {"name": "b", "description": "bare"},
        {"function": {"name": "c"}},
        "not_a_dict",  # should be skipped
        {},  # should be skipped
    ]
    result = _normalize_tools(tools)
    assert len(result) == 3
    names = [t["function"]["name"] for t in result]
    assert names == ["a", "b", "c"]


def test_normalize_tools_empty():
    assert _normalize_tools([]) == []


# ---------------------------------------------------------------------------
# Sampling & balancing
# ---------------------------------------------------------------------------

def test_sample_balanced():
    """Test that balanced sampling covers token buckets."""
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
    token_counts = {p["input_token_count"] for p in sampled}
    assert len(token_counts) >= 2


def test_sample_balanced_more_than_pool():
    """When target > pool size, return everything available."""
    prompts = [
        _make_entry("short_chat", "test", [{"role": "user", "content": "x"}],
                     token_count=100, max_tokens=1024),
    ]
    sampled = _sample_balanced(prompts, "short_chat", 10)
    assert len(sampled) == 1


# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

def test_categories_weights_sum():
    total = sum(spec["weight"] for spec in CATEGORIES.values())
    assert abs(total - 1.0) < 0.001


def test_all_categories_have_buckets():
    for cat, spec in CATEGORIES.items():
        assert len(spec["buckets"]) >= 2, f"{cat} needs at least 2 buckets"
        for lo, hi in spec["buckets"]:
            assert lo < hi, f"{cat} bucket ({lo}, {hi}) is invalid"


# ---------------------------------------------------------------------------
# _process_hermes_fc (mocked)
# ---------------------------------------------------------------------------

def _hermes_fc_parsed_row(user_msg=None, tool_name="get_weather", args=None):
    """Build a fake hermes-function-calling-v1-parsed row."""
    if user_msg is None:
        user_msg = (
            "I need to check the current weather conditions for my upcoming trip. "
            "Can you tell me what the weather is like right now in New York City? "
            "I want to know the temperature, humidity, wind speed, and whether "
            "it will rain today or tomorrow. Also include the forecast."
        )
    if args is None:
        args = {"location": "NYC"}
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with tool access.", "tool_calls": None},
            {"role": "user", "content": user_msg, "tool_calls": None},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {"name": tool_name, "arguments": json.dumps(args)}}
            ]},
        ],
        "tools": json.dumps([{
            "type": "function",
            "function": {"name": tool_name, "description": f"Call {tool_name}", "parameters": {
                "type": "object", "properties": {k: {"type": "string"} for k in args},
            }},
        }]),
    }


@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_process_hermes_fc_parsed(mock_tqdm, mock_load):
    rows = [_hermes_fc_parsed_row(), _hermes_fc_parsed_row("Find restaurants", "search_restaurants", {"cuisine": "italian"})]
    mock_load.return_value = rows

    results = _process_hermes_fc(ENC, 5, 80000)
    assert len(results) >= 1
    for r in results:
        assert r["category"] == "tool_call"
        assert r["source"] == "hermes_fc"
        assert "tools" in r
        assert isinstance(r["tools"], list)
        assert r["tools"][0]["type"] == "function"
        assert "function" in r["tools"][0]
        assert r["expected_tool"] is not None
        assert "name" in r["expected_tool"]
        assert "required_args" in r["expected_tool"]


@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_process_hermes_fc_skips_no_user(mock_tqdm, mock_load):
    """Row with only system messages (no user) should be skipped."""
    row = {
        "messages": [{"role": "system", "content": "System only", "tool_calls": None}],
        "tools": "[]",
    }
    mock_load.return_value = [row]
    results = _process_hermes_fc(ENC, 5, 80000)
    assert len(results) == 0


@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_process_hermes_fc_zero_target(mock_tqdm, mock_load):
    results = _process_hermes_fc(ENC, 0, 80000)
    assert results == []
    mock_load.assert_not_called()


# ---------------------------------------------------------------------------
# _process_toolace (mocked)
# ---------------------------------------------------------------------------

def _toolace_row(user_msg=None, tool_name="get_stock", args=None):
    """Build a fake toolace-parsed row."""
    if user_msg is None:
        user_msg = (
            "I'm interested in checking the current stock price and recent performance "
            "of Apple Inc on the market. Can you look up the latest trading data for "
            "their stock ticker? I'd like to know the current price, today's percentage "
            "change, and the total trading volume for the session."
        )
    if args is None:
        args = {"ticker": "AAPL"}
    return {
        "messages": [
            {"role": "user", "content": user_msg, "tool_calls": None},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {"name": tool_name, "arguments": json.dumps(args)}}
            ]},
        ],
        "tools": json.dumps([{
            "type": "function",
            "function": {"name": tool_name, "description": f"Call {tool_name}", "parameters": {
                "type": "object", "properties": {k: {"type": "string"} for k in args},
            }},
        }]),
    }


@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_process_toolace(mock_tqdm, mock_load):
    rows = [_toolace_row(), _toolace_row("Book flight", "book_flight", {"destination": "LA"})]
    mock_ds = MagicMock()
    mock_ds.__iter__ = lambda self: iter(rows)
    mock_ds.__len__ = lambda self: len(rows)
    mock_load.return_value = mock_ds

    results = _process_toolace(ENC, 5, 80000)
    assert len(results) >= 1
    for r in results:
        assert r["category"] == "tool_call"
        assert r["source"] == "toolace"
        assert "tools" in r
        assert r["tools"][0]["type"] == "function"


@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_process_toolace_requires_tools(mock_tqdm, mock_load):
    """Row without tools field should be skipped."""
    row = {
        "messages": [{"role": "user", "content": "Hello", "tool_calls": None}],
        "tools": "",
    }
    mock_ds = MagicMock()
    mock_ds.__iter__ = lambda self: iter([row])
    mock_ds.__len__ = lambda self: 1
    mock_load.return_value = mock_ds

    results = _process_toolace(ENC, 5, 80000)
    assert len(results) == 0


@patch("saturated_blitz_bench.dataset_builder.load_dataset", side_effect=Exception("unavailable"))
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_process_toolace_fallback(mock_tqdm, mock_load):
    """When toolace-parsed is unavailable, should fall back to glaive (also mocked to fail)."""
    results = _process_toolace(ENC, 5, 80000)
    # Glaive fallback also fails since load_dataset raises
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# _process_scrolls (mocked)
# ---------------------------------------------------------------------------

@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_process_scrolls(mock_tqdm, mock_load):
    """Scrolls should produce long_context entries from English text."""
    long_text = _long_text(30000)  # ~30K tokens
    rows = [{"input": long_text}]
    mock_load.return_value = rows

    results = _process_scrolls(ENC, 5, 80000)
    assert len(results) >= 1
    for r in results:
        assert r["category"] == "long_context"
        assert r["source"] == "scrolls"
        assert r["input_token_count"] >= 20000


@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_process_scrolls_filters_short(mock_tqdm, mock_load):
    """Short documents should be filtered out."""
    short_text = "This is a short document. " * 100
    mock_load.return_value = [{"input": short_text}]

    results = _process_scrolls(ENC, 5, 80000)
    assert len(results) == 0


@patch("saturated_blitz_bench.dataset_builder.load_dataset", side_effect=Exception("unavailable"))
def test_process_scrolls_unavailable(mock_load):
    results = _process_scrolls(ENC, 5, 80000)
    assert results == []


@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_process_scrolls_truncates_oversized(mock_tqdm, mock_load):
    """Documents exceeding 70K tokens should be truncated, not skipped."""
    huge_text = _long_text(100000)  # Way over 70K
    mock_load.return_value = [{"input": huge_text}]

    results = _process_scrolls(ENC, 5, 80000)
    assert len(results) >= 1
    # After truncation, should be within [20K, 70K]
    assert results[0]["input_token_count"] <= 70000
    assert results[0]["input_token_count"] >= 20000


# ---------------------------------------------------------------------------
# _process_govreport (mocked)
# ---------------------------------------------------------------------------

@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_process_govreport(mock_tqdm, mock_load):
    long_report = _long_text(30000)
    mock_load.return_value = [{"report": long_report}]

    results = _process_govreport(ENC, 5, 80000)
    assert len(results) >= 1
    assert results[0]["category"] == "long_context"
    assert results[0]["source"] == "govreport"
    assert results[0]["input_token_count"] >= 20000


@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_process_govreport_truncates_oversized(mock_tqdm, mock_load):
    """Oversized reports should be truncated."""
    huge_report = _long_text(100000)
    mock_load.return_value = [{"report": huge_report}]

    results = _process_govreport(ENC, 5, 80000)
    assert len(results) >= 1
    assert results[0]["input_token_count"] <= 70000


@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_process_govreport_filters_short(mock_tqdm, mock_load):
    """Short reports should be filtered out."""
    mock_load.return_value = [{"report": "Short report."}]
    results = _process_govreport(ENC, 5, 80000)
    assert len(results) == 0


# ---------------------------------------------------------------------------
# _process_pg19 (mocked)
# ---------------------------------------------------------------------------

@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_process_pg19(mock_tqdm, mock_load):
    long_book = _long_text(40000)
    mock_load.return_value = [{"text": long_book, "short_book_title": "Test Book"}]

    results = _process_pg19(ENC, 5, 80000)
    assert len(results) >= 1
    assert results[0]["category"] == "long_context"
    assert results[0]["source"] == "pg19"


@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_process_pg19_truncates_oversized(mock_tqdm, mock_load):
    """Full books (100K+ tokens) should be truncated."""
    huge_book = _long_text(120000)
    mock_load.return_value = [{"text": huge_book, "short_book_title": "Long Book"}]

    results = _process_pg19(ENC, 5, 80000)
    assert len(results) >= 1
    assert results[0]["input_token_count"] <= 70000
    assert results[0]["input_token_count"] >= 20000


# ---------------------------------------------------------------------------
# _truncate_text
# ---------------------------------------------------------------------------

def test_truncate_text_within_limit():
    text = "Hello world this is a test sentence."
    result = _truncate_text(text, 100, ENC)
    assert result == text


def test_truncate_text_over_limit():
    long_text = _long_text(1000)
    result = _truncate_text(long_text, 500, ENC)
    result_tokens = len(ENC.encode(result, disallowed_special=()))
    assert result_tokens <= 500


# ---------------------------------------------------------------------------
# _process_gsm8k (mocked)
# ---------------------------------------------------------------------------

@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_process_gsm8k(mock_tqdm, mock_load):
    rows = [
        {"question": (
            "A store had 150 apples in stock at the beginning of the week. On Monday, "
            "they sold 23 apples to customers. On Tuesday, they received a shipment of "
            "45 new apples and sold 31 more. On Wednesday, 12 apples were found to be "
            "spoiled and had to be removed from the inventory. On Thursday they sold "
            "another 18 apples. How many good apples does the store have at the end "
            "of Thursday? Please solve this step by step showing all your calculations."
        )},
        {"question": ""},  # should be skipped
        {"question": "short"},  # too short
    ]
    mock_load.return_value = rows

    results = _process_gsm8k(ENC, 5)
    assert len(results) == 1
    assert results[0]["category"] == "reasoning"
    assert results[0]["source"] == "gsm8k"


# ---------------------------------------------------------------------------
# _process_longbench (mocked)
# ---------------------------------------------------------------------------

@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_process_longbench(mock_tqdm, mock_load):
    long_context = _long_text(30000)
    rows = [{"context": long_context, "input": "Summarize this report."}]
    mock_load.return_value = rows

    results = _process_longbench(ENC, 5, 80000)
    assert len(results) >= 1
    assert results[0]["category"] == "long_context"
    assert results[0]["source"] == "longbench"


# ---------------------------------------------------------------------------
# _process_code_contests (mocked)
# ---------------------------------------------------------------------------

@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_process_code_contests(mock_tqdm, mock_load):
    rows = [{
        "description": (
            "Given an array of n integers, find the contiguous subarray with the maximum sum. "
            "The array can contain both positive and negative integers, and you need to find "
            "the subarray (contiguous elements) whose sum is the largest among all possible "
            "subarrays. If all numbers are negative, the maximum subarray sum is the largest "
            "single element. Your solution should run in O(n) time complexity using Kadane's "
            "algorithm or a similar approach.\n\n"
            "Input Format:\n"
            "The first line contains a single integer n (1 <= n <= 200000) representing the "
            "number of elements in the array. The second line contains n space-separated "
            "integers a_1, a_2, ..., a_n (-10^9 <= a_i <= 10^9) representing the elements "
            "of the array.\n\n"
            "Output Format:\n"
            "Output a single integer representing the maximum subarray sum.\n\n"
            "Constraints:\n"
            "- 1 <= n <= 200000\n"
            "- -10^9 <= a_i <= 10^9\n"
            "- The answer fits in a 64-bit signed integer\n"
            "- Time limit: 2 seconds\n"
            "- Memory limit: 256 MB\n\n"
            "Note: You should handle edge cases such as arrays with a single element, arrays "
            "with all positive elements, arrays with all negative elements, and mixed arrays "
            "with both positive and negative values. Consider the time and memory constraints "
            "carefully when designing your solution. A brute force O(n^2) solution will not "
            "pass within the time limit for large inputs. You need to implement an efficient "
            "algorithm that processes each element at most once. The classical approach known "
            "as Kadane's algorithm maintains a running sum and resets it when it becomes negative. "
            "This greedy approach works because a negative prefix sum can never contribute to "
            "a maximum subarray sum. Initialize your maximum with the first element rather "
            "than zero to correctly handle the case where all elements are negative. Be careful "
            "with integer overflow when summing large values - use 64-bit integers throughout "
            "your computation. Also note that the empty subarray is not considered valid, so "
            "you must select at least one element in your answer.\n\n"
            "Examples and Explanations:\n"
            "For the array [1, -3, 2, 1, -1], the maximum subarray is [2, 1] with sum 3. "
            "The algorithm works by scanning from left to right, maintaining the maximum sum "
            "ending at the current position. At each step, we decide whether to extend the "
            "current subarray or start a new one. If the running sum becomes negative, we "
            "reset it to zero (or in the variant that handles all-negative arrays, we track "
            "the maximum element separately). For the array [-2, -3, -1, -5], the answer is "
            "-1 since we must pick at least one element. For [5, 4, -1, 7, 8], the answer "
            "is 23 which is the sum of the entire array. Your implementation should read from "
            "standard input and write to standard output. Make sure to use fast I/O methods "
            "as the input can be large. In Python, use sys.stdin for faster reading. In C++, "
            "use scanf or ios_base::sync_with_stdio(false) for competitive performance."
        ),
        "public_tests": {
            "input": ["1 2 -1 3", "-1 -2 -3"],
            "output": ["5", "-1"],
        },
        "name": "max_subarray",
    }]
    mock_load.return_value = rows

    results = _process_code_contests(ENC, 5, 80000)
    assert len(results) >= 1
    assert results[0]["category"] == "code_generation"
    assert results[0]["source"] == "code_contests"


# ---------------------------------------------------------------------------
# _process_wildchat (mocked)
# ---------------------------------------------------------------------------

@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_process_wildchat(mock_tqdm, mock_load):
    rows = [
        {
            "language": "English",
            "toxic": False,
            "redacted": False,
            "conversation": [
                {"role": "user", "content": (
                    "Hello, I'm trying to learn how to cook the perfect Italian pasta from scratch. "
                    "Could you please explain the step-by-step process including how to make fresh "
                    "pasta dough, the best flour to use, how long to knead it, and what sauces pair "
                    "well with different pasta shapes? I'm especially interested in fettuccine and "
                    "ravioli recipes that would be good for a dinner party this weekend."
                )},
                {"role": "assistant", "content": "Of course! I'd be happy to help you with that."},
            ],
            "conversation_hash": "abc123",
        },
    ]
    mock_load.return_value = rows

    targets = {"short_chat": 5, "medium_chat": 0, "multi_turn": 0, "reasoning": 0}
    results = _process_wildchat(ENC, targets, 80000)
    assert len(results["short_chat"]) >= 1
    assert results["short_chat"][0]["source"] == "wildchat"


@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_process_wildchat_filters_non_english(mock_tqdm, mock_load):
    rows = [
        {
            "language": "Chinese",
            "conversation": [{"role": "user", "content": "你好"}],
            "conversation_hash": "xyz",
        },
    ]
    mock_load.return_value = rows

    targets = {"short_chat": 5, "medium_chat": 0, "multi_turn": 0, "reasoning": 0}
    results = _process_wildchat(ENC, targets, 80000)
    assert len(results["short_chat"]) == 0


@patch("saturated_blitz_bench.dataset_builder.load_dataset", side_effect=Exception("timeout"))
def test_process_wildchat_handles_load_error(mock_load):
    targets = {"short_chat": 5, "medium_chat": 0, "multi_turn": 0, "reasoning": 0}
    results = _process_wildchat(ENC, targets, 80000)
    assert all(len(v) == 0 for v in results.values())


# ---------------------------------------------------------------------------
# Tool call entry format validation
# ---------------------------------------------------------------------------

@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_tool_call_entry_has_separate_tools_field(mock_tqdm, mock_load):
    """Critical: tools must be a top-level field, not embedded in messages."""
    rows = [_hermes_fc_parsed_row()]
    mock_load.return_value = rows

    results = _process_hermes_fc(ENC, 5, 80000)
    assert len(results) >= 1

    entry = results[0]
    # Tools must be a separate top-level field
    assert "tools" in entry
    assert isinstance(entry["tools"], list)
    assert len(entry["tools"]) > 0

    # Tools must be in OpenAI format
    tool = entry["tools"][0]
    assert tool["type"] == "function"
    assert "function" in tool
    assert "name" in tool["function"]

    # Messages should NOT contain tool definitions
    for msg in entry["messages"]:
        assert msg["role"] in ("system", "user", "assistant")
        assert isinstance(msg["content"], str)


@patch("saturated_blitz_bench.dataset_builder.load_dataset")
@patch("saturated_blitz_bench.dataset_builder.tqdm", side_effect=lambda x, **kw: x)
def test_tool_call_expected_tool_extracted(mock_tqdm, mock_load):
    """expected_tool should have the correct function name and args."""
    rows = [_hermes_fc_parsed_row(
        "I need to get the current weather conditions in New York City for my upcoming "
        "trip. Please check the temperature, humidity, wind speed, precipitation chance, "
        "and provide both Celsius and Fahrenheit readings along with metric units.",
        "get_weather", {"location": "NYC", "units": "metric"},
    )]
    mock_load.return_value = rows

    results = _process_hermes_fc(ENC, 5, 80000)
    assert len(results) >= 1

    et = results[0]["expected_tool"]
    assert et["name"] == "get_weather"
    assert set(et["required_args"]) == {"location", "units"}
