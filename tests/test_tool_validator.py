"""Tests for tool call validation."""

from saturated_blitz_bench.workload.tool_validator import (
    extract_tool_calls_from_chunks,
    validate_tool_call,
)


def test_validate_correct_tool_call():
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco", "unit": "celsius"}',
            },
        }
    ]
    expected = {"name": "get_weather", "required_args": ["location"]}
    assert validate_tool_call("", tool_calls, expected) is True


def test_validate_wrong_tool_name():
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "search_flights",
                "arguments": '{"origin": "SFO"}',
            },
        }
    ]
    expected = {"name": "book_flight", "required_args": ["origin"]}
    assert validate_tool_call("", tool_calls, expected) is False


def test_validate_missing_required_args():
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"unit": "celsius"}',
            },
        }
    ]
    expected = {"name": "get_weather", "required_args": ["location"]}
    assert validate_tool_call("", tool_calls, expected) is False


def test_validate_no_tool_calls():
    expected = {"name": "get_weather", "required_args": []}
    assert validate_tool_call("", None, expected) is False
    assert validate_tool_call("", [], expected) is False


def test_validate_not_a_tool_prompt():
    assert validate_tool_call("", None, None) is None


def test_validate_invalid_json_args():
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "not valid json",
            },
        }
    ]
    expected = {"name": "get_weather", "required_args": []}
    assert validate_tool_call("", tool_calls, expected) is False


def test_extract_tool_calls_from_chunks():
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "get_", "arguments": ""},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "weather", "arguments": '{"loc'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": 'ation": "SF"}'},
                            }
                        ]
                    }
                }
            ]
        },
    ]

    result = extract_tool_calls_from_chunks(chunks)
    assert len(result) == 1
    assert result[0]["function"]["name"] == "get_weather"
    assert result[0]["function"]["arguments"] == '{"location": "SF"}'
    assert result[0]["id"] == "call_abc"


def test_extract_multiple_tool_calls():
    """Multiple tool calls with different indices are assembled separately."""
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"loc": "NY"}'},
                            },
                            {
                                "index": 1,
                                "id": "call_2",
                                "type": "function",
                                "function": {"name": "get_time", "arguments": '{"tz": "EST"}'},
                            },
                        ]
                    }
                }
            ]
        },
    ]
    result = extract_tool_calls_from_chunks(chunks)
    assert len(result) == 2
    assert result[0]["function"]["name"] == "get_weather"
    assert result[1]["function"]["name"] == "get_time"
    assert result[0]["id"] == "call_1"
    assert result[1]["id"] == "call_2"


def test_validate_args_not_dict():
    """Arguments that parse to a non-dict (array) should fail validation."""
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '["San Francisco"]',
            },
        }
    ]
    expected = {"name": "get_weather", "required_args": []}
    assert validate_tool_call("", tool_calls, expected) is False
