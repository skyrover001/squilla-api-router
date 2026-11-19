from squilla_api_router.quality_signals import (
    StreamToolCallTracker,
    response_quality_failure,
)


def test_chat_invalid_tool_json_is_detected() -> None:
    result = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "write",
                                "arguments": "{invalid",
                            }
                        }
                    ]
                }
            }
        ]
    }
    assert response_quality_failure(result, "openai_chat") == "tool_call_invalid_json"


def test_responses_missing_tool_name_is_detected() -> None:
    result = {
        "output": [
            {
                "type": "function_call",
                "name": "",
                "arguments": "{}",
            }
        ]
    }
    assert (
        response_quality_failure(result, "openai_responses")
        == "tool_call_missing_name"
    )


def test_stream_tool_arguments_are_accumulated() -> None:
    tracker = StreamToolCallTracker()
    tracker.feed(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "name": "write",
                                    "arguments": '{"path":',
                                },
                            }
                        ]
                    }
                }
            ]
        }
    )
    tracker.feed(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '"a.py"}'},
                            }
                        ]
                    }
                }
            ]
        }
    )
    assert tracker.failure_reason() is None

    tracker.feed(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "function": {
                                    "name": "read",
                                    "arguments": "{broken",
                                },
                            }
                        ]
                    }
                }
            ]
        }
    )
    assert tracker.failure_reason() == "tool_call_invalid_json"


def test_anthropic_tool_input_accepts_dict() -> None:
    result = {
        "content": [
            {
                "type": "tool_use",
                "name": "write",
                "input": {"path": "a.py"},
            }
        ]
    }
    assert response_quality_failure(result, "anthropic") is None


def test_stream_signature_is_stable_for_same_call() -> None:
    tracker = StreamToolCallTracker()
    event = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "name": "read",
                                "arguments": '{"path":"a.py"}',
                            },
                        }
                    ]
                }
            }
        ]
    }
    tracker.feed(event)
    assert tracker.signature() == tracker.signature()
    assert "read" in tracker.signature()
