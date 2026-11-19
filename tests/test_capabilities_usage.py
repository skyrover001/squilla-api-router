from __future__ import annotations

import app as router_app
from squilla_api_router.cost_policy import estimate_capability_input_tokens
from squilla_api_router.usage_estimation import (
    StreamOutputTokenEstimator,
    estimate_response_output_tokens,
    estimate_text_tokens,
)


def _tiers() -> dict[str, dict]:
    return {
        "c0": {
            "provider": "mock",
            "model": "mock-c0",
            "supports_image": True,
            "supports_video": False,
            "context_length": 100,
        },
        "c1": {
            "provider": "mock",
            "model": "mock-c1",
            "supports_image": False,
            "supports_video": True,
            "context_length": 10_000,
        },
        "c2": {
            "provider": "mock",
            "model": "mock-c2",
            "supports_image": True,
            "supports_video": True,
            "context_length": 100_000,
        },
    }


def test_media_and_context_capability_filtering(monkeypatch) -> None:
    monkeypatch.setattr(router_app, "TIERS", _tiers())
    monkeypatch.setattr(
        router_app,
        "PROVIDERS",
        {"mock": {"base_url": "http://mock/v1", "api_key": "x", "format": "openai_chat"}},
    )
    body = {"messages": [{"role": "user", "content": "hello"}]}

    assert router_app._matching_tiers(body) == ["c0", "c1", "c2"]
    assert router_app._matching_tiers(body, require_image=True) == ["c0", "c2"]
    assert router_app._matching_tiers(body, require_video=True) == ["c1", "c2"]
    assert router_app._matching_tiers(
        body,
        require_image=True,
        require_video=True,
    ) == ["c2"]

    large_body = {"messages": [{"role": "user", "content": "x" * 2000}]}
    assert "c0" not in router_app._matching_tiers(large_body)
    assert "c1" in router_app._matching_tiers(large_body)


def test_capability_estimate_omits_embedded_media_bytes() -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + ("A" * 100_000)
                        },
                    },
                ],
            }
        ]
    }

    estimated = estimate_capability_input_tokens(body, media_tokens=1024)

    assert estimated < 3000


def test_fallback_chain_respects_video_candidates(monkeypatch) -> None:
    monkeypatch.setattr(router_app, "TIERS", _tiers())
    monkeypatch.setattr(
        router_app,
        "PROVIDERS",
        {"mock": {"base_url": "http://mock/v1", "api_key": "x", "format": "openai_chat"}},
    )

    chain = router_app._fallback_chain(
        "R0",
        {},
        vision=True,
        video=True,
        valid_tiers=["c1", "c2"],
    )

    assert [tier for _, tier in chain] == ["c1", "c2"]


def test_usage_estimation_supports_all_response_formats() -> None:
    chat = {
        "choices": [
            {
                "message": {
                    "content": "hello world",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"a.py"}',
                            }
                        }
                    ],
                }
            }
        ]
    }
    responses = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "hello world"}],
            }
        ]
    }
    anthropic = {
        "content": [
            {"type": "text", "text": "hello world"},
            {"type": "tool_use", "name": "read_file", "input": {"path": "a.py"}},
        ]
    }

    assert estimate_response_output_tokens(chat, "openai_chat") > 0
    assert estimate_response_output_tokens(responses, "openai_responses") > 0
    assert estimate_response_output_tokens(anthropic, "anthropic") > 0
    assert estimate_text_tokens("hello") == 2


def test_stream_usage_estimator_accumulates_content_only() -> None:
    estimator = StreamOutputTokenEstimator()
    estimator.feed(
        {
            "choices": [
                {"delta": {"role": "assistant", "content": "hello"}}
            ]
        }
    )
    estimator.feed(
        {
            "choices": [
                {"delta": {"content": " world"}, "finish_reason": "stop"}
            ]
        }
    )

    assert estimator.tokens == estimate_text_tokens("hello world")
