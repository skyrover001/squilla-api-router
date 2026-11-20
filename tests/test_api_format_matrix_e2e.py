from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import app as router_app
from squilla_api_router.backend_health import CircuitBreaker
from squilla_api_router.routing_state import RoutingStateStore


FORMATS = ["openai_chat", "openai_responses", "anthropic"]
FORMAT_TO_PATH = {
    "openai_chat": "/chat/completions",
    "openai_responses": "/responses",
    "anthropic": "/messages",
}
PATH_TO_FORMAT = {value: key for key, value in FORMAT_TO_PATH.items()}
INBOUND_PATHS = {
    "openai_chat": "/v1/chat/completions",
    "openai_responses": "/v1/responses",
    "anthropic": "/v1/messages",
}


def _usage_free_chat_response(model: str) -> dict:
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "mock answer"},
                "finish_reason": "stop",
            }
        ],
    }


def _usage_free_responses_response(model: str) -> dict:
    return {
        "id": "resp_mock",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": "msg_mock",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "mock answer",
                        "annotations": [],
                    }
                ],
            }
        ],
    }


def _usage_free_anthropic_response(model: str) -> dict:
    return {
        "id": "msg_mock",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "mock answer"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
    }


def _chat_stream(model: str) -> list[dict]:
    return [
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "mock answer"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]


def _responses_stream(model: str) -> list[dict]:
    response = _usage_free_responses_response(model)
    return [
        {"type": "response.created", "response": response},
        {
            "type": "response.output_text.delta",
            "item_id": "msg_mock",
            "output_index": 0,
            "content_index": 0,
            "delta": "mock answer",
        },
        {"type": "response.completed", "response": response},
    ]


def _anthropic_stream(model: str) -> list[dict]:
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_mock",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "mock answer"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
        },
        {"type": "message_stop"},
    ]


class MockBackendHandler(BaseHTTPRequestHandler):
    received: list[tuple[str, dict]] = []
    lock = threading.Lock()

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        with self.lock:
            self.received.append((self.path, body))
        fmt = PATH_TO_FORMAT[self.path]
        model = str(body.get("model") or "")
        if body.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            if fmt == "openai_responses":
                events = _responses_stream(model)
            elif fmt == "anthropic":
                events = _anthropic_stream(model)
            else:
                events = _chat_stream(model)
            for event in events:
                self.wfile.write(
                    ("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode()
                )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        if fmt == "openai_responses":
            result = _usage_free_responses_response(model)
        elif fmt == "anthropic":
            result = _usage_free_anthropic_response(model)
        else:
            result = _usage_free_chat_response(model)
        payload = json.dumps(result, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        return


@pytest.fixture(scope="module")
def mock_backend() -> tuple[str, type[MockBackendHandler]]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockBackendHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", MockBackendHandler
    finally:
        server.shutdown()
        server.server_close()


def _configure_router(
    monkeypatch,
    base_url: str,
    backend_format: str,
    *,
    tiers: dict[str, dict] | None = None,
) -> Path:
    db_path = Path(__file__).with_name(f".matrix-{uuid4().hex}.db")
    monkeypatch.setattr(router_app, "ROUTER_API_KEY", "")
    monkeypatch.setattr(router_app, "ROUTER_COST_AWARE_ENABLED", False)
    monkeypatch.setattr(router_app, "ROUTER_REASONING_ENABLED", False)
    monkeypatch.setattr(router_app, "ROUTER_SESSION_STICKY", False)
    monkeypatch.setattr(
        router_app,
        "PROVIDERS",
        {
            "mock": {
                "base_url": base_url,
                "api_key": "x",
                "format": backend_format,
            }
        },
    )
    monkeypatch.setattr(
        router_app,
        "TIERS",
        tiers
        or {
            f"c{index}": {
                "provider": "mock",
                "model": f"mock-c{index}",
                "supports_image": True,
                "supports_video": True,
                "context_length": 1_000_000,
                "reasoning": None,
            }
            for index in range(4)
        },
    )
    monkeypatch.setattr(
        router_app,
        "ROUTE_CLASS_TO_TIER",
        {f"R{index}": f"c{index}" for index in range(4)},
    )
    monkeypatch.setattr(
        router_app,
        "_classify",
        lambda *args, valid_tiers=None, **kwargs: {
            "route_class": f"R{(valid_tiers or ['c0'])[0][1:]}",
            "raw_route_class": f"R{(valid_tiers or ['c0'])[0][1:]}",
            "confidence": 1.0,
            "thinking_mode": "T0",
            "prompt_policy": "P0",
            "material_tokens": 1,
            "probabilities": {},
        },
    )
    monkeypatch.setattr(
        router_app,
        "_ROUTING_STATE_STORE",
        RoutingStateStore(db_path),
    )
    monkeypatch.setattr(
        router_app,
        "_CIRCUIT_BREAKER",
        CircuitBreaker(enabled=False),
    )
    return db_path


def _request_body(fmt: str, *, stream: bool, media: str | None = None) -> dict:
    media_part: dict | None = None
    if media == "video":
        media_part = {"type": "video_url", "video_url": {"url": "https://x/v.mp4"}}
    elif media == "image":
        media_part = {"type": "image_url", "image_url": {"url": "https://x/i.png"}}
    elif media == "image_video":
        return {
            "model": "router",
            "stream": stream,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://x/i.png"},
                        },
                        {
                            "type": "video_url",
                            "video_url": {"url": "https://x/v.mp4"},
                        },
                    ],
                }
            ],
        }

    if fmt == "openai_responses":
        content: list[dict] = [{"type": "input_text", "text": "hello"}]
        if media_part:
            content.append(media_part)
        return {
            "model": "router",
            "stream": stream,
            "max_output_tokens": 32,
            "input": [{"role": "user", "content": content}],
        }
    if fmt == "anthropic":
        content = "hello"
        if media_part:
            content = [{"type": "text", "text": "hello"}, media_part]
        return {
            "model": "router",
            "stream": stream,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": content}],
        }
    content = "hello"
    if media_part:
        content = [{"type": "text", "text": "hello"}, media_part]
    return {
        "model": "router",
        "stream": stream,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": content}],
    }


@pytest.mark.parametrize("inbound_format", FORMATS)
@pytest.mark.parametrize("backend_format", FORMATS)
@pytest.mark.parametrize("stream", [False, True])
def test_all_api_format_paths(
    monkeypatch,
    mock_backend,
    inbound_format: str,
    backend_format: str,
    stream: bool,
) -> None:
    base_url, handler = mock_backend
    db_path = _configure_router(monkeypatch, base_url, backend_format)
    session_id = f"matrix-{uuid4().hex}"
    body = _request_body(inbound_format, stream=stream)
    try:
        with TestClient(router_app.app) as client:
            if stream:
                with client.stream(
                    "POST",
                    INBOUND_PATHS[inbound_format],
                    headers={"X-Squilla-Session-Id": session_id},
                    json=body,
                ) as response:
                    payload = b"".join(response.iter_bytes())
            else:
                response = client.post(
                    INBOUND_PATHS[inbound_format],
                    headers={"X-Squilla-Session-Id": session_id},
                    json=body,
                )
                payload = response.content

        assert response.status_code == 200
        expected_converted = inbound_format != backend_format
        if stream:
            assert response.headers["x-router-tier"] == "R0"
            assert response.headers["x-router-model"] == "mock-c0"
            assert response.headers["x-router-converted"] == (
                "1" if expected_converted else "0"
            )
        else:
            router_meta = json.loads(payload)["_router"]
            assert router_meta["tier"] == "R0"
            assert router_meta["model"] == "mock-c0"
            assert router_meta["converted"] is expected_converted
        if inbound_format == "openai_chat":
            if stream:
                assert b'"object": "chat.completion.chunk"' in payload
            else:
                assert "choices" in json.loads(payload)
        elif inbound_format == "openai_responses":
            if stream:
                assert b'"type": "response.' in payload
            else:
                assert "output" in json.loads(payload)
        else:
            if stream:
                assert b'"type": "message_' in payload
            else:
                assert "content" in json.loads(payload)
        assert handler.received
        assert handler.received[-1][0] == FORMAT_TO_PATH[backend_format]
    finally:
        db_path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("media", "expected_tier", "expected_model"),
    [
        ("image", "R0", "mock-c0"),
        ("video", "R1", "mock-c1"),
        ("image_video", "R2", "mock-c2"),
    ],
)
def test_multimodal_capability_routing(
    monkeypatch,
    mock_backend,
    media: str,
    expected_tier: str,
    expected_model: str,
) -> None:
    base_url, _ = mock_backend
    tiers = {
        "c0": {
            "provider": "mock",
            "model": "mock-c0",
            "supports_image": True,
            "supports_video": False,
            "context_length": 1_000_000,
        },
        "c1": {
            "provider": "mock",
            "model": "mock-c1",
            "supports_image": False,
            "supports_video": True,
            "context_length": 1_000_000,
        },
        "c2": {
            "provider": "mock",
            "model": "mock-c2",
            "supports_image": True,
            "supports_video": True,
            "context_length": 1_000_000,
        },
        "c3": {
            "provider": "mock",
            "model": "mock-c3",
            "supports_image": False,
            "supports_video": False,
            "context_length": 1_000_000,
        },
    }
    db_path = _configure_router(
        monkeypatch,
        base_url,
        "openai_chat",
        tiers=tiers,
    )
    try:
        with TestClient(router_app.app) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"X-Squilla-Session-Id": f"media-{uuid4().hex}"},
                json=_request_body("openai_chat", stream=False, media=media),
            )
        assert response.status_code == 200
        router_meta = response.json()["_router"]
        assert router_meta["tier"] == expected_tier
        assert router_meta["model"] == expected_model
    finally:
        db_path.unlink(missing_ok=True)


@pytest.mark.parametrize("stream", [False, True])
def test_missing_usage_is_estimated_and_marked(
    monkeypatch,
    mock_backend,
    stream: bool,
) -> None:
    base_url, _ = mock_backend
    db_path = _configure_router(monkeypatch, base_url, "openai_chat")
    session_id = f"usage-{uuid4().hex}"
    try:
        with TestClient(router_app.app) as client:
            if stream:
                with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    headers={"X-Squilla-Session-Id": session_id},
                    json=_request_body("openai_chat", stream=True),
                ) as response:
                    b"".join(response.iter_bytes())
            else:
                response = client.post(
                    "/v1/chat/completions",
                    headers={"X-Squilla-Session-Id": session_id},
                    json=_request_body("openai_chat", stream=False),
                )
        assert response.status_code == 200
        calls = router_app._routing_state_store().list_calls(session_id)
        assert len(calls) == 1
        assert calls[0].input_tokens > 0
        assert calls[0].output_tokens > 0
        assert calls[0].usage_source in {"estimated", "mixed"}
    finally:
        db_path.unlink(missing_ok=True)


def test_no_capable_video_backend_returns_503(monkeypatch, mock_backend) -> None:
    base_url, _ = mock_backend
    tiers = {
        "c0": {
            "provider": "mock",
            "model": "mock-c0",
            "supports_image": True,
            "supports_video": False,
            "context_length": 1_000_000,
        }
    }
    db_path = _configure_router(
        monkeypatch,
        base_url,
        "openai_chat",
        tiers=tiers,
    )
    try:
        with TestClient(router_app.app) as client:
            response = client.post(
                "/v1/chat/completions",
                json=_request_body("openai_chat", stream=False, media="video"),
            )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "NO_CAPABLE_BACKEND"
    finally:
        db_path.unlink(missing_ok=True)
