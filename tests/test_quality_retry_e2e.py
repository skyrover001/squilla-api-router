from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

import app as router_app
from squilla_api_router.routing_state import RoutingStateStore


class MockBackendHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        model = payload.get("model")
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        if model == "mock-c0":
            chunks = [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "bad-call",
                                        "function": {
                                            "name": "write",
                                            "arguments": "{invalid",
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {"delta": {}, "finish_reason": "tool_calls"}
                    ]
                },
            ]
        else:
            chunks = [
                {"choices": [{"delta": {"content": "fixed"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
        chunks.append(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }
        )
        for chunk in chunks:
            self.wfile.write(
                ("data: " + json.dumps(chunk) + "\n\n").encode()
            )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format: str, *args) -> None:
        return


def test_stream_invalid_tool_call_marks_next_turn_redo(monkeypatch) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockBackendHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    db_path = Path(__file__).with_name(f".quality-retry-{uuid4().hex}.db")
    try:
        monkeypatch.setattr(router_app, "ROUTER_API_KEY", "")
        monkeypatch.setattr(router_app, "ROUTER_SESSION_STICKY", True)
        monkeypatch.setattr(router_app, "ROUTER_COST_AWARE_ENABLED", False)
        monkeypatch.setattr(router_app, "ROUTER_REASONING_ENABLED", False)
        monkeypatch.setattr(
            router_app,
            "PROVIDERS",
            {
                "mock": {
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "api_key": "x",
                    "format": "openai_chat",
                }
            },
        )
        monkeypatch.setattr(
            router_app,
            "TIERS",
            {
                "c0": {
                    "provider": "mock",
                    "model": "mock-c0",
                    "supports_image": False,
                    "supports_video": False,
                    "reasoning": None,
                },
                "c1": {
                    "provider": "mock",
                    "model": "mock-c1",
                    "supports_image": False,
                    "supports_video": False,
                    "reasoning": None,
                },
            },
        )
        monkeypatch.setattr(
            router_app,
            "ROUTE_CLASS_TO_TIER",
            {"R0": "c0", "R1": "c1"},
        )
        monkeypatch.setattr(
            router_app,
            "_ROUTING_STATE_STORE",
            RoutingStateStore(db_path),
        )
        monkeypatch.setattr(router_app, "_CIRCUIT_BREAKER", None)
        monkeypatch.setattr(
            router_app,
            "_classify",
            lambda *args, **kwargs: {
                "route_class": "R0",
                "raw_route_class": "R0",
                "confidence": 1.0,
                "thinking_mode": "T0",
                "prompt_policy": "P0",
                "material_tokens": 1,
                "probabilities": {"R0": 1.0},
            },
        )

        session_id = "quality-e2e-" + uuid4().hex[:8]
        headers = {"X-Squilla-Session-Id": session_id}
        payload = {
            "model": "router",
            "stream": True,
            "messages": [{"role": "user", "content": "write a file"}],
        }
        with TestClient(router_app.app) as client:
            first = client.post(
                "/v1/chat/completions", headers=headers, json=payload
            )
            assert first.status_code == 200
            assert first.headers["x-router-tier"] == "R0"
            state = router_app._routing_state_store().get(session_id)
            assert state is not None
            assert state.forced_intent == "redo"
            assert "tool_call_invalid_json" in state.forced_reason

            second = client.post(
                "/v1/chat/completions", headers=headers, json=payload
            )
            assert second.status_code == 200
            assert second.headers["x-squilla-intent"] == "redo"
            assert second.headers["x-router-tier"] == "R1"
            final_state = router_app._routing_state_store().get(session_id)
            assert final_state is not None
            assert final_state.tier == "c1"
            assert final_state.forced_intent == ""
    finally:
        server.shutdown()
        server.server_close()
        db_path.unlink(missing_ok=True)
