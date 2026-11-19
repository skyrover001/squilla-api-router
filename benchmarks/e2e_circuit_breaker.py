"""E2E: repeated c1 failure should trigger cooldown and skip on request two."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
ENV = {}
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        key, value = line.split("=", 1)
        ENV[key.strip()] = value.strip()


def run() -> dict:
    session = "breaker-" + uuid.uuid4().hex[:8]
    headers = {
        "Authorization": "Bearer " + ENV["ROUTER_API_KEY"],
        "Content-Type": "application/json",
        "X-Squilla-Session-Id": session,
        "X-Squilla-Intent": "new_task",
        "X-Squilla-Cost-Policy": "classifier",
    }
    payload = {
        "model": "router",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": "写一个 Python 脚本读取 JSON 文件并打印指定字段。",
            }
        ],
    }
    started = time.perf_counter()
    with httpx.Client(trust_env=False, timeout=300) as client:
        with client.stream(
            "POST",
            "http://127.0.0.1:8021/v1/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            for _ in response.iter_lines():
                pass
            return {
                "session": session,
                "status": response.status_code,
                "tier": response.headers.get("x-router-tier"),
                "model": response.headers.get("x-router-model"),
                "classifier_tier": response.headers.get("x-squilla-classifier-tier"),
                "latency_s": round(time.perf_counter() - started, 3),
            }


def main() -> None:
    first = run()
    second = run()
    auth = {"Authorization": "Bearer " + ENV["ROUTER_API_KEY"]}
    with httpx.Client(trust_env=False, timeout=30) as client:
        first_usage = client.get(
            "http://127.0.0.1:8021/admin/routing-usage",
            headers=auth,
            params={"session_id": first["session"]},
        ).json()
        second_usage = client.get(
            "http://127.0.0.1:8021/admin/routing-usage",
            headers=auth,
            params={"session_id": second["session"]},
        ).json()
    print(
        json.dumps(
            {
                "first": first,
                "second": second,
                "first_usage": first_usage,
                "second_usage": second_usage,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
