"""E2E check: full provider-visible context can raise the execution tier."""

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

context = "A" * 78_000
payload = {
    "model": "router",
    "stream": True,
    "max_tokens": 16,
    "messages": [
        {
            "role": "system",
            "content": "Reference context follows. Ignore it except for capacity testing.\n" + context,
        },
        {"role": "user", "content": "只回复 OK。"},
    ],
}
headers = {
    "Authorization": "Bearer " + ENV["ROUTER_API_KEY"],
    "Content-Type": "application/json",
    "X-Squilla-Session-Id": "large-context-" + uuid.uuid4().hex[:8],
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
        result = {
            "status": response.status_code,
            "classifier_tier": response.headers.get("x-squilla-classifier-tier"),
            "large_context_floor": response.headers.get("x-squilla-large-context-floor"),
            "route_tier": response.headers.get("x-router-tier"),
            "model": response.headers.get("x-router-model"),
            "latency_s": round(time.perf_counter() - started, 3),
        }
print(json.dumps(result, ensure_ascii=False))
