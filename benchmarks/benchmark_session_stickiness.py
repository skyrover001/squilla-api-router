"""Compare sticky Session routing with stateless per-turn routing."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_name("e2e_session_stickiness.json")
ENV = {}
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        key, value = line.split("=", 1)
        ENV[key.strip()] = value.strip()

AUTH = {
    "Authorization": "Bearer " + ENV["ROUTER_API_KEY"],
    "Content-Type": "application/json",
}
PRICES = {
    "c0": (1, 10, 40),
    "c1": (4, 40, 160),
    "c2": (10, 100, 400),
    "c3": (20, 200, 800),
}
MODEL_TIER = {
    "Qwen3.8-27B": "c0",
    "Qwen3.5-397B-A17B": "c1",
    "GLM-5.3-Flash": "c2",
    "DeepSeek-V4.1-Flash": "c3",
}
TURNS = [
    ("new_translate", "把 hello world 翻译成中文。"),
    ("continue_translate", "请继续，把翻译改成正式语气。"),
    ("continue_short", "再继续，保持简短。"),
    ("new_palindrome", "现在换一个任务：写一个 Python 函数判断回文。"),
    ("continue_tests", "继续，为这个函数补两个测试。"),
    ("redo_tests", "重新生成一次。"),
    ("new_design", "现在换一个任务：设计分布式订单系统。"),
    ("continue_recovery", "继续，补充故障恢复方案。"),
]


def usage_parts(usage: dict[str, Any]) -> tuple[int, int, int, int]:
    inp = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    inp_details = (
        usage.get("prompt_tokens_details")
        or usage.get("input_tokens_details")
        or {}
    )
    out_details = (
        usage.get("completion_tokens_details")
        or usage.get("output_tokens_details")
        or {}
    )
    cached = min(inp, int(inp_details.get("cached_tokens") or 0))
    reasoning = int(out_details.get("reasoning_tokens") or 0)
    return inp, cached, out, reasoning


def cost_units(usage: dict[str, Any], tier: str) -> int:
    inp, cached, out, _ = usage_parts(usage)
    hit, miss, output = PRICES[tier]
    return cached * hit + (inp - cached) * miss + out * output


def run_turn(
    messages: list[dict[str, Any]], headers: dict[str, str]
) -> tuple[dict[str, Any], str]:
    payload = {
        "model": "router",
        "stream": True,
        "max_tokens": 128,
        "messages": messages,
    }
    started = time.perf_counter()
    usage: dict[str, Any] = {}
    text = ""
    with httpx.Client(trust_env=False, timeout=180) as client:
        with client.stream(
            "POST",
            "http://127.0.0.1:8021/v1/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    continue
                try:
                    event = json.loads(raw)
                except Exception:
                    continue
                if event.get("usage"):
                    usage = event["usage"]
                delta = (event.get("choices") or [{}])[0].get("delta", {})
                text += str(delta.get("content") or "")
            tier = MODEL_TIER.get(
                response.headers.get("x-router-model") or "", "c3"
            )
            return (
                {
                    "status": response.status_code,
                    "intent": response.headers.get("x-squilla-intent"),
                    "reason": response.headers.get("x-squilla-intent-reason"),
                    "sticky": response.headers.get("x-squilla-sticky"),
                    "task_id": response.headers.get("x-squilla-task-id"),
                    "route_header": response.headers.get("x-router-tier"),
                    "model": response.headers.get("x-router-model"),
                    "tier": tier,
                    "usage": usage,
                    "cost_units": cost_units(usage, tier),
                    "latency_s": round(time.perf_counter() - started, 3),
                    "content_chars": len(text),
                },
                text,
            )


def run_mode(mode: str) -> dict[str, Any]:
    session = f"bench-{mode}-{uuid.uuid4().hex[:8]}"
    messages: list[dict[str, Any]] = []
    rows = []
    for index, (name, prompt) in enumerate(TURNS, start=1):
        messages.append({"role": "user", "content": prompt})
        headers = dict(AUTH)
        headers["X-Squilla-Session-Id"] = (
            session if mode == "sticky" else f"stateless-{index}-{uuid.uuid4().hex[:8]}"
        )
        result, text = run_turn(messages, headers)
        result["turn"] = name
        rows.append(result)
        messages.append({"role": "assistant", "content": text})
    changes = 0
    previous = None
    for row in rows:
        key = (row["model"], row["tier"])
        if previous is not None and key != previous:
            changes += 1
        previous = key
    return {
        "session": session,
        "rows": rows,
        "switches": changes,
        "models": sorted({row["model"] for row in rows if row["model"]}),
        "input_tokens": sum(usage_parts(row["usage"])[0] for row in rows),
        "cached_tokens": sum(usage_parts(row["usage"])[1] for row in rows),
        "output_tokens": sum(usage_parts(row["usage"])[2] for row in rows),
        "reasoning_tokens": sum(usage_parts(row["usage"])[3] for row in rows),
        "cost_units": sum(row["cost_units"] for row in rows),
        "latency_s": round(sum(row["latency_s"] for row in rows), 3),
    }


def main() -> None:
    sticky = run_mode("sticky")
    stateless = run_mode("stateless")
    with httpx.Client(trust_env=False, timeout=30) as client:
        ledger = client.get(
            "http://127.0.0.1:8021/admin/routing-usage",
            headers=AUTH,
            params={"session_id": sticky["session"]},
        ).json()
    result = {
        "policy_reference": {
            "opensquilla": "Continue retains tier; Redo advances one tier",
            "stateless": "every turn treated as a new task",
        },
        "sticky": sticky,
        "stateless": stateless,
        "ledger": ledger,
        "comparison": {
            "switch_delta": stateless["switches"] - sticky["switches"],
            "cache_hit_delta": sticky["cached_tokens"] - stateless["cached_tokens"],
            "token_delta": (
                sticky["input_tokens"]
                + sticky["output_tokens"]
                - stateless["input_tokens"]
                - stateless["output_tokens"]
            ),
            "cost_saving_pct": round(
                (1 - sticky["cost_units"] / stateless["cost_units"]) * 100, 2
            ),
            "latency_reduction_pct": round(
                (1 - sticky["latency_s"] / stateless["latency_s"]) * 100, 2
            ),
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
