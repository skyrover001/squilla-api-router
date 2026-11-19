"""P1 E2E: OpenSquilla-style c1 choice vs cost-aware c2 choice."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_name("e2e_cost_aware_vs_opensquilla.json")
ENV = {}
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        key, value = line.split("=", 1)
        ENV[key.strip()] = value.strip()

SYSTEM = (
    "You are a senior software engineer. Answer accurately and concisely. "
    "Keep the final answer under 180 English words."
)
PROMPT = "写一个 Python 脚本读取 JSON 文件并打印指定字段。"
PRICES = {"c0": (1, 10, 40), "c1": (4, 40, 160), "c2": (10, 100, 400), "c3": (20, 200, 800)}


def usage_parts(usage: dict[str, Any]) -> tuple[int, int, int]:
    inp = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = min(inp, int(details.get("cached_tokens") or 0))
    return inp, cached, out


def cost(usage: dict[str, Any], tier: str) -> int:
    inp, cached, out = usage_parts(usage)
    hit, miss, output = PRICES[tier]
    return cached * hit + (inp - cached) * miss + out * output


def run_router() -> dict[str, Any]:
    headers = {
        "Authorization": "Bearer " + ENV["ROUTER_API_KEY"],
        "Content-Type": "application/json",
        "X-Squilla-Session-Id": "p1-cost-aware-e2e",
        "X-Squilla-Intent": "new_task",
    }
    payload = {
        "model": "router",
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT},
        ],
    }
    started = time.perf_counter()
    usage = {}
    with httpx.Client(trust_env=False, timeout=240) as client:
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
            return {
                "status": response.status_code,
                "classifier_tier": response.headers.get("x-squilla-classifier-tier"),
                "cost_aware": response.headers.get("x-squilla-cost-aware"),
                "tier": response.headers.get("x-router-tier"),
                "model": response.headers.get("x-router-model"),
                "reasoning": response.headers.get("x-router-reasoning"),
                "usage": usage,
                "cost_units": cost(usage, "c2"),
                "latency_s": round(time.perf_counter() - started, 3),
            }


def run_opensquilla_policy() -> dict[str, Any]:
    payload = {
        "model": "Qwen3.5-397B-A17B",
        "stream": True,
        "input": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT},
        ],
        "reasoning": {"effort": "low"},
    }
    headers = {
        "Authorization": "Bearer " + ENV["STARFIRE_API_KEY"],
        "Content-Type": "application/json",
    }
    started = time.perf_counter()
    usage = {}
    with httpx.Client(trust_env=False, timeout=240) as client:
        with client.stream(
            "POST",
            ENV["STARFIRE_BASE_URL"].rstrip("/") + "/responses",
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
                if event.get("type") == "response.completed":
                    usage = (event.get("response") or {}).get("usage") or {}
            return {
                "status": response.status_code,
                "tier": "c1",
                "model": "Qwen3.5-397B-A17B",
                "reasoning": "low",
                "usage": usage,
                "cost_units": cost(usage, "c1"),
                "latency_s": round(time.perf_counter() - started, 3),
            }


def main() -> None:
    routed = run_router()
    baseline = run_opensquilla_policy()
    baseline_source = "live"
    if baseline["status"] != 200 or not baseline["usage"]:
        historical = json.loads(
            (ROOT / "benchmarks" / "e2e_comprehensive_same_reasoning_nolimit.json")
            .read_text(encoding="utf-8")
        )
        row = next(
            item for item in historical["results"] if item["task"] == "r1_read_json"
        )["routed"]
        baseline = {
            "status": row["status"],
            "tier": row["tier"],
            "model": row["model"],
            "reasoning": row["effort"],
            "usage": row["usage"],
            "cost_units": row["cost_units"],
            "latency_s": row["latency_s"],
            "source": "historical_e2e_same_prompt",
        }
        baseline_source = "historical_e2e_same_prompt"
    result = {
        "task": PROMPT,
        "router_cost_aware": routed,
        "opensquilla_style_c1": baseline,
        "baseline_source": baseline_source,
        "comparison": {
            "cost_reduction_pct": (
                round((1 - routed["cost_units"] / baseline["cost_units"]) * 100, 2)
                if baseline["cost_units"]
                else None
            ),
            "latency_reduction_pct": (
                round((1 - routed["latency_s"] / baseline["latency_s"]) * 100, 2)
                if baseline["latency_s"]
                else None
            ),
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
