"""E2E A/B: P1 cost-aware policy vs classifier-only tier policy."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_name("e2e_p1_vs_classifier.json")
ENV = {}
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        key, value = line.split("=", 1)
        ENV[key.strip()] = value.strip()
PRICES = {"c0": (1, 10, 40), "c1": (4, 40, 160), "c2": (10, 100, 400), "c3": (20, 200, 800)}
MODEL_TIER = {
    "Qwen3.8-27B": "c0",
    "Qwen3.5-397B-A17B": "c1",
    "GLM-5.3-Flash": "c2",
    "DeepSeek-V4.1-Flash": "c3",
}
SYSTEM = (
    "You are a senior software engineer. Answer accurately and concisely. "
    "Keep the final answer under 180 English words."
)
TASKS = [
    ("json", "写一个 Python 脚本读取 JSON 文件并打印指定字段。"),
    ("pytest", "为 add(a,b) 写 pytest 单元测试，覆盖正常值和异常值。"),
    ("jwt", "用三个要点解释 JWT 的组成和常见安全风险。"),
    ("retry", "写一个带重试、超时和指数退避的 HTTP 客户端。"),
]


def parts(usage: dict[str, Any]) -> tuple[int, int, int, int]:
    inp = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    ide = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    ode = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    return inp, min(inp, int(ide.get("cached_tokens") or 0)), out, int(ode.get("reasoning_tokens") or 0)


def cost(usage: dict[str, Any], tier: str) -> int:
    inp, cached, out, _ = parts(usage)
    hit, miss, output = PRICES[tier]
    return cached * hit + (inp - cached) * miss + out * output


def run(prompt: str, policy: str) -> dict[str, Any]:
    headers = {
        "Authorization": "Bearer " + ENV["ROUTER_API_KEY"],
        "Content-Type": "application/json",
        "X-Squilla-Session-Id": f"p1-{policy}-{uuid.uuid4().hex[:8]}",
        "X-Squilla-Intent": "new_task",
        "X-Squilla-Cost-Policy": policy,
    }
    payload = {
        "model": "router",
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    started = time.perf_counter()
    usage = {}
    with httpx.Client(trust_env=False, timeout=300) as client:
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
            model = response.headers.get("x-router-model") or ""
            tier = MODEL_TIER.get(model, "c3")
            return {
                "status": response.status_code,
                "policy": response.headers.get("x-squilla-cost-policy"),
                "classifier_tier": response.headers.get("x-squilla-classifier-tier"),
                "cost_aware": response.headers.get("x-squilla-cost-aware"),
                "route_tier": response.headers.get("x-router-tier"),
                "model": model,
                "tier": tier,
                "reasoning": response.headers.get("x-router-reasoning"),
                "usage": usage,
                "cost_units": cost(usage, tier),
                "latency_s": round(time.perf_counter() - started, 3),
            }


def main() -> None:
    rows = []
    for task, prompt in TASKS:
        classifier = run(prompt, "classifier")
        aware = run(prompt, "cost_aware")
        rows.append(
            {
                "task": task,
                "prompt": prompt,
                "classifier": classifier,
                "cost_aware": aware,
                "comparison": {
                    "cost_reduction_pct": (
                        round((1 - aware["cost_units"] / classifier["cost_units"]) * 100, 2)
                        if classifier["cost_units"]
                        else None
                    ),
                    "latency_reduction_pct": (
                        round((1 - aware["latency_s"] / classifier["latency_s"]) * 100, 2)
                        if classifier["latency_s"]
                        else None
                    ),
                },
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    classifier_cost = sum(row["classifier"]["cost_units"] for row in rows)
    aware_cost = sum(row["cost_aware"]["cost_units"] for row in rows)
    classifier_latency = sum(row["classifier"]["latency_s"] for row in rows)
    aware_latency = sum(row["cost_aware"]["latency_s"] for row in rows)
    summary = {
        "tasks": len(rows),
        "classifier_cost_units": classifier_cost,
        "cost_aware_units": aware_cost,
        "cost_reduction_pct": (
            round((1 - aware_cost / classifier_cost) * 100, 2)
            if classifier_cost
            else None
        ),
        "classifier_latency_s": round(classifier_latency, 3),
        "cost_aware_latency_s": round(aware_latency, 3),
        "latency_reduction_pct": (
            round((1 - aware_latency / classifier_latency) * 100, 2)
            if classifier_latency
            else None
        ),
    }
    OUT.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
