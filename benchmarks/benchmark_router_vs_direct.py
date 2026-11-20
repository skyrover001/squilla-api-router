"""Compare Squilla routing against a fixed strongest-model baseline.

The baseline sends every task directly to DeepSeek-V4.1-Flash through
starfire's OpenAI Responses API. The routed run sends the same prompt through
Squilla's OpenAI Chat endpoint with automatic tier and reasoning selection.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
OUT_FILE = Path(__file__).with_name("e2e_router_vs_direct_nolimit.json")
SYSTEM_PROMPT = (
    "You are a senior software engineer. Answer accurately and concisely. "
    "Give the requested result directly; keep the final answer under 250 Chinese "
    "characters or 180 English words."
)
PRICE_UNITS = {
    "c0": {"cache_hit": 1, "cache_miss": 10, "output": 40},
    "c1": {"cache_hit": 4, "cache_miss": 40, "output": 160},
    "c2": {"cache_hit": 10, "cache_miss": 100, "output": 400},
    "c3": {"cache_hit": 20, "cache_miss": 200, "output": 800},
}
MODEL_TIER = {
    "Qwen3.8-27B": "c0",
    "Qwen3.5-397B-A17B": "c1",
    "GLM-5.3-Flash": "c2",
    "DeepSeek-V4.1-Flash": "c3",
}


TASKS = [
    ("t0_translate", "把 hello world 翻译成中文。"),
    ("t1_palindrome", "写一个 Python 函数判断字符串是否为回文。"),
    ("r1_json", "帮我写一个简单的 Python 脚本读取 JSON 文件并打印字段。"),
    ("r1_upload", "设计一个支持断点续传的文件上传接口。"),
    ("r2_502", "排查 FastAPI 接口偶发 502 的常见原因，并给出诊断步骤。"),
    ("r2_retry", "写一个带重试、超时和指数退避的 HTTP 客户端。"),
    (
        "r3_order",
        "设计一个分布式订单系统，保证幂等、消息可靠投递和故障恢复，并给出核心代码。",
    ),
    (
        "r3_sql",
        "帮我优化一条慢 SQL，分析索引、执行计划和改写方案。",
    ),
]


def read_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def run_router(prompt: str, env: dict[str, str]) -> dict[str, Any]:
    payload = {
        "model": "router",
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": "Bearer " + env["ROUTER_API_KEY"],
        "Content-Type": "application/json",
    }
    started = time.perf_counter()
    first_content: float | None = None
    usage: dict[str, Any] = {}
    text = ""
    with httpx.Client(trust_env=False, timeout=httpx.Timeout(180, connect=10)) as client:
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
                choice = (event.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                if content or delta.get("reasoning") or delta.get("reasoning_content"):
                    if first_content is None:
                        first_content = time.perf_counter() - started
                    text += content
                if event.get("usage"):
                    usage = event["usage"]
            return {
                "status": response.status_code,
                "tier": response.headers.get("x-router-tier"),
                "model": response.headers.get("x-router-model"),
                "ttft_s": round(first_content or 0, 3),
                "latency_s": round(time.perf_counter() - started, 3),
                "usage": usage,
                "output_chars": len(text),
            }


def run_direct(prompt: str, env: dict[str, str]) -> dict[str, Any]:
    payload = {
        "model": "DeepSeek-V4.1-Flash",
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
    }
    headers = {
        "Authorization": "Bearer " + env["STARFIRE_API_KEY"],
        "Content-Type": "application/json",
    }
    url = env["STARFIRE_BASE_URL"].rstrip("/") + "/responses"
    started = time.perf_counter()
    first_content: float | None = None
    usage: dict[str, Any] = {}
    text = ""
    with httpx.Client(trust_env=False, timeout=httpx.Timeout(180, connect=10)) as client:
        with client.stream("POST", url, headers=headers, json=payload) as response:
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
                event_type = event.get("type")
                if event_type in (
                    "response.reasoning_text.delta",
                    "response.output_text.delta",
                ):
                    if first_content is None:
                        first_content = time.perf_counter() - started
                    text += str(event.get("delta") or "")
                if event_type == "response.completed":
                    usage = (event.get("response") or {}).get("usage") or {}
            return {
                "status": response.status_code,
                "tier": "direct",
                "model": "DeepSeek-V4.1-Flash",
                "ttft_s": round(first_content or 0, 3),
                "latency_s": round(time.perf_counter() - started, 3),
                "usage": usage,
                "output_chars": len(text),
            }


def with_retries(fn, prompt: str, env: dict[str, str]) -> dict[str, Any]:
    retries = 0
    while True:
        result = fn(prompt, env)
        result["retries"] = retries
        if result["status"] != 429 or retries >= 2:
            return result
        retries += 1
        time.sleep(5)


def token_parts(run: dict[str, Any]) -> tuple[int, int, int, int]:
    usage = run.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(
        usage.get("completion_tokens") or usage.get("output_tokens") or 0
    )
    details = (
        usage.get("prompt_tokens_details")
        or usage.get("input_tokens_details")
        or {}
    )
    cache_hit = int(details.get("cached_tokens") or 0)
    return input_tokens, cache_hit, max(0, input_tokens - cache_hit), output_tokens


def cost_units(run: dict[str, Any], tier: str) -> int:
    _, cached, cache_miss, output = token_parts(run)
    price = PRICE_UNITS[tier]
    return (
        cached * price["cache_hit"]
        + cache_miss * price["cache_miss"]
        + output * price["output"]
    )


def main() -> None:
    env = read_env()
    results = []
    for task_id, prompt in TASKS:
        print(f"[task] {task_id}", flush=True)
        routed = with_retries(run_router, prompt, env)
        direct = with_retries(run_direct, prompt, env)
        routed_tier = MODEL_TIER.get(routed.get("model") or "", "c3")
        routed["cost_tier"] = routed_tier
        routed["cost_units"] = cost_units(routed, routed_tier)
        direct["cost_tier"] = "c3"
        direct["cost_units"] = cost_units(direct, "c3")
        results.append(
            {
                "task": task_id,
                "prompt": prompt,
                "routed": routed,
                "direct": direct,
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    def token_total(run: dict[str, Any]) -> int:
        return int((run.get("usage") or {}).get("total_tokens") or 0)

    routed_tokens = sum(token_total(item["routed"]) for item in results)
    direct_tokens = sum(token_total(item["direct"]) for item in results)
    routed_latency = sum(item["routed"]["latency_s"] for item in results)
    direct_latency = sum(item["direct"]["latency_s"] for item in results)
    routed_cost = sum(item["routed"]["cost_units"] for item in results)
    direct_cost = sum(item["direct"]["cost_units"] for item in results)
    summary = {
        "task_count": len(results),
        "routed_tokens": routed_tokens,
        "direct_tokens": direct_tokens,
        "token_saving_pct": (
            round((1 - routed_tokens / direct_tokens) * 100, 2)
            if direct_tokens
            else None
        ),
        "routed_total_latency_s": round(routed_latency, 3),
        "direct_total_latency_s": round(direct_latency, 3),
        "latency_reduction_pct": (
            round((1 - routed_latency / direct_latency) * 100, 2)
            if direct_latency
            else None
        ),
        "routed_cost_units": routed_cost,
        "direct_c3_cost_units": direct_cost,
        "cost_saving_pct": (
            round((1 - routed_cost / direct_cost) * 100, 2)
            if direct_cost
            else None
        ),
        "cost_ratio": (
            round(direct_cost / routed_cost, 2) if routed_cost else None
        ),
    }
    OUT_FILE.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"saved: {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
