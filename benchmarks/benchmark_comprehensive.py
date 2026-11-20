"""Comprehensive no-limit comparison with equal reasoning effort.

For every prompt:
  1. Run through Squilla and record the selected tier + effective effort.
  2. Run directly against c3 with that same effort.
  3. Compare latency and price-weighted usage using the user's price ratios.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = Path(__file__).with_name(
    "e2e_comprehensive_same_reasoning_nolimit.json"
)
SYSTEM_PROMPT = (
    "You are a senior software engineer. Answer accurately and concisely. "
    "Give the requested result directly; keep the final answer under 250 Chinese "
    "characters or 180 English words."
)
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
TASKS = [
    ("r0_translate", "basic", "把“我们的服务将在周五晚上进行维护”翻译成英文。"),
    ("r0_math", "basic", "一个订单原价 128 元，打七五折后再加 6% 税，最终多少钱？"),
    ("r0_oauth", "basic", "审查 OAuth2 回调接口的安全性，覆盖 state、PKCE、重放和开放重定向。"),
    ("r0_python_error", "debug", "分析错误 TypeError: object is not subscriptable，给出排查和修复步骤。"),
    ("r1_read_json", "code", "写一个 Python 脚本读取 JSON 文件并打印指定字段。"),
    ("r1_unit_test", "code", "为 add(a,b) 写 pytest 单元测试，覆盖正常值和异常值。"),
    ("r2_explain", "explain", "用三个要点解释 JWT 的组成和常见安全风险。"),
    ("r2_502", "debug", "排查 FastAPI 接口偶发 502 的常见原因，并给出诊断顺序。"),
    ("r2_retry", "code", "写一个带重试、超时和指数退避的 HTTP 客户端。"),
    ("r3_order", "design", "设计一个分布式订单系统，保证幂等、消息可靠投递和故障恢复。"),
    ("r3_cache", "design", "设计一个支持缓存穿透、击穿和雪崩防护的 Redis 层。"),
    ("r3_incident", "design", "线上服务错误率突然升到 30%，给出完整的排障和回滚方案。"),
]


def read_env() -> dict[str, str]:
    env = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def usage_parts(usage: dict[str, Any]) -> tuple[int, int, int]:
    inp = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    details = (
        usage.get("prompt_tokens_details")
        or usage.get("input_tokens_details")
        or {}
    )
    cached = int(details.get("cached_tokens") or 0)
    return inp, min(cached, inp), out


def cost_units(usage: dict[str, Any], tier: str) -> int:
    inp, cached, out = usage_parts(usage)
    hit_price, miss_price, out_price = PRICES[tier]
    return cached * hit_price + (inp - cached) * miss_price + out * out_price


def run_router(prompt: str, env: dict[str, str]) -> dict[str, Any]:
    headers = {
        "Authorization": "Bearer " + env["ROUTER_API_KEY"],
        "Content-Type": "application/json",
    }
    payload = {
        "model": "router",
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    started = time.perf_counter()
    first = None
    usage = {}
    chars = 0
    with httpx.Client(trust_env=False, timeout=httpx.Timeout(240, connect=10)) as c:
        with c.stream(
            "POST",
            "http://127.0.0.1:8021/v1/chat/completions",
            headers=headers,
            json=payload,
        ) as r:
            for line in r.iter_lines():
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
                    if first is None:
                        first = time.perf_counter() - started
                    chars += len(content)
                if event.get("usage"):
                    usage = event["usage"]
            reasoning_header = r.headers.get("x-router-reasoning") or ""
            return {
                "status": r.status_code,
                "tier": MODEL_TIER.get(r.headers.get("x-router-model") or "", "c3"),
                "route_header": r.headers.get("x-router-tier"),
                "model": r.headers.get("x-router-model"),
                "reasoning": reasoning_header,
                "effort": reasoning_header.split(":", 1)[-1] or None,
                "ttft_s": round(first or 0, 3),
                "latency_s": round(time.perf_counter() - started, 3),
                "usage": usage,
                "output_chars": chars,
            }


def run_direct(
    prompt: str, effort: str | None, env: dict[str, str]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "DeepSeek-V4.1-Flash",
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
    }
    if effort and effort != "-":
        payload["reasoning"] = {"effort": effort}
    headers = {
        "Authorization": "Bearer " + env["secondary_API_KEY"],
        "Content-Type": "application/json",
    }
    url = env["secondary_BASE_URL"].rstrip("/") + "/responses"
    started = time.perf_counter()
    first = None
    usage = {}
    chars = 0
    with httpx.Client(trust_env=False, timeout=httpx.Timeout(240, connect=10)) as c:
        with c.stream("POST", url, headers=headers, json=payload) as r:
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    continue
                try:
                    event = json.loads(raw)
                except Exception:
                    continue
                if event.get("type") in (
                    "response.reasoning_text.delta",
                    "response.output_text.delta",
                ):
                    if first is None:
                        first = time.perf_counter() - started
                    chars += len(str(event.get("delta") or ""))
                if event.get("type") == "response.completed":
                    usage = (event.get("response") or {}).get("usage") or {}
            return {
                "status": r.status_code,
                "model": "DeepSeek-V4.1-Flash",
                "effort": effort,
                "ttft_s": round(first or 0, 3),
                "latency_s": round(time.perf_counter() - started, 3),
                "usage": usage,
                "output_chars": chars,
            }


def with_retry(fn, *args) -> dict[str, Any]:
    retries = 0
    while True:
        result = fn(*args)
        result["retries"] = retries
        if result["status"] != 429 or retries >= 2:
            return result
        retries += 1
        time.sleep(5)


def main() -> None:
    env = read_env()
    results = []
    for task_id, group, prompt in TASKS:
        print(f"[task] {task_id}", flush=True)
        routed = with_retry(run_router, prompt, env)
        direct = with_retry(run_direct, prompt, routed.get("effort"), env)
        routed["cost_units"] = cost_units(routed["usage"], routed["tier"])
        direct["cost_units"] = cost_units(direct["usage"], "c3")
        item = {
            "task": task_id,
            "group": group,
            "prompt": prompt,
            "routed": routed,
            "direct_c3_same_reasoning": direct,
        }
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    routed_tokens = sum(
        int((x["routed"]["usage"] or {}).get("total_tokens") or 0)
        for x in results
    )
    direct_tokens = sum(
        int((x["direct_c3_same_reasoning"]["usage"] or {}).get("total_tokens") or 0)
        for x in results
    )
    routed_cost = sum(x["routed"]["cost_units"] for x in results)
    direct_cost = sum(x["direct_c3_same_reasoning"]["cost_units"] for x in results)
    routed_latency = sum(x["routed"]["latency_s"] for x in results)
    direct_latency = sum(
        x["direct_c3_same_reasoning"]["latency_s"] for x in results
    )
    summary = {
        "task_count": len(results),
        "routed_tokens": routed_tokens,
        "direct_tokens": direct_tokens,
        "token_reduction_pct": round((1 - routed_tokens / direct_tokens) * 100, 2),
        "routed_cost_units": routed_cost,
        "direct_cost_units": direct_cost,
        "cost_reduction_pct": round((1 - routed_cost / direct_cost) * 100, 2),
        "cost_ratio_direct_over_routed": round(direct_cost / routed_cost, 2),
        "routed_total_latency_s": round(routed_latency, 3),
        "direct_total_latency_s": round(direct_latency, 3),
        "latency_reduction_pct": round(
            (1 - routed_latency / direct_latency) * 100, 2
        ),
    }
    OUT_FILE.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
