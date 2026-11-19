"""Summarize real OpenCode router-vs-direct A/B runs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = {
    "router_c0": {
        "path": ROOT / "opencode_router_task" / "events.jsonl",
        "tier": "c0",
        "elapsed_s": 25.32,
    },
    "direct_c3": {
        "path": ROOT / "opencode_direct_task" / "events.jsonl",
        "tier": "c3",
        "elapsed_s": 23.65,
    },
}
PRICES = {"c0": (1, 10, 40), "c3": (20, 200, 800)}


def summarize(path: Path, tier: str, elapsed_s: float) -> dict:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    steps = [row["part"] for row in rows if row.get("type") == "step_finish"]
    tools = [row for row in rows if row.get("type") == "tool_use"]
    input_tokens = sum(step["tokens"].get("input", 0) for step in steps)
    output_tokens = sum(step["tokens"].get("output", 0) for step in steps)
    reasoning_tokens = sum(step["tokens"].get("reasoning", 0) for step in steps)
    cache_read = sum(step["tokens"].get("cache", {}).get("read", 0) for step in steps)
    cache_write = sum(step["tokens"].get("cache", {}).get("write", 0) for step in steps)
    hit, miss, output_price = PRICES[tier]
    cost_units = (
        cache_read * hit
        + input_tokens * miss
        + output_tokens * output_price
    )
    return {
        "steps": len(steps),
        "tool_calls": len(tools),
        "elapsed_s": elapsed_s,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "total_tokens": input_tokens + output_tokens + reasoning_tokens + cache_read,
        "cost_units": cost_units,
        "tier": tier,
    }


def main() -> None:
    result = {}
    for name, config in RUNS.items():
        result[name] = summarize(
            config["path"], config["tier"], config["elapsed_s"]
        )
    routed = result["router_c0"]
    direct = result["direct_c3"]
    result["comparison"] = {
        "cost_reduction_pct": round(
            (1 - routed["cost_units"] / direct["cost_units"]) * 100, 2
        ),
        "latency_change_pct": round(
            (1 - routed["elapsed_s"] / direct["elapsed_s"]) * 100, 2
        ),
        "tool_call_delta": direct["tool_calls"] - routed["tool_calls"],
        "cache_read_delta": routed["cache_read_tokens"] - direct["cache_read_tokens"],
    }
    output = ROOT / "opencode_ab_summary.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
