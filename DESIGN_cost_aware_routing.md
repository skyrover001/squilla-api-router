# Cost-Aware Tier Selection

The classifier tier is a quality floor, not always the final execution tier.
For `new_task` and confirmed `task_changed`, the router may choose a stronger
tier when its lower expected output length makes total cost lower.

## Inputs

- full request input-token estimate (read-only)
- classifier tier
- price weights
- observed average output tokens per tier
- observed cache-hit ratio per tier
- task-class-specific output profiles (`code`, `debug`, `design`, `explain`, ...)

## Decision

```text
candidate tiers = configured tiers >= classifier tier
expected cost =
  cached input × cache-hit price
  + uncached input × cache-miss price
  + expected output × output price
select minimum expected cost
```

When the caller supplies `max_tokens` or `max_output_tokens`, the expected
output is capped by that value. Without an explicit cap, measured tier output
profiles are used.

`continue` keeps the active tier for cache affinity. `redo` remains a strict
one-tier upgrade and is not changed by cost ranking.

## Configuration

```env
ROUTER_COST_AWARE_ENABLED=1
ROUTER_MIN_PROFILE_OBSERVATIONS=3
ROUTER_PRICE_C0_CACHE_HIT=1
ROUTER_PRICE_C0_CACHE_MISS=10
ROUTER_PRICE_C0_OUTPUT=40
ROUTER_LARGE_CONTEXT_R2_TOKENS=25000
ROUTER_LARGE_CONTEXT_R3_TOKENS=80000
```

The ledger stores real input/cache/output/reasoning tokens and latency. Tier
profiles are learned from accepted calls without modifying request content.
Task-class profiles take precedence once they have enough observations; global
tier profiles are the fallback.

`X-Squilla-Cost-Policy: classifier` disables cost re-ranking for one request
without changing its prompt or tool definitions.

For every turn, the router estimates the complete provider-visible request:
messages, system/developer material, tools, history, and tool results. A
Continue request that crosses a context floor is promoted to c2/c3 even if the
task intent is unchanged.
