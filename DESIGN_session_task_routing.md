# Transparent Session/Task Routing

This router is not an agent harness. It never rewrites prompts, messages, tool
definitions, or user-provided output limits.

## Request Metadata

Optional headers let a harness provide stable routing context:

```text
X-Squilla-Session-Id
X-Squilla-Task-Id
X-Squilla-Intent: new_task|continue|redo|task_changed
X-Squilla-Quality-Feedback: validation_failed|tool_failed|no_progress
```

If no session header is supplied, the router derives a stable fingerprint from
the client identity, system material, tool schema, and first user message.

## Intent State Machine

| Intent | Default action |
|---|---|
| `new_task` | classify and select a tier |
| `continue` | retain the task's active provider/model/tier |
| `redo` | advance exactly one tier and never fall back below it |
| `task_changed` | create a new task ID and classify again |

Non-success quality feedback is normalized to `redo`. The feedback value is
routing metadata only; prompts and tool definitions remain unchanged.

Malformed provider tool calls are detected after completion and stored as a
pending `redo` for the same session. The next request consumes that signal.
Consecutive identical tool-call signatures are counted per session; reaching
`ROUTER_REPEAT_TOOL_CALL_THRESHOLD` also produces a pending `redo`.

Automatic detection uses:

- explicit headers
- system/tool schema changes
- tool-result continuation
- redo/new-task phrases
- task-anchor similarity with a configurable threshold

## Configuration

```env
ROUTER_SESSION_STICKY=1
ROUTER_TASK_CHANGE_SIMILARITY=0.15
```

State is stored in SQLite at `.squilla_router_state.db` in the running process
working directory unless `ROUTER_STATE_DB` is set.
State rows carry a monotonic revision for optimistic concurrency and a TTL.
Stale writers are rejected rather than overwriting a newer task state.

## Observability

Responses include:

```text
X-Squilla-Session-Id
X-Squilla-Task-Id
X-Squilla-Intent
X-Squilla-Intent-Source
X-Squilla-Intent-Reason
X-Squilla-Sticky
X-Squilla-Task-Similarity
```

Inspect or delete one session state:

```text
GET    /admin/routing-state?session_id=...
DELETE /admin/routing-state?session_id=...
```

Inspect actual provider usage and price-weighted cost:

```text
GET /admin/routing-usage?session_id=...
```

The usage response includes successful provider calls and failed attempts. Failed
attempts retain status, error type, latency, and circuit-breaker cooldown.

The ledger records input, cache-hit, output, reasoning tokens, latency, and
price-weighted cost for each accepted provider call.

