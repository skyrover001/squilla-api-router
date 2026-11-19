"""Squilla API Router - one HTTP service.



Receives OpenAI-compatible chat completion requests, classifies the user

message using the V4 Phase 3 ML classifier (same as OpenSquilla), routes

to the corresponding backend model, and returns a standard OpenAI response.



One Python process. No Docker, no LiteLLM, no external classifier service.

"""



from __future__ import annotations



import os
import logging
import logging.handlers
import asyncio
import json
import re
import secrets
import time

from pathlib import Path

from dataclasses import asdict, dataclass
from typing import Any, Callable
from uuid import uuid4



import httpx

import yaml

from dotenv import load_dotenv

from fastapi import FastAPI, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from fastapi.responses import JSONResponse, StreamingResponse



from squilla_api_router.conversation_context import ConversationContext, RouteDecisionContext
from squilla_api_router.backend_health import CircuitBreaker
from squilla_api_router.cost_policy import (
    choose_tier as choose_cost_aware_tier,
    classify_task,
    estimate_capability_input_tokens,
    estimate_input_tokens,
    large_context_floor_tier as policy_large_context_floor_tier,
)
from squilla_api_router.quality_signals import (
    StreamToolCallTracker,
    response_quality_failure,
    response_tool_signature,
)
from squilla_api_router.usage_estimation import (
    StreamOutputTokenEstimator,
    estimate_response_output_tokens,
)

from squilla_api_router.final_policy import apply_final_policy
from squilla_api_router.routing_state import (
    RequestIdentity,
    RouteState,
    RoutingStateStore,
    StateConflict,
    decide_intent,
    resolve_request_identity,
)



load_dotenv()



app = FastAPI(title="Squilla API Router", version="0.1.0", docs_url=None)

logger = logging.getLogger("squilla.router")


def _configure_router_logging(
    *,
    force: bool = False,
) -> logging.handlers.RotatingFileHandler | None:
    """Configure a size-rotated application log file.

    ``_run_server.py`` can install the same handler as part of uvicorn's logging
    configuration.  ``force`` is used by tests to replace that handler.
    """
    if os.environ.get("ROUTER_LOG_ENABLED", "1") != "1":
        return None
    existing = next(
        (
            handler
            for handler in logger.handlers
            if getattr(handler, "_squilla_router_file", False)
        ),
        None,
    )
    if existing is not None and not force:
        return existing
    if force:
        for handler in list(logger.handlers):
            if getattr(handler, "_squilla_router_file", False):
                logger.removeHandler(handler)
                handler.close()

    log_file = Path(
        os.environ.get(
            "ROUTER_LOG_FILE",
            str(Path.cwd() / "squilla_router.log"),
        )
    )
    max_bytes = max(1024, int(os.environ.get("ROUTER_LOG_MAX_BYTES", "20971520")))
    backup_count = max(1, int(os.environ.get("ROUTER_LOG_BACKUP_COUNT", "5")))
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("file logging unavailable: %s", exc)
        return None
    handler.setFormatter(
        logging.Formatter(
            "timestamp=%(asctime)s level=%(levelname)s logger=%(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    handler._squilla_router_file = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return handler


if os.environ.get("ROUTER_EXTERNAL_LOGGING") != "1":
    _configure_router_logging()



ROUTER_API_KEY = os.environ.get("ROUTER_API_KEY", "")
ROUTER_TENANT_API_KEYS: dict[str, str] = {}
# Upstream (backend) call timeout, in seconds.
# Long reasoning generations can easily exceed two minutes, so default is generous.
# For streaming, bound the time spent waiting for the first content event so a
# stalled backend can fail over. After content starts, use an idle timeout
# rather than an unbounded read.
BACKEND_TIMEOUT = float(os.environ.get("BACKEND_TIMEOUT", "300"))
BACKEND_CONNECT_TIMEOUT = float(os.environ.get("BACKEND_CONNECT_TIMEOUT", "30"))
BACKEND_STREAM_START_TIMEOUT = float(
    os.environ.get("BACKEND_STREAM_START_TIMEOUT", "30")
)
BACKEND_STREAM_FAST_START_TIMEOUT = float(
    os.environ.get("BACKEND_STREAM_FAST_START_TIMEOUT", "8")
)
BACKEND_STREAM_IDLE_TIMEOUT = float(
    os.environ.get("BACKEND_STREAM_IDLE_TIMEOUT", "300")
)

# ---------------------------------------------------------------------------
# Reasoning / thinking-strength routing.
#
# The ML classifier already produces a thinking mode (T0-T3).  This block maps
# that mode onto each backend model's *declared* reasoning capability
# (C<N>_REASONING_*), then injects the result in the agent's native format so
# llm-rosetta can translate it to whatever the backend actually speaks.
#
# Disabled by default: with ROUTER_REASONING_ENABLED unset every request
# behaves exactly as before (thinking_mode stays metadata only).
# ---------------------------------------------------------------------------
_EFFORT_LADDER = ("minimal", "low", "medium", "high", "xhigh", "max")
_EFFORT_RANK = {v: i for i, v in enumerate(_EFFORT_LADDER)}
_DEFAULT_THINKING_EFFORT = {
    "T0": "minimal",
    "T1": "low",
    "T2": "medium",
    "T3": "high",
}
_THINKING_MODE_ORDER = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}

ROUTER_REASONING_ENABLED = os.environ.get("ROUTER_REASONING_ENABLED", "0") == "1"
ROUTER_SESSION_STICKY = os.environ.get("ROUTER_SESSION_STICKY", "1") == "1"
ROUTER_TASK_CHANGE_SIMILARITY = float(
    os.environ.get("ROUTER_TASK_CHANGE_SIMILARITY", "0.15")
)
ROUTER_COST_AWARE_ENABLED = os.environ.get("ROUTER_COST_AWARE_ENABLED", "1") == "1"
ROUTER_MIN_PROFILE_OBSERVATIONS = int(
    os.environ.get("ROUTER_MIN_PROFILE_OBSERVATIONS", "3")
)
ROUTER_CIRCUIT_BREAKER_ENABLED = (
    os.environ.get("ROUTER_CIRCUIT_BREAKER_ENABLED", "1") == "1"
)
ROUTER_FAILURE_COOLDOWN_SECONDS = float(
    os.environ.get("ROUTER_FAILURE_COOLDOWN_SECONDS", "60")
)
ROUTER_RATE_LIMIT_COOLDOWN_SECONDS = float(
    os.environ.get("ROUTER_RATE_LIMIT_COOLDOWN_SECONDS", "15")
)
_ROUTING_STATE_STORE: RoutingStateStore | None = None
_CIRCUIT_BREAKER: CircuitBreaker | None = None

ROUTER_REASONING_PRIORITY = (
    os.environ.get("ROUTER_REASONING_PRIORITY", "agent").strip().lower() or "agent"
)


# Rosetta conversion gateway (optional). If set, classified requests are
# forwarded to rosetta (model rewritten to `<provider>/<model>`) which converts
# format to each backend's native API.
ROSETTA_URL = os.environ.get("ROSETTA_URL", "")
ROSETTA_API_KEY = os.environ.get("ROSETTA_API_KEY", "sk-proxy-test")

_security = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class RouterPrincipal:
    tenant_id: str
    is_admin: bool
    auth_kind: str


def _secure_key_equal(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return secrets.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def _load_tenant_api_keys(raw: str | None = None) -> dict[str, str]:
    value = os.environ.get("ROUTER_TENANT_API_KEYS", "") if raw is None else raw
    if not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        logger.warning("ROUTER_TENANT_API_KEYS is not valid JSON; ignored")
        return {}
    if not isinstance(parsed, dict):
        logger.warning("ROUTER_TENANT_API_KEYS must be a JSON object; ignored")
        return {}
    tenants: dict[str, str] = {}
    seen_keys: set[bytes] = set()
    for tenant, api_key in parsed.items():
        tenant_id = str(tenant).strip()
        key = str(api_key).strip()
        if not tenant_id or not key:
            continue
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", tenant_id) is None:
            logger.warning("invalid tenant id ignored")
            continue
        if ROUTER_API_KEY and _secure_key_equal(key, ROUTER_API_KEY):
            logger.warning(
                "tenant key for tenant=%s duplicates the global key; ignored",
                tenant_id,
            )
            continue
        key_bytes = key.encode("utf-8")
        if key_bytes in seen_keys:
            logger.warning("duplicate tenant API key ignored for tenant=%s", tenant_id)
            continue
        seen_keys.add(key_bytes)
        tenants[tenant_id] = key
    return tenants


ROUTER_TENANT_API_KEYS = _load_tenant_api_keys()


def _authenticate(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None,
) -> RouterPrincipal:
    """Verify the caller presents the configured router API key (if set).

    Accepts either standard ``Authorization: Bearer <key>`` or the
    Anthropic-style ``x-api-key: <key>`` header, so Claude Code / Anthropic
    SDK can connect to /v1/messages without custom auth plumbing."""
    requested_tenant = (
        request.headers.get("x-squilla-tenant-id")
        or request.headers.get("x-tenant-id")
        or request.headers.get("x-workspace-id")
        or ""
    ).strip()
    if not ROUTER_API_KEY:
        # No global key and no tenant keys configured -> open mode.
        if not ROUTER_TENANT_API_KEYS:
            principal = RouterPrincipal(
                tenant_id=requested_tenant,
                is_admin=True,
                auth_kind="open",
            )
            request.state.router_principal = principal
            return principal
    provided = ""
    if credentials is not None:
        provided = credentials.credentials or ""
    if not provided:
        # Anthropic-style clients (Claude Code / Anthropic SDK) send x-api-key.
        provided = request.headers.get("x-api-key", "")
    if not provided:
        from fastapi import HTTPException, status as http_status
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if ROUTER_API_KEY and _secure_key_equal(provided, ROUTER_API_KEY):
        principal = RouterPrincipal(
            tenant_id=requested_tenant,
            is_admin=True,
            auth_kind="global",
        )
        request.state.router_principal = principal
        return principal
    for tenant_id, tenant_key in ROUTER_TENANT_API_KEYS.items():
        if not _secure_key_equal(provided, tenant_key):
            continue
        if requested_tenant and requested_tenant != tenant_id:
            from fastapi import HTTPException, status as http_status

            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail="Tenant API key does not match x-squilla-tenant-id",
            )
        principal = RouterPrincipal(
            tenant_id=tenant_id,
            is_admin=False,
            auth_kind="tenant",
        )
        request.state.router_principal = principal
        return principal
    from fastapi import HTTPException, status as http_status

    raise HTTPException(
        status_code=http_status.HTTP_401_UNAUTHORIZED,
        detail="Invalid router API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _verify_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> RouterPrincipal:
    return _authenticate(request, credentials)


def _verify_admin_auth(
    principal: RouterPrincipal = Depends(_verify_auth),
) -> RouterPrincipal:
    if not principal.is_admin:
        from fastapi import HTTPException, status as http_status

        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Tenant API keys cannot access admin endpoints",
        )
    return principal



BUNDLE_DIR = Path(__file__).resolve().parent / "model_bundle"



# Vision-capable model (routed to when request contains images)
VISION_MODEL = os.environ.get("VISION_MODEL", "GLM-5.3-Flash")
VISION_BASE_URL = os.environ.get("VISION_BASE_URL", os.environ.get("BACKEND_BASE_URL", ""))
VISION_API_KEY = os.environ.get("VISION_API_KEY", os.environ.get("BACKEND_API_KEY", ""))

# Provider table: provider name -> endpoint + key.
# Tiers reference these by name, no repeated base_url/api_key per model.
# ---------------------------------------------------------------------------
# Provider / tier autodiscovery.
#
# Providers are discovered from the environment: any <PREFIX>_BASE_URL
# registers a provider named <prefix> (lowercased).  The legacy
# BACKEND_* prefix is aliased to provider name tianhe.  Each provider
# reads <PREFIX>_API_KEY and <PREFIX>_FORMAT.
#
# Tiers are discovered from C<N>_MODEL (+ optional C<N>_PROVIDER,
# C<N>_SUPPORTS_IMAGE/VIDEO) and sorted by N, producing c0..cN.
# The V4 classifier routes R0..R(n-1), mapped 1:1 to the discovered tiers.
# ---------------------------------------------------------------------------
_PROVIDER_BASE_URL_KEYS = {
    "BACKEND": "tianhe",  # legacy alias
}


def _normalize_format(fmt: str) -> str:
    fmt = (fmt or "").strip().lower()
    if fmt in ("/responses", "responses", "openai_responses"):
        return "openai_responses"
    if fmt in ("/messages", "messages", "anthropic"):
        return "anthropic"
    if fmt in ("", "chat", "/chat/completions", "openai_chat"):
        return "openai_chat"
    return fmt


def _discover_providers() -> dict[str, dict[str, str]]:
    prov: dict[str, dict[str, str]] = {}
    for key, val in os.environ.items():
        up = key.upper()
        if not up.endswith("_BASE_URL") or not val.strip():
            continue
        prefix = up[:-len("_BASE_URL")]
        if prefix in ("VISION", "ROUTER"):
            continue
        name = _PROVIDER_BASE_URL_KEYS.get(prefix, prefix.lower())
        fmt = os.environ.get(f"{prefix}_FORMAT", "")
        # If no explicit FORMAT, infer from the endpoint path.
        if not fmt.strip():
            if "/anthropic" in val or val.endswith("/messages"):
                fmt = "anthropic"
        fmt = _normalize_format(fmt)
        prov[name] = {
            "base_url": val.rstrip("/"),
            "api_key": os.environ.get(f"{prefix}_API_KEY", ""),
            "format": fmt,
        }
    return prov


def _parse_reasoning_capability(prefix: str) -> dict | None:
    """Read `<prefix>_REASONING_*` from the environment.

    Returns None when `<prefix>_REASONING_MODES` is absent/empty, which means
    "this model does not support reasoning control" -- the router will then
    never send thinking parameters to it.
    """
    modes_raw = os.environ.get(f"{prefix}_REASONING_MODES", "").strip()
    if not modes_raw:
        return None
    modes = [m.strip().lower() for m in modes_raw.split(",") if m.strip()]
    if not modes:
        return None

    def _opt(name: str) -> str | None:
        val = os.environ.get(f"{prefix}_REASONING_{name}", "").strip().lower()
        return val or None

    ratio_raw = os.environ.get(f"{prefix}_REASONING_BUDGET_RATIO", "").strip()
    try:
        budget_ratio = float(ratio_raw) if ratio_raw else None
    except ValueError:
        budget_ratio = None

    effort_min = _opt("EFFORT_MIN")
    effort_max = _opt("EFFORT_MAX")
    if effort_min not in _EFFORT_RANK:
        effort_min = None
    if effort_max not in _EFFORT_RANK:
        effort_max = None
    if effort_min and effort_max and _EFFORT_RANK[effort_min] > _EFFORT_RANK[effort_max]:
        effort_min, effort_max = effort_max, effort_min

    efforts_raw = os.environ.get(f"{prefix}_REASONING_EFFORTS", "").strip()
    efforts = sorted(
        {
            effort.strip().lower()
            for effort in efforts_raw.split(",")
            if effort.strip().lower() in _EFFORT_RANK
        },
        key=_EFFORT_RANK.get,
    )
    if efforts:
        # An explicit allow-list also acts as the range for non-contiguous
        # capabilities such as Qwen's {low, medium, xhigh}.
        within_range = [
            effort
            for effort in efforts
            if (not effort_min or _EFFORT_RANK[effort] >= _EFFORT_RANK[effort_min])
            and (not effort_max or _EFFORT_RANK[effort] <= _EFFORT_RANK[effort_max])
        ]
        if within_range:
            efforts = within_range

    default_mode = _opt("DEFAULT_MODE")
    if default_mode not in modes:
        default_mode = "auto" if "auto" in modes else modes[0]

    return {
        "modes": modes,
        "default_mode": default_mode,
        "efforts": efforts,
        "effort_min": effort_min,
        "effort_max": effort_max,
        "budget_ratio": budget_ratio,
    }


def _discover_tiers() -> dict[str, dict]:
    tiers: dict[str, dict] = {}
    indices = set()
    for key in os.environ:
        up = key.upper()
        if up.startswith("C") and up.endswith("_MODEL"):
            num = up[1:-len("_MODEL")]
            if num.isdigit():
                indices.add(int(num))
    for n in sorted(indices):
        tier = f"c{n}"
        prefix = f"C{n}"
        context_raw = os.environ.get(f"{prefix}_CONTEXT_LENGTH", "").strip()
        try:
            context_length = int(context_raw) if context_raw else None
        except ValueError:
            context_length = None
        if context_length is not None and context_length <= 0:
            context_length = None
        tiers[tier] = {
            "provider": os.environ.get(f"{prefix}_PROVIDER", "tianhe").lower(),
            "model": os.environ.get(f"{prefix}_MODEL", ""),
            "supports_image": os.environ.get(f"{prefix}_SUPPORTS_IMAGE", "0") == "1",
            "supports_video": os.environ.get(f"{prefix}_SUPPORTS_VIDEO", "0") == "1",
            "context_length": context_length,
            "reasoning": _parse_reasoning_capability(prefix),
        }
    return tiers


def _build_route_map(tiers: dict[str, dict]) -> dict[str, str]:
    # Route class R0..R(n-1) -> c0..c(n-1), one tier per route class.
    ordered = sorted(tiers.keys(), key=lambda t: int(t[1:]))
    return {f"R{i}": tier for i, tier in enumerate(ordered)}


# ---------------------------------------------------------------------------
# Reasoning merge helpers
# ---------------------------------------------------------------------------
def _thinking_effort_map() -> dict[str, str]:
    """T0-T3 -> effort, overridable via THINKING_T<N>_EFFORT."""
    out = dict(_DEFAULT_THINKING_EFFORT)
    for mode, default in _DEFAULT_THINKING_EFFORT.items():
        val = os.environ.get(f"THINKING_{mode}_EFFORT", "").strip().lower()
        if val in _EFFORT_RANK:
            out[mode] = val
    return out


def _read_reasoning_intent(body: dict, inbound_fmt: str) -> dict:
    """Extract the agent's requested reasoning intent in IR-ish terms.

    Handles the three ingress dialects.  Unknown/absent -> {} (no intent).
    Values are NOT validated here; the tier capability clamps them later.
    """
    intent: dict[str, Any] = {}

    def _mode_from_type(val: Any) -> str | None:
        if not isinstance(val, str):
            return None
        low = val.strip().lower()
        return {
            "auto": "auto",
            "adaptive": "auto",
            "enabled": "enabled",
            "disabled": "disabled",
            "none": "disabled",
        }.get(low)

    if inbound_fmt == "anthropic":
        thinking = body.get("thinking")
        if isinstance(thinking, dict):
            mode = _mode_from_type(thinking.get("type"))
            if mode:
                intent["mode"] = mode
            budget = thinking.get("budget_tokens")
            if isinstance(budget, int) and budget > 0:
                intent["budget_tokens"] = budget
        output_config = body.get("output_config")
        if isinstance(output_config, dict):
            effort = str(output_config.get("effort") or "").strip().lower()
            if effort in _EFFORT_RANK or effort == "none":
                intent["effort"] = effort
    else:
        # OpenAI Chat / Responses
        effort = body.get("reasoning_effort")
        if isinstance(effort, str):
            val = effort.strip().lower()
            if val in _EFFORT_RANK or val == "none":
                intent["effort"] = val
        reasoning = body.get("reasoning")
        if isinstance(reasoning, dict):
            r_effort = str(reasoning.get("effort") or "").strip().lower()
            if r_effort in _EFFORT_RANK or r_effort == "none":
                intent["effort"] = r_effort
            mode = _mode_from_type(reasoning.get("type"))
            if mode:
                intent.setdefault("mode", mode)
        thinking = body.get("thinking")
        if isinstance(thinking, dict):
            mode = _mode_from_type(thinking.get("type"))
            if mode:
                intent.setdefault("mode", mode)
            budget = thinking.get("budget_tokens")
            if isinstance(budget, int) and budget > 0:
                intent.setdefault("budget_tokens", budget)

    # Normalise "none" -> mode=disabled (llm-rosetta does the same).
    if intent.get("effort") == "none":
        intent.pop("effort", None)
        intent["mode"] = "disabled"
    return intent


def _clamp_effort(effort: str, cap: dict) -> tuple[str, bool]:
    """Clamp an effort into the tier's declared range or explicit allow-list."""
    rank = _EFFORT_RANK.get(effort)
    if rank is None:
        return effort, False
    allowed = cap.get("efforts") or []
    if allowed:
        nearest = min(
            allowed,
            key=lambda candidate: (
                abs(_EFFORT_RANK[candidate] - rank),
                _EFFORT_RANK[candidate],
            ),
        )
        return nearest, nearest != effort
    lo = _EFFORT_RANK.get(cap.get("effort_min") or "", 0)
    hi = _EFFORT_RANK.get(cap.get("effort_max") or "", len(_EFFORT_LADDER) - 1)
    clamped_rank = max(lo, min(hi, rank))
    return _EFFORT_LADDER[clamped_rank], clamped_rank != rank


def _merge_reasoning(
    intent: dict,
    cap: dict | None,
    thinking_mode: str | None,
    max_tokens: int | None,
) -> tuple[dict, dict]:
    """Resolve the final reasoning config for ONE candidate tier.

    Returns (effective, meta):
      effective = {"mode"?: str, "effort"?: str, "budget_tokens"?: int}
      meta      = {"source", "requested", "effective", "clamped", "dropped_reason"}
    """
    meta: dict[str, Any] = {
        "source": "none",
        "requested": thinking_mode,
        "effective": None,
        "clamped": False,
        "dropped_reason": None,
    }

    if not ROUTER_REASONING_ENABLED:
        meta["dropped_reason"] = "global_off"
        return {}, meta

    if cap is None:
        meta["dropped_reason"] = "unsupported_model"
        return {}, meta

    effective: dict[str, Any] = {}

    use_agent = bool(intent) and ROUTER_REASONING_PRIORITY == "agent"
    if use_agent:
        effective.update(intent)
        meta["source"] = "agent"
    else:
        effort = _thinking_effort_map().get(str(thinking_mode or ""), "")
        if effort:
            effective["effort"] = effort
            meta["source"] = "router" if thinking_mode else "none"

    if not effective:
        return {}, meta

    # 1) mode filtering
    modes = cap.get("modes") or []
    mode = effective.get("mode")
    if mode and mode not in modes:
        if mode == "disabled":
            # The model cannot be switched off: send nothing at all rather
            # than an invalid "disabled" value.
            meta["dropped_reason"] = "disabled_not_allowed"
            return {}, meta
        effective["mode"] = cap.get("default_mode") or "auto"

    # 2) effort clamping
    effort = effective.get("effort")
    if effort:
        new_effort, clamped = _clamp_effort(effort, cap)
        effective["effort"] = new_effort
        if clamped:
            meta["clamped"] = True

    # 3) budget constraints
    limit = (max_tokens - 1) if isinstance(max_tokens, int) and max_tokens > 0 else None
    budget = effective.get("budget_tokens")
    if isinstance(budget, int) and budget > 0:
        if limit is not None:
            effective["budget_tokens"] = min(budget, limit)
        elif max_tokens is not None:
            effective.pop("budget_tokens", None)
    elif cap.get("budget_ratio") and limit is not None:
        derived = max(1024, int(max_tokens * float(cap["budget_ratio"])))
        effective["budget_tokens"] = min(derived, limit)

    if not effective:
        return {}, meta

    meta["effective"] = dict(effective)
    return effective, meta


def _inject_reasoning(body: dict, inbound_fmt: str, effective: dict) -> dict:
    """Write an effective reasoning config into the request in the *agent's*
    native dialect, so llm-rosetta can translate it to the backend's dialect."""
    if not effective:
        return body

    mode = effective.get("mode")
    effort = effective.get("effort")
    budget = effective.get("budget_tokens")

    if inbound_fmt == "anthropic":
        thinking = dict(body.get("thinking") or {})
        if mode:
            thinking["type"] = {"auto": "adaptive", "enabled": "enabled",
                                "disabled": "disabled"}[mode]
        if budget:
            thinking["budget_tokens"] = budget
        if thinking:
            body["thinking"] = thinking
        if effort:
            cfg = dict(body.get("output_config") or {})
            cfg["effort"] = effort
            body["output_config"] = cfg
        return body

    if inbound_fmt == "openai_responses":
        reasoning = dict(body.get("reasoning") or {})
        if effort:
            reasoning["effort"] = effort
        if mode:
            reasoning["type"] = "enabled" if mode == "auto" else mode
        if reasoning:
            body["reasoning"] = reasoning
        return body

    # OpenAI Chat
    if effort:
        body["reasoning_effort"] = effort
    elif mode == "disabled":
        body["reasoning_effort"] = "none"
    if budget:
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}
    return body


def _strip_reasoning(body: dict, inbound_fmt: str) -> dict:
    """Remove ALL reasoning fields from the request.

    Used when reasoning routing is enabled but the chosen model cannot honour
    the request (no declared capability, or it cannot be switched off).  Any
    leftover field (including one supplied by the agent itself) would be
    forwarded verbatim and could make the backend return 400.
    """
    for key in ("reasoning_effort",):
        body.pop(key, None)

    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        for key in ("effort", "type", "summary"):
            reasoning.pop(key, None)
        if reasoning:
            body["reasoning"] = reasoning
        else:
            body.pop("reasoning", None)

    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        for key in ("type", "budget_tokens"):
            thinking.pop(key, None)
        if thinking:
            body["thinking"] = thinking
        else:
            body.pop("thinking", None)

    output_config = body.get("output_config")
    if isinstance(output_config, dict):
        output_config.pop("effort", None)
        if output_config:
            body["output_config"] = output_config
        else:
            body.pop("output_config", None)
    return body


PROVIDERS = _discover_providers()
TIERS = _discover_tiers()
ROUTE_CLASS_TO_TIER = _build_route_map(TIERS)


# ---------------------------------------------------------------------------
# Config hot-reload
#
# PROVIDERS / TIERS / ROUTE_CLASS_TO_TIER and the other env-driven settings
# are built from .env at import time.  `_reload_config()` re-reads .env and
# rebuilds every table, so edits to .env take effect WITHOUT restarting the
# service.  It is triggered two ways:
#   * automatically, by a per-request mtime check (`_reload_config_if_changed`)
#   * explicitly, via `POST /admin/reload-config`
# The module globals are reassigned atomically (fresh dict objects), so an
# in-flight request keeps using the table it already resolved.
# ---------------------------------------------------------------------------
_ENV_FILE = Path(__file__).resolve().parent / ".env"
_config_mtime: float | None = None


def _reload_config() -> bool:
    """Re-read .env and rebuild the provider/tier tables.  Returns True."""
    global PROVIDERS, TIERS, ROUTE_CLASS_TO_TIER
    global ROUTER_API_KEY, ROUTER_TENANT_API_KEYS
    global BACKEND_TIMEOUT, BACKEND_CONNECT_TIMEOUT
    global BACKEND_STREAM_START_TIMEOUT, BACKEND_STREAM_FAST_START_TIMEOUT
    global BACKEND_STREAM_IDLE_TIMEOUT
    global ROUTER_REASONING_ENABLED, ROUTER_REASONING_PRIORITY
    global ROUTER_SESSION_STICKY, ROUTER_TASK_CHANGE_SIMILARITY
    global ROUTER_COST_AWARE_ENABLED, ROUTER_MIN_PROFILE_OBSERVATIONS
    global VISION_MODEL, VISION_BASE_URL, VISION_API_KEY, _config_mtime

    # override=True so edited values win over the values loaded at import.
    load_dotenv(_ENV_FILE, override=True)

    PROVIDERS = _discover_providers()
    TIERS = _discover_tiers()
    ROUTE_CLASS_TO_TIER = _build_route_map(TIERS)

    ROUTER_API_KEY = os.environ.get("ROUTER_API_KEY", "")
    ROUTER_TENANT_API_KEYS = _load_tenant_api_keys()
    BACKEND_TIMEOUT = float(os.environ.get("BACKEND_TIMEOUT", "300"))
    BACKEND_CONNECT_TIMEOUT = float(os.environ.get("BACKEND_CONNECT_TIMEOUT", "30"))
    BACKEND_STREAM_START_TIMEOUT = float(
        os.environ.get("BACKEND_STREAM_START_TIMEOUT", "30")
    )
    BACKEND_STREAM_FAST_START_TIMEOUT = float(
        os.environ.get("BACKEND_STREAM_FAST_START_TIMEOUT", "8")
    )
    BACKEND_STREAM_IDLE_TIMEOUT = float(
        os.environ.get("BACKEND_STREAM_IDLE_TIMEOUT", "300")
    )
    ROUTER_REASONING_ENABLED = os.environ.get("ROUTER_REASONING_ENABLED", "0") == "1"
    ROUTER_REASONING_PRIORITY = (
        os.environ.get("ROUTER_REASONING_PRIORITY", "agent").strip().lower() or "agent"
    )
    ROUTER_SESSION_STICKY = os.environ.get("ROUTER_SESSION_STICKY", "1") == "1"
    ROUTER_TASK_CHANGE_SIMILARITY = float(
        os.environ.get("ROUTER_TASK_CHANGE_SIMILARITY", "0.15")
    )
    ROUTER_COST_AWARE_ENABLED = (
        os.environ.get("ROUTER_COST_AWARE_ENABLED", "1") == "1"
    )
    ROUTER_MIN_PROFILE_OBSERVATIONS = int(
        os.environ.get("ROUTER_MIN_PROFILE_OBSERVATIONS", "3")
    )
    VISION_MODEL = os.environ.get("VISION_MODEL", "GLM-5.3-Flash")
    VISION_BASE_URL = os.environ.get("VISION_BASE_URL", os.environ.get("BACKEND_BASE_URL", ""))
    VISION_API_KEY = os.environ.get("VISION_API_KEY", os.environ.get("BACKEND_API_KEY", ""))

    try:
        _config_mtime = _ENV_FILE.stat().st_mtime
    except Exception:
        _config_mtime = None
    logger.info("config reloaded: providers=%s tiers=%s",
                list(PROVIDERS.keys()), list(TIERS.keys()))
    return True


def _reload_config_if_changed() -> bool:
    """Reload only when .env's mtime changed (cheap check per request)."""
    global _config_mtime
    try:
        mtime = _ENV_FILE.stat().st_mtime
    except Exception:
        return False
    if _config_mtime is None:
        _config_mtime = mtime
        return False
    if mtime != _config_mtime:
        return _reload_config()
    return False


@app.middleware("http")
async def _config_hot_reload(request: Request, call_next):
    """Pick up .env edits before handling each request (no restart needed)."""
    _reload_config_if_changed()
    return await call_next(request)


def _provider(provider: str) -> dict[str, str]:
    """Resolve a provider entry to {'model','api_key','base_url'}."""
    p = PROVIDERS.get(provider, {})
    return {
        "model": "",
        "api_key": p.get("api_key", ""),
        "base_url": p.get("base_url", ""),
    }




def _extract_user_text(messages: list[dict[str, Any]]) -> str:

    """Return the last user message text content."""

    for msg in reversed(messages):

        if msg.get("role") == "user":

            content = msg.get("content", "")

            if isinstance(content, str):

                return content

            if isinstance(content, list):

                return "\n".join(

                    part.get("text", "")

                    for part in content

                    if isinstance(part, dict) and part.get("type") == "text"

                )

    return ""





def _load_core():

    """Load the V4 Phase 3 InferenceCore from the model bundle."""

    from squilla_api_router.v4_runtime.inference.core import InferenceCore

    from squilla_api_router.v4_runtime.inference.types import InferenceRequest



    config_path = BUNDLE_DIR / "router.runtime.yaml"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    use_aux_head = bool(config.get("v4", {}).get("aux_head_inference", False))

    core = InferenceCore.from_model_dir(str(BUNDLE_DIR), config, use_aux_head=use_aux_head)

    return core, InferenceRequest





# Initialize core at startup.

_core = None

_request_type = None

try:

    _core, _request_type = _load_core()

except Exception:

    pass





def _detect_image_content(messages: list[dict[str, Any]]) -> bool:
    """Detect if request contains images (OpenAI image_url or Anthropic image format)."""
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("image_url", "image", "document"):
                return True
    return False


def _detect_video_content(messages: list[dict[str, Any]]) -> bool:
    """Detect if request contains video content."""
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("video", "video_url"):
                return True
    return False


def _tier_configured(tier: str) -> bool:
    cfg = TIERS.get(tier) or {}
    provider = PROVIDERS.get(str(cfg.get("provider") or "")) or {}
    return bool(cfg.get("model") and provider.get("base_url"))


def _matching_tiers(
    body: dict[str, Any],
    *,
    require_image: bool = False,
    require_video: bool = False,
    enforce_context: bool = True,
) -> list[str]:
    """Return configured tiers compatible with the request's declared needs.

    A missing context length means "unknown" and does not restrict the tier.
    This keeps local/proxy backends usable without inventing a hard limit.
    """
    input_tokens = estimate_capability_input_tokens(
        body,
        media_tokens=int(os.environ.get("ROUTER_MEDIA_TOKEN_ESTIMATE", "1024")),
    )
    matches: list[str] = []
    for tier in sorted(TIERS, key=lambda item: int(item[1:])):
        cfg = TIERS.get(tier) or {}
        if not _tier_configured(tier):
            continue
        if require_image and not cfg.get("supports_image"):
            continue
        if require_video and not cfg.get("supports_video"):
            continue
        context_length = cfg.get("context_length")
        if (
            enforce_context
            and isinstance(context_length, int)
            and input_tokens > context_length
        ):
            continue
        matches.append(tier)
    return matches


def _vision_valid_tiers(
    *,
    required_video: bool = False,
    body: dict[str, Any] | None = None,
) -> list[str]:
    return _matching_tiers(
        body or {},
        require_image=True,
        require_video=required_video,
    )


def _is_failure(result: dict, status: int, outbound_format: str = "openai_chat") -> bool:
    """Return True if the backend response should trigger failover.

    Output truncation (Responses ``status: incomplete`` / chat
    ``finish_reason: "length"`` / Anthropic ``stop_reason: "max_tokens"``) is
    a normal response state, not a backend failure: gateways pass it through
    with the truncation marker preserved so the caller can decide to retry
    with a larger budget."""
    if status >= 500 or status == 429 or status >= 400:
        return True
    if outbound_format == "openai_responses":
        # Responses backends return `output` (list) + `status`.
        # ``incomplete`` is a truncation state, not a failure -> pass through.
        output = result.get("output")
        if not output:
            return True
        return False
    if outbound_format == "anthropic":
        return not result.get("content")
    # openai_chat
    if "choices" not in result or not result.get("choices"):
        return True
    msg = (result.get("choices") or [{}])[0].get("message") or {}
    if not (msg.get("content") or msg.get("reasoning") or msg.get("tool_calls")):
        return True
    return False


def _stream_chunk_has_content(chunk: bytes) -> bool:
    """Return True once an SSE chunk carries model output."""
    text = chunk.decode("utf-8", errors="replace")
    for frame in text.split("\n"):
        if not frame.startswith("data: ") or "[DONE]" in frame:
            continue
        try:
            obj = json.loads(frame[6:])
        except Exception:
            continue
        delta = (obj.get("choices") or [{}])[0].get("delta") or {}
        if (
            delta.get("content")
            or delta.get("reasoning")
            or delta.get("reasoning_content")
            or delta.get("tool_calls")
        ):
            return True
        event = obj.get("type") or ""
        if event in (
            "response.output_text.delta",
            "response.reasoning_text.delta",
            "response.function_call_arguments.delta",
        ):
            return True
        if event == "content_block_delta":
            block_delta = obj.get("delta") or {}
            if block_delta.get("text") or block_delta.get("thinking"):
                return True
            if block_delta.get("partial_json") or block_delta.get("input_json_delta"):
                return True
        if event == "content_block_start":
            block = obj.get("content_block") or {}
            if block.get("text") or block.get("thinking") or block.get("tool_use"):
                return True
    return False


def _usage_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    usage = event.get("usage")
    if isinstance(usage, dict):
        return usage
    response = event.get("response")
    if isinstance(response, dict) and isinstance(response.get("usage"), dict):
        return response["usage"]
    message = event.get("message")
    if isinstance(message, dict) and isinstance(message.get("usage"), dict):
        return message["usage"]
    return None


async def _call_backend(
    body: dict,
    backend: dict[str, str],
    route_class: str,
    *,
    inbound_fmt: str = "openai_chat",
    vision: bool = False,
    video: bool = False,
    thinking_mode: str | None = None,
    fallback_backends: list[dict[str, str]] | None = None,
    fallback_route_classes: list[str] | None = None,
    usage_callback: Callable[[dict[str, Any]], None] | None = None,
    attempt_callback: Callable[[dict[str, Any]], None] | None = None,
    quality_callback: Callable[[str], None] | None = None,
    repeat_callback: Callable[[str], None] | None = None,
):
    from squilla_api_router._formats_rosetta import (
        FORMAT_TO_PATH,
        convert_request,
        convert_response,
        convert_stream,
        cd_available,
    )

    can_convert = cd_available()
    is_stream = bool(body.get("stream", False))
    request_started = time.monotonic()
    stream_start_timeout = (
        BACKEND_STREAM_FAST_START_TIMEOUT
        if thinking_mode in ("T0", "T1")
        and BACKEND_STREAM_FAST_START_TIMEOUT > 0
        else BACKEND_STREAM_START_TIMEOUT
    )

    attempts: list[str] = []

    # Agent-expressed reasoning intent (if any); resolved per candidate below
    # because each model has its own declared capability/range.
    agent_reasoning = _read_reasoning_intent(body, inbound_fmt)
    body_max_tokens = body.get("max_tokens") or body.get("max_output_tokens")
    if not isinstance(body_max_tokens, int):
        body_max_tokens = None

    candidates: list[tuple[dict[str, str], str]] = [(backend, route_class)]
    for fb, frc in zip(fallback_backends or [], fallback_route_classes or []):
        candidates.append((fb, frc))
    skipped_backends: list[str] = []
    breaker = _circuit_breaker()
    if breaker.enabled:
        healthy = [
            candidate
            for candidate in candidates
            if breaker.available(_backend_key(candidate[0]))
        ]
        if healthy and len(healthy) < len(candidates):
            skipped_backends = [
                candidate[1]
                for candidate in candidates
                if candidate not in healthy
            ]
            candidates = healthy
            logger.warning(
                "[CIRCUIT] skipping cooldown tiers=%s",
                ",".join(skipped_backends),
            )
    skipped_header = ",".join(skipped_backends)

    if is_stream:
        # The producer task owns the upstream iterator and the rosetta stream
        # converter end-to-end. Handing either async generator to Starlette's
        # response task after pre-reading it causes httpx.ReadError and drops
        # the final answer/[DONE] sentinel.
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=64)

        async def produce_stream() -> None:
            stream_attempts: list[str] = []
            for idx, (cand_backend, cand_route) in enumerate(candidates):
                stream_attempts.append(cand_route)
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {cand_backend['api_key']}",
                }
                outbound_fmt = cand_backend.get("format") or inbound_fmt
                outbound_path = FORMAT_TO_PATH.get(outbound_fmt, "/chat/completions")
                url = f"{cand_backend['base_url']}{outbound_path}"
                logger.info(
                    "[ATTEMPT] %d/%d tier=%s model=%s fmt=%s stream=True",
                    idx + 1,
                    len(candidates),
                    cand_route,
                    cand_backend.get("model", ""),
                    outbound_fmt,
                )

                outbound_body = dict(body)
                candidate_thinking = _promote_thinking_mode(
                    thinking_mode, cand_route
                )
                reasoning_eff, reasoning_meta = _merge_reasoning(
                    agent_reasoning,
                    cand_backend.get("reasoning"),
                    candidate_thinking,
                    body_max_tokens,
                )
                if reasoning_eff:
                    outbound_body = _inject_reasoning(
                        outbound_body, inbound_fmt, reasoning_eff
                    )
                elif ROUTER_REASONING_ENABLED:
                    outbound_body = _strip_reasoning(outbound_body, inbound_fmt)
                outbound_body["model"] = cand_backend["model"]

                conv_used = False
                if can_convert and inbound_fmt != outbound_fmt:
                    try:
                        outbound_body = convert_request(
                            outbound_body, inbound_fmt, outbound_fmt
                        )
                        conv_used = True
                    except Exception:
                        outbound_body = body

                if reasoning_meta.get("effective") or reasoning_meta.get("dropped_reason"):
                    logger.info(
                        "[REASONING] tier=%s model=%s fmt=%s src=%s "
                        "requested=%s -> %s clamped=%s dropped=%s sent=%s",
                        cand_route,
                        cand_backend.get("model", ""),
                        outbound_fmt,
                        reasoning_meta.get("source"),
                        reasoning_meta.get("requested"),
                        reasoning_meta.get("effective"),
                        reasoning_meta.get("clamped"),
                        reasoning_meta.get("dropped_reason"),
                        {
                            k: outbound_body.get(k)
                            for k in (
                                "reasoning_effort",
                                "reasoning",
                                "thinking",
                                "output_config",
                            )
                            if outbound_body.get(k) is not None
                        },
                    )

                client = None
                resp = None
                raw_iter = None
                accepted = False
                attempt_started = time.monotonic()
                def report_attempt(
                    success: bool, status: int | None, error_type: str
                ) -> None:
                    if attempt_callback is None:
                        return
                    try:
                        attempt_callback(
                            {
                                "tier": cand_route,
                                "route_class": cand_route,
                                "model": cand_backend["model"],
                                "provider": cand_backend.get("provider", ""),
                                "base_url": cand_backend.get("base_url", ""),
                                "format": outbound_fmt,
                                "success": success,
                                "status_code": status,
                                "error_type": error_type,
                                "latency_s": time.monotonic() - attempt_started,
                            }
                        )
                    except Exception as exc:
                        logger.warning("attempt callback failed: %s", exc)
                try:
                    client = httpx.AsyncClient(
                        timeout=httpx.Timeout(
                            connect=BACKEND_CONNECT_TIMEOUT,
                            read=BACKEND_STREAM_IDLE_TIMEOUT,
                            write=30.0,
                            pool=30.0,
                        ),
                        trust_env=False,
                    )
                    loop = asyncio.get_running_loop()
                    deadline = loop.time() + stream_start_timeout
                    resp = await asyncio.wait_for(
                        client.send(
                            client.build_request(
                                "POST", url, json=outbound_body, headers=headers
                            ),
                            stream=True,
                        ),
                        timeout=stream_start_timeout,
                    )
                    if resp.status_code != 200:
                        report_attempt(False, resp.status_code, "http_error")
                        logger.warning(
                            "[STREAM] tier=%s HTTP %s; falling back",
                            cand_route,
                            resp.status_code,
                        )
                        continue

                    raw_iter = resp.aiter_bytes().__aiter__()
                    buffered: list[bytes] = []
                    while True:
                        remaining = deadline - loop.time()
                        if remaining <= 0:
                            raise asyncio.TimeoutError
                        try:
                            raw_chunk = await asyncio.wait_for(
                                raw_iter.__anext__(), timeout=remaining
                            )
                        except StopAsyncIteration:
                            break
                        buffered.append(raw_chunk)
                        if _stream_chunk_has_content(raw_chunk):
                            break

                    if not buffered or not _stream_chunk_has_content(buffered[-1]):
                        raise asyncio.TimeoutError

                    await queue.put((
                        "accepted",
                        {
                            "tier": cand_route,
                            "model": cand_backend["model"],
                            "converted": conv_used,
                            "reasoning": reasoning_meta,
                        },
                    ))
                    accepted = True

                    async def raw_stream():
                        for chunk in buffered:
                            yield chunk
                        async for chunk in raw_iter:
                            yield chunk

                    saw_done = False
                    stream_usage: dict[str, Any] = {}
                    tool_tracker = StreamToolCallTracker()
                    output_estimator = StreamOutputTokenEstimator()
                    sse_buffer = ""

                    def absorb_usage(chunk: bytes) -> None:
                        nonlocal sse_buffer
                        text = sse_buffer + chunk.decode("utf-8", errors="replace")
                        frames = text.split("\n")
                        sse_buffer = frames.pop()
                        for frame in frames:
                            if not frame.startswith("data: "):
                                continue
                            raw = frame[6:].strip()
                            if not raw or raw == "[DONE]":
                                continue
                            try:
                                event = json.loads(raw)
                            except Exception:
                                continue
                            tool_tracker.feed(event)
                            output_estimator.feed(event)
                            found = _usage_from_event(event)
                            if found:
                                stream_usage.clear()
                                stream_usage.update(found)

                    if can_convert and inbound_fmt != outbound_fmt:
                        async for out_chunk in convert_stream(
                            raw_stream(),
                            inbound_fmt,
                            outbound_fmt,
                            request_body=body,
                        ):
                            saw_done = saw_done or b"[DONE]" in out_chunk
                            absorb_usage(out_chunk)
                            await queue.put(("chunk", out_chunk))
                    else:
                        async for out_chunk in raw_stream():
                            saw_done = saw_done or b"[DONE]" in out_chunk
                            absorb_usage(out_chunk)
                            await queue.put(("chunk", out_chunk))

                    absorb_usage(b"\n")
                    if not saw_done:
                        await queue.put(("chunk", b"data: [DONE]\n\n"))
                    logger.info(
                        "[STREAM] tier=%s first content accepted; relay complete",
                        cand_route,
                    )
                    report_attempt(True, 200, "")
                    quality_reason = tool_tracker.failure_reason()
                    if quality_callback:
                        if quality_reason:
                            quality_callback(quality_reason)
                    if repeat_callback and not quality_reason:
                        repeat_callback(tool_tracker.signature())
                    if usage_callback:
                        try:
                            usage_callback(
                                {
                                    "tier": cand_route,
                                    "route_class": cand_route,
                                    "model": cand_backend["model"],
                                    "provider": cand_backend.get("provider", ""),
                                    "usage": stream_usage,
                                    "request_body": body,
                                    "estimated_output_tokens": output_estimator.tokens,
                                    "latency_s": time.monotonic() - request_started,
                                }
                            )
                        except Exception as exc:
                            logger.warning("usage callback failed: %s", exc)
                    await queue.put(("done", None))
                    return
                except asyncio.TimeoutError:
                    report_attempt(False, None, "first_content_timeout")
                    logger.warning(
                        "[STREAM] tier=%s no first content within %.1fs; "
                        "falling back",
                        cand_route,
                        stream_start_timeout,
                    )
                except Exception as exc:
                    report_attempt(False, None, type(exc).__name__)
                    logger.warning(
                        "[STREAM] tier=%s failed %s: %r; falling back",
                        cand_route,
                        type(exc).__name__,
                        exc,
                    )
                    if accepted:
                        await queue.put((
                            "error",
                            {
                                "message": f"stream relay failed: {exc!r}",
                                "attempts": stream_attempts,
                            },
                        ))
                        return
                finally:
                    if raw_iter is not None:
                        try:
                            await raw_iter.aclose()
                        except Exception:
                            pass
                    if resp is not None:
                        try:
                            await resp.aclose()
                        except Exception:
                            pass
                    if client is not None:
                        try:
                            await client.aclose()
                        except Exception:
                            pass

            await queue.put((
                "error",
                {
                    "message": "all backends failed",
                    "attempts": stream_attempts,
                },
            ))

        producer = asyncio.create_task(produce_stream())
        first_kind, first_payload = await queue.get()
        if first_kind == "error":
            await producer
            return JSONResponse(content={"error": first_payload}, status_code=502)

        async def relay_stream():
            try:
                while True:
                    kind, payload = await queue.get()
                    if kind == "chunk":
                        yield payload
                    elif kind == "done":
                        break
                    elif kind == "error":
                        yield (
                            b"data: " + json.dumps(
                                {"error": payload}, ensure_ascii=False
                            ).encode() + b"\n\ndata: [DONE]\n\n"
                        )
                        break
            finally:
                if not producer.done():
                    producer.cancel()
                try:
                    await producer
                except (asyncio.CancelledError, Exception):
                    pass

        return StreamingResponse(
            relay_stream(),
            media_type="text/event-stream",
            headers={
                "X-Router-Tier": first_payload["tier"],
                "X-Router-Model": first_payload["model"],
                "X-Router-Converted": "1" if first_payload["converted"] else "0",
                "X-Router-Reasoning": (
                    f"{first_payload['reasoning'].get('source')}:"
                    f"{(first_payload['reasoning'].get('effective') or {}).get('effort') or '-'}"
                ),
            },
        )

    for idx, (cand_backend, cand_route) in enumerate(candidates):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cand_backend['api_key']}",
        }
        attempts.append(cand_route)
        # Per-candidate format/path: providers may natively speak different
        # formats (tianhe=chat, starfire=responses), so each candidate must
        # be converted to / sent at its own native endpoint.
        outbound_fmt = cand_backend.get("format") or inbound_fmt
        outbound_path = FORMAT_TO_PATH.get(outbound_fmt, "/chat/completions")
        url = f"{cand_backend['base_url']}{outbound_path}"
        logger.info(
            "[ATTEMPT] %d/%d tier=%s model=%s fmt=%s stream=%s",
            idx + 1,
            len(candidates),
            cand_route,
            cand_backend.get("model", ""),
            outbound_fmt,
            is_stream,
        )
        logger.info("attempt %d/%d tier=%s model=%s fmt=%s stream=%s url=%s",
                    idx + 1, len(candidates), cand_route, cand_backend.get("model", ""),
                    outbound_fmt, is_stream, url)
        try:
            outbound_body = body
            # Player-safe: each candidate must send its OWN model name; the
            # endpoint layer only set body.model to the initially-selected
            # tier's model.  Without this, a fallback to another provider
            # would send the wrong model name (e.g. deepseek-flash to
            # starfire's /responses -> 404).
            outbound_body = dict(body)
            candidate_thinking = _promote_thinking_mode(
                thinking_mode, cand_route
            )
            reasoning_eff, reasoning_meta = _merge_reasoning(
                agent_reasoning,
                cand_backend.get("reasoning"),
                candidate_thinking,
                body_max_tokens,
            )
            if reasoning_eff:
                outbound_body = _inject_reasoning(outbound_body, inbound_fmt, reasoning_eff)
            elif ROUTER_REASONING_ENABLED:
                # Routing is on but this model cannot honour the request
                # (unsupported, or cannot be disabled): strip any reasoning
                # field -- including one the agent supplied -- so it is never
                # forwarded to a backend that does not understand it.
                outbound_body = _strip_reasoning(outbound_body, inbound_fmt)
            outbound_body["model"] = cand_backend["model"]
            _conv_used = False
            if can_convert and inbound_fmt != outbound_fmt:
                try:
                    outbound_body = convert_request(outbound_body, inbound_fmt, outbound_fmt)
                    _conv_used = True
                except Exception:
                    outbound_body = body

            # Log AFTER conversion so `sent` shows what the backend really gets.
            if reasoning_meta.get("effective") or reasoning_meta.get("dropped_reason"):
                logger.info(
                    "[REASONING] tier=%s model=%s fmt=%s src=%s "
                    "requested=%s -> %s clamped=%s dropped=%s sent=%s",
                    cand_route,
                    cand_backend.get("model", ""),
                    outbound_fmt,
                    reasoning_meta.get("source"),
                    reasoning_meta.get("requested"),
                    reasoning_meta.get("effective"),
                    reasoning_meta.get("clamped"),
                    reasoning_meta.get("dropped_reason"),
                    {
                        k: outbound_body.get(k)
                        for k in (
                            "reasoning_effort",
                            "reasoning",
                            "thinking",
                            "output_config",
                        )
                        if outbound_body.get(k) is not None
                    },
                )

            if is_stream:
                # Wait only until the first *content* event, then relay the
                # rest live. Buffering the complete stream here made clients
                # appear hung and prevented failover from a stalled backend.
                client_ctx = httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=BACKEND_CONNECT_TIMEOUT,
                        read=BACKEND_STREAM_IDLE_TIMEOUT,
                        write=30.0,
                        pool=30.0,
                    ),
                    trust_env=False,
                )
                stream_ctx = client_ctx.stream("POST", url, json=outbound_body, headers=headers)
                client = await client_ctx.__aenter__()

                buffered: list[bytes] = []
                resp = None
                raw_iter = None
                upstream_error = False
                got_content = False
                try:
                    loop = asyncio.get_running_loop()
                    deadline = loop.time() + stream_start_timeout
                    resp = await asyncio.wait_for(
                        stream_ctx.__aenter__(), timeout=stream_start_timeout
                    )
                    if resp.status_code != 200:
                        logger.warning(
                            "[STREAM] tier=%s HTTP %s; falling back",
                            cand_route,
                            resp.status_code,
                        )
                        logger.warning("stream candidate %s returned HTTP %d",
                                       cand_route, resp.status_code)
                        upstream_error = True
                    else:
                        raw_iter = resp.aiter_bytes().__aiter__()
                        while True:
                            remaining = deadline - loop.time()
                            if remaining <= 0:
                                raise asyncio.TimeoutError
                            try:
                                out_chunk = await asyncio.wait_for(
                                    raw_iter.__anext__(), timeout=remaining
                                )
                            except StopAsyncIteration:
                                break
                            buffered.append(out_chunk)
                            if _stream_chunk_has_content(out_chunk):
                                got_content = True
                                break
                            if sum(len(b) for b in buffered) > 1024 * 1024:
                                logger.warning(
                                    "stream candidate %s buffered 1 MB without content",
                                    cand_route,
                                )
                                break
                except asyncio.TimeoutError:
                    logger.warning(
                        "[STREAM] tier=%s no first content within %.1fs; "
                        "falling back",
                        cand_route,
                        stream_start_timeout,
                    )
                    logger.warning(
                        "stream candidate %s produced no content within %.1fs",
                        cand_route,
                        stream_start_timeout,
                    )
                    upstream_error = True
                except Exception as exc:
                    logger.warning(
                        "[STREAM] tier=%s start failed %s: %s; falling back",
                        cand_route,
                        type(exc).__name__,
                        exc,
                    )
                    logger.warning("stream candidate %s start failed: %s", cand_route, exc)
                    upstream_error = True

                if upstream_error or not got_content:
                    logger.warning("stream candidate %s produced no content; falling back",
                                   cand_route)
                    if raw_iter is not None:
                        try:
                            await raw_iter.aclose()
                        except Exception:
                            pass
                    if resp is not None:
                        try:
                            await resp.aclose()
                        except Exception:
                            pass
                    try:
                        await client.aclose()
                    except Exception:
                        pass
                    continue

                async def _stream(
                    _client=client,
                    _resp=resp,
                    _buffered=list(buffered),
                    _raw_iter=raw_iter,
                    _tier=cand_route,
                ):
                    async def _raw_stream():
                        for chunk in _buffered:
                            yield chunk
                        if _raw_iter is not None:
                            async for chunk in _raw_iter:
                                yield chunk

                    try:
                        if can_convert and inbound_fmt != outbound_fmt:
                            async for chunk in convert_stream(
                                _raw_stream(),
                                inbound_fmt,
                                outbound_fmt,
                                request_body=body,
                            ):
                                yield chunk
                        else:
                            async for chunk in _raw_stream():
                                yield chunk
                    except Exception as exc:
                        logger.warning(
                            "[STREAM] relay %s failed %s: %r",
                            _tier,
                            type(exc).__name__,
                            exc,
                        )
                    finally:
                        if _raw_iter is not None:
                            try:
                                await _raw_iter.aclose()
                            except Exception:
                                pass
                        try:
                            await _resp.aclose()
                        except Exception:
                            pass
                        try:
                            await _client.aclose()
                        except Exception:
                            pass

                logger.info("stream candidate %s accepted (first chunks=%d bytes)",
                            cand_route, sum(len(b) for b in buffered))
                logger.info(
                    "[STREAM] tier=%s first content after %.2fs; relay started",
                    cand_route,
                    loop.time() - (deadline - stream_start_timeout),
                )
                logger.info(
                    "[ROUTE] stream tier=%s model=%s outbound=%s url=%s",
                    cand_route,
                    cand_backend.get("model", ""),
                    outbound_fmt,
                    url,
                )
                return StreamingResponse(
                    _stream(),
                    media_type="text/event-stream",
                    headers={
                        "X-Router-Tier": cand_route,
                        "X-Router-Model": cand_backend["model"],
                        "X-Router-Converted": "1" if _conv_used else "0",
                        "X-Router-Reasoning": (
                            f"{reasoning_meta.get('source')}:"
                            f"{(reasoning_meta.get('effective') or {}).get('effort') or '-'}"
                        ),
                    },
                )

            resp = httpx.post(url, json=outbound_body, headers=headers, timeout=BACKEND_TIMEOUT, trust_env=False)
            result = resp.json()
            status = resp.status_code
            raw_failed = _is_failure(result, status, outbound_fmt)
            if raw_failed and attempt_callback:
                attempt_callback(
                    {
                        "tier": cand_route,
                        "route_class": cand_route,
                        "model": cand_backend["model"],
                        "provider": cand_backend.get("provider", ""),
                        "base_url": cand_backend.get("base_url", ""),
                        "format": outbound_fmt,
                        "success": False,
                        "status_code": status,
                        "error_type": "response_failure",
                        "latency_s": time.monotonic() - request_started,
                    }
                )
            logger.info("candidate %s result status=%d failed=%s conv=%s keys=%s",
                        cand_route, status, raw_failed, _conv_used, list(result.keys())[:8])
            out_resp = result
            if status == 200 and can_convert and _conv_used and inbound_fmt != outbound_fmt:
                try:
                    out_resp = convert_response(result, inbound_fmt, outbound_fmt)
                except Exception:
                    out_resp = result

            # Inspect the RAW backend response for failure (not the converted
            # one, which is shaped for the agent and lacks outbound fields).
            if not _is_failure(result, status, outbound_fmt) or idx == len(candidates) - 1:
                if raw_failed:
                    logger.warning("candidate %s still failed but is last candidate; returning as-is",
                                   cand_route)
                else:
                    logger.info("candidate %s accepted", cand_route)
                logger.info(
                    "[ROUTE] %s model=%s outbound=%s status=%s url=%s",
                    cand_route,
                    cand_backend.get("model", ""),
                    outbound_fmt,
                    status,
                    url,
                )
                out_resp["_router"] = {
                    "tier": cand_route,
                    "model": cand_backend["model"],
                    "vision": vision,
                    "video": video,
                    "source": "vision_route" if vision else "ml_route",
                    "endpoint": outbound_path,
                    "outbound_format": outbound_fmt,
                    "converted": _conv_used and inbound_fmt != outbound_fmt,
                    "reasoning": reasoning_meta,
                    "attempts": attempts,
                    "fallback_used": idx > 0,
                }
                if usage_callback:
                    try:
                        usage_callback(
                            {
                                "tier": cand_route,
                                "route_class": cand_route,
                                "model": cand_backend["model"],
                                "provider": cand_backend.get("provider", ""),
                                "usage": result.get("usage") or {},
                                "request_body": body,
                                "estimated_output_tokens": estimate_response_output_tokens(
                                    result,
                                    outbound_fmt,
                                ),
                                "latency_s": time.monotonic() - request_started,
                            }
                        )
                    except Exception as exc:
                        logger.warning("usage callback failed: %s", exc)
                if attempt_callback:
                    attempt_callback(
                        {
                            "tier": cand_route,
                            "route_class": cand_route,
                            "model": cand_backend["model"],
                            "provider": cand_backend.get("provider", ""),
                            "base_url": cand_backend.get("base_url", ""),
                            "format": outbound_fmt,
                            "success": True,
                            "status_code": status,
                            "error_type": "",
                            "latency_s": time.monotonic() - request_started,
                        }
                    )
                quality_reason = None
                if quality_callback:
                    quality_reason = response_quality_failure(
                        out_resp, inbound_fmt
                    )
                    if quality_reason:
                        quality_callback(quality_reason)
                if repeat_callback and not quality_reason:
                    repeat_callback(response_tool_signature(out_resp, inbound_fmt))
                return JSONResponse(content=out_resp, status_code=status)
        except Exception as exc:
            logger.exception("candidate %s raised exception", cand_route)
            if idx == len(candidates) - 1:
                result = {"error": {"message": str(exc)}}
                result["_router"] = {
                    "tier": cand_route,
                    "model": cand_backend["model"],
                    "vision": vision,
                    "video": video,
                    "attempts": attempts,
                    "fallback_used": idx > 0,
                    "source": "vision_route" if vision else "ml_route",
                }
                return JSONResponse(content=result, status_code=502)

    return JSONResponse(content={"error": {"message": "all backends failed"}}, status_code=502)

def _fallback_chain(
    route_class: str,
    route_info: dict,
    vision: bool = False,
    video: bool = False,
    minimum_tier: str | None = None,
    valid_tiers: list[str] | None = None,
) -> list[tuple[dict, str]]:
    """Build failover chain: selected tier first, then remaining tiers by
    canonical order (c0..c3). For vision, only include vision-capable tiers.

    llm-rosetta converts standard OpenAI function tools to each backend's
    native tool format (e.g. Anthropic tool_use) before forwarding, so every
    tier — including anthropic backends like DeepSeek — is a valid candidate.
    """
    # Normalize R0..Rn to its tier (c0..cn) before subtracting the selected
    # candidate; otherwise c0 is attempted twice (once as R0, once as c0).
    current = ROUTE_CLASS_TO_TIER.get(route_class, route_class)
    # Get raw route class (R0..R3) to determine order for text vs vision remap
    tiers_in_order = (
        list(valid_tiers)
        if valid_tiers is not None
        else sorted(TIERS.keys(), key=lambda t: int(t[1:]))
    )
    if vision and valid_tiers is None:
        tiers_in_order = _vision_valid_tiers(required_video=video)
    # remove current, build chain
    rest = [
        t
        for t in tiers_in_order
        if t != current
        and (
            minimum_tier is None
            or int(t[1:]) >= int(minimum_tier[1:])
        )
    ]
    chain = []
    if current in tiers_in_order:
        try:
            chain.append((_get_backend(current), current))
        except Exception:
            pass
    for t in rest:
        try:
            chain.append((_get_backend(t), t))
        except Exception:
            continue
    return chain


def _classify(
    user_text: str,
    messages: list[dict[str, Any]],
    tools: list | None,
    valid_tiers: list[str] | None = None,
    context_tokens_estimate: int | None = None,
) -> dict:

    """Run the full SquillaRouter pipeline: ML classify + 4-gate policy."""

    # When valid_tiers is provided (vision routing), build a remap from
    # R0-R3 to the nearest vision-capable tier.
    tier_remap = None
    if valid_tiers is not None:
        tier_remap = {
            f"R{i}": valid_tiers[min(i, len(valid_tiers) - 1)]
            for i in range(len(TIERS))
        }

    if _core is None or _request_type is None:

        # Fallback: no ML classifier available, use default R1.

        route_class = "R1"

        confidence = 0.5

        thinking_mode = "T2"

        prompt_policy = "P1"

        trace = []

    else:

        # Build the request for the ML core.

        history_user_texts = []

        prev_assistant_text = None

        for msg in messages[:-1]:

            role = msg.get("role", "")

            content = msg.get("content", "")

            if role == "user" and isinstance(content, str) and content.strip():

                history_user_texts.append(content.strip()[-8000:])

            elif role == "assistant" and isinstance(content, str) and content.strip():

                prev_assistant_text = content.strip()[-8000:]

        history_user_texts = history_user_texts[-4:]



        context_tokens_est = max(
            0,
            (len(user_text) + sum(len(t) for t in history_user_texts) + len(prev_assistant_text or "")) // 4,
            int(context_tokens_estimate or 0),
        )



        request = _request_type(

            current_user_text=user_text,

            history_user_texts=history_user_texts,

            prev_assistant_text=prev_assistant_text,

            prev_assistant_usage=None,

            prev_route_decisions=[],

            context_metadata={

                "turn_index": len(history_user_texts),

                "history_user_turn_count": len(history_user_texts),

                "context_tokens_est": context_tokens_est,

                "has_code_block": "```" in user_text,

                "has_prev_assistant": prev_assistant_text is not None,

            },

        )

        result = _core.predict(request)

        decision = result.decision

        route_class = str(decision.route_class)

        confidence = float(result.probabilities.get(route_class, 0.5))

        thinking_mode = str(decision.thinking_mode)

        prompt_policy = str(decision.prompt_policy)



        # Apply the 4-gate policy.

        # We use a simplified ConversationContext for policy only.

        material_chars = len(user_text) + sum(len(t) for t in history_user_texts) + len(prev_assistant_text or "")

        material_tokens = max(
            material_chars // 4, int(context_tokens_estimate or 0)
        )

        has_image = _detect_image_content(messages)



        policy_context = ConversationContext(

            current_user_text=user_text,

            current_user_has_image=has_image,

            tool_calling_required=bool(tools),

            history_user_texts=history_user_texts,

            previous_assistant_text=prev_assistant_text,

            route_history=[],

        )

        policy_result = apply_final_policy(

            route_class=route_class,

            confidence=confidence,

            thinking_mode=thinking_mode,

            prompt_policy=prompt_policy,

            context=policy_context,

        )



        # Manually apply the gates since we don't have a full ConversationContext.

        # In practice, apply_final_policy with context=None should work for

        # confidence_gate + complaint_upgrade + large_context_floor.

        # We re-implement the key logic here for simplicity.

        final_route_class = route_class



        # Large context floor (dynamic: use highest tier for huge contexts,
        # second-highest for large contexts).
        route_classes = sorted(ROUTE_CLASS_TO_TIER.keys(), key=lambda r: int(r[1:]))
        highest = route_classes[-1] if route_classes else "R0"
        second_highest = route_classes[-2] if len(route_classes) > 1 else highest
        lowest_two = route_classes[:2]

        r2_floor_tokens = int(
            os.environ.get("ROUTER_LARGE_CONTEXT_R2_TOKENS", "25000")
        )
        r3_floor_tokens = int(
            os.environ.get("ROUTER_LARGE_CONTEXT_R3_TOKENS", "80000")
        )
        if material_tokens >= r3_floor_tokens:

            if route_class != highest:

                final_route_class = highest

        elif material_tokens >= r2_floor_tokens:

            if route_class in lowest_two:

                final_route_class = second_highest



        final_thinking_mode = thinking_mode

        final_prompt_policy = prompt_policy



        resolved_class = tier_remap.get(final_route_class, final_route_class) if tier_remap else final_route_class
        return {

            "route_class": resolved_class,

            "raw_route_class": route_class,

            "confidence": confidence,

            "thinking_mode": final_thinking_mode,

            "prompt_policy": final_prompt_policy,

            "material_tokens": material_tokens,

            "probabilities": dict(result.probabilities) if hasattr(result, "probabilities") else {},

        }



    # Fallback path (no ML core)

    resolved_class = tier_remap.get(route_class, route_class) if tier_remap else route_class
    return {

        "route_class": resolved_class,

        "raw_route_class": route_class,

        "confidence": confidence,

        "thinking_mode": thinking_mode,

        "prompt_policy": prompt_policy,

        "material_tokens": 0,

        "probabilities": {},

    }





def _get_backend(route_class: str) -> dict[str, str]:
    """Return the backend config for a route_class, resolving provider creds."""

    def _resolve(tier_name: str) -> dict[str, str] | None:
        cfg = TIERS.get(tier_name)
        if not cfg or not cfg.get("model"):
            return None
        provider = str(cfg.get("provider") or "")
        p = PROVIDERS.get(provider) or {}
        fmt = p.get("format") or ""
        if fmt == "/responses":
            fmt = "openai_responses"
        elif fmt == "/messages":
            fmt = "anthropic"
        elif not fmt:
            fmt = "openai_chat"
        return {
            "model": cfg["model"],
            "api_key": p.get("api_key", ""),
            "base_url": p.get("base_url", ""),
            "format": fmt,
            "provider": provider,
            "reasoning": cfg.get("reasoning"),
        }

    tier = route_class if route_class in TIERS else ROUTE_CLASS_TO_TIER.get(route_class, "c1")
    resolved = _resolve(tier)
    if resolved:
        return resolved
    # Fall back to c1.
    fallback = _resolve("c1")
    if fallback:
        return fallback
    raise ValueError(f"no backend configured for {route_class}")


@dataclass(frozen=True)
class SessionRoutePlan:
    identity: RequestIdentity
    previous: RouteState | None
    intent: str
    intent_source: str
    intent_reason: str
    similarity: float | None
    task_id: str
    task_anchor: str
    route_info: dict[str, Any]
    sticky: bool
    minimum_tier: str | None


def _routing_state_store() -> RoutingStateStore | None:
    global _ROUTING_STATE_STORE
    if _ROUTING_STATE_STORE is None:
        try:
            _ROUTING_STATE_STORE = RoutingStateStore(
                ttl_seconds=float(
                    os.environ.get("ROUTER_STATE_TTL_SECONDS", "86400")
                )
            )
        except Exception as exc:
            logger.warning("routing state unavailable: %s", exc)
            return None
    return _ROUTING_STATE_STORE


def _circuit_breaker() -> CircuitBreaker:
    global _CIRCUIT_BREAKER
    if _CIRCUIT_BREAKER is None:
        state_path = (
            os.environ.get("ROUTER_CIRCUIT_STATE_DB")
            or os.environ.get("ROUTER_STATE_DB")
            or str(Path.cwd() / ".squilla_router_state.db")
        )
        options = dict(
            enabled=ROUTER_CIRCUIT_BREAKER_ENABLED,
            failure_threshold=1,
            cooldown_seconds=ROUTER_FAILURE_COOLDOWN_SECONDS,
            rate_limit_cooldown_seconds=ROUTER_RATE_LIMIT_COOLDOWN_SECONDS,
        )
        try:
            _CIRCUIT_BREAKER = CircuitBreaker(
                **options,
                state_path=state_path,
            )
        except Exception as exc:
            logger.warning(
                "persistent circuit state unavailable; using process memory: %s",
                exc,
            )
            _CIRCUIT_BREAKER = CircuitBreaker(**options)
    return _CIRCUIT_BREAKER


def _backend_key(backend: dict[str, Any]) -> str:
    return "|".join(
        (
            str(backend.get("provider") or ""),
            str(backend.get("model") or ""),
            str(backend.get("base_url") or ""),
            str(backend.get("format") or ""),
        )
    )


def _route_class_for_tier(tier: str) -> str:
    for route_class, mapped_tier in ROUTE_CLASS_TO_TIER.items():
        if mapped_tier == tier:
            return route_class
    return "R1"


def _upgrade_session_tier(tier: str, steps: int = 1) -> str:
    ordered = sorted(TIERS, key=lambda item: int(item[1:]))
    if tier not in ordered:
        return tier
    return ordered[min(ordered.index(tier) + steps, len(ordered) - 1)]


def _large_context_floor_tier(
    tokens: int,
    available_tiers: list[str] | None = None,
) -> str | None:
    return policy_large_context_floor_tier(
        tokens,
        available_tiers=tuple(available_tiers or TIERS),
        r2_tokens=int(os.environ.get("ROUTER_LARGE_CONTEXT_R2_TOKENS", "25000")),
        r3_tokens=int(os.environ.get("ROUTER_LARGE_CONTEXT_R3_TOKENS", "80000")),
    )


def _route_info_from_state(state: RouteState) -> dict[str, Any]:
    return {
        "route_class": _route_class_for_tier(state.tier),
        "raw_route_class": state.route_class,
        "confidence": 1.0,
        "thinking_mode": state.thinking_mode,
        "prompt_policy": state.prompt_policy,
        "material_tokens": 0,
        "probabilities": {},
        "session_reused": True,
    }


def _minimum_thinking_mode(tier: str) -> str:
    return {"c1": "T1", "c2": "T2", "c3": "T3"}.get(tier, "T0")


def _promote_thinking_mode(mode: str | None, tier: str) -> str:
    minimum = _minimum_thinking_mode(tier)
    current = mode if mode in _THINKING_MODE_ORDER else "T0"
    return (
        current
        if _THINKING_MODE_ORDER[current] >= _THINKING_MODE_ORDER[minimum]
        else minimum
    )


def _headers_with_authenticated_tenant(request: Request) -> dict[str, str]:
    headers = dict(request.headers)
    principal = getattr(request.state, "router_principal", None)
    tenant_id = (
        principal.tenant_id
        if isinstance(principal, RouterPrincipal)
        else ""
    )
    if tenant_id:
        # The authenticated key is authoritative.  Request metadata cannot
        # select a different tenant after authentication.
        headers["x-squilla-tenant-id"] = tenant_id
    return headers


def _prepare_session_route(
    request: Request,
    body: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list | None,
    *,
    valid_tiers: list[str] | None = None,
) -> SessionRoutePlan:
    identity = resolve_request_identity(
        _headers_with_authenticated_tenant(request),
        body,
    )
    requested_cost_policy = str(
        request.headers.get("x-squilla-cost-policy") or ""
    ).strip().lower()
    cost_aware_enabled = ROUTER_COST_AWARE_ENABLED and requested_cost_policy != "classifier"
    store = _routing_state_store()
    previous = store.get(identity.session_id) if store else None
    decision = decide_intent(
        identity,
        previous,
        explicit_intent=identity.explicit_intent,
        task_similarity_threshold=ROUTER_TASK_CHANGE_SIMILARITY,
    )
    quality_feedback = str(
        request.headers.get("x-squilla-quality-feedback")
        or request.headers.get("x-squilla-retry-reason")
        or ""
    ).strip()
    if quality_feedback and decision.intent == "redo":
        decision = type(decision)(
            "redo",
            "explicit",
            f"quality_feedback:{quality_feedback[:80]}",
            decision.similarity,
        )
    if not ROUTER_SESSION_STICKY and decision.intent in {"continue", "redo"}:
        decision = type(decision)("new_task", "rule", "session_sticky_disabled")

    full_context_tokens = estimate_input_tokens(body)
    reusable = bool(
        previous
        and previous.tier in TIERS
        and (valid_tiers is None or previous.tier in valid_tiers)
    )
    route_info = None
    minimum_tier = None
    if decision.intent == "continue" and reusable and previous is not None:
        route_info = _route_info_from_state(previous)
        floor_tier = _large_context_floor_tier(
            full_context_tokens,
            valid_tiers,
        )
        if floor_tier and int(floor_tier[1:]) > int(previous.tier[1:]):
            route_info["route_class"] = _route_class_for_tier(floor_tier)
            route_info["large_context_floor"] = {
                "from_tier": previous.tier,
                "to_tier": floor_tier,
                "context_tokens_estimate": full_context_tokens,
            }
            route_info["session_reused"] = True
            reusable = False
    elif decision.intent == "redo" and reusable and previous is not None:
        route_info = _classify(
            _extract_user_text(messages),
            messages,
            tools,
            valid_tiers=valid_tiers,
            context_tokens_estimate=full_context_tokens,
        )
        upgraded_tier = _upgrade_session_tier(previous.tier)
        if valid_tiers is not None and upgraded_tier not in valid_tiers:
            upgraded_tier = previous.tier
        minimum_tier = upgraded_tier
        route_info["route_class"] = _route_class_for_tier(upgraded_tier)
        route_info["session_reused"] = True
        route_info["redo_from_tier"] = previous.tier

    if route_info is None:
        route_info = _classify(
            _extract_user_text(messages),
            messages,
            tools,
            valid_tiers=valid_tiers,
            context_tokens_estimate=full_context_tokens,
        )
    task_class = classify_task(body)
    route_info["task_class"] = task_class
    raw_classifier_tier = ROUTE_CLASS_TO_TIER.get(
        route_info.get("raw_route_class"), route_info.get("route_class")
    )
    if raw_classifier_tier:
        route_info["classifier_tier"] = raw_classifier_tier
    selected_tier = ROUTE_CLASS_TO_TIER.get(
        route_info["route_class"], route_info["route_class"]
    )
    floor_tier = _large_context_floor_tier(
        int(route_info.get("material_tokens") or 0),
        valid_tiers,
    )
    if (
        floor_tier
        and raw_classifier_tier
        and int(selected_tier[1:]) > int(raw_classifier_tier[1:])
    ):
        route_info["large_context_floor"] = {
            "from_tier": raw_classifier_tier,
            "to_tier": selected_tier,
            "context_tokens_estimate": route_info.get("material_tokens"),
        }
    if (
        cost_aware_enabled
        and decision.intent in {"new_task", "task_changed"}
        and selected_tier in TIERS
    ):
        global_profiles = store.tier_profiles() if store else {}
        task_profiles = store.task_tier_profiles(task_class) if store else {}
        normalized_profiles = {}
        for tier in TIERS:
            average_output, cache_hit_ratio, observations = task_profiles.get(
                tier, (None, 0.0, 0)
            )
            if observations < ROUTER_MIN_PROFILE_OBSERVATIONS:
                average_output, cache_hit_ratio, observations = global_profiles.get(
                    tier, (None, 0.0, 0)
                )
            if observations < ROUTER_MIN_PROFILE_OBSERVATIONS:
                average_output = None
                cache_hit_ratio = 0.0
            normalized_profiles[tier] = (
                average_output,
                cache_hit_ratio,
                observations,
            )
        candidate_tiers = valid_tiers or sorted(
            TIERS, key=lambda tier: int(tier[1:])
        )
        requested_output_cap = body.get("max_tokens") or body.get(
            "max_output_tokens"
        )
        if not isinstance(requested_output_cap, int):
            requested_output_cap = None
        cost_tier, estimates = choose_cost_aware_tier(
            selected_tier,
            candidate_tiers,
            estimate_input_tokens(body),
            normalized_profiles,
            output_cap=requested_output_cap,
        )
        classifier_tier = route_info.get("classifier_tier", selected_tier)
        route_info["cost_estimates"] = [asdict(item) for item in estimates]
        route_info["cost_aware_selected"] = cost_tier != selected_tier
        route_info["route_class"] = _route_class_for_tier(cost_tier)
        selected_tier = cost_tier
    if valid_tiers is not None and selected_tier not in valid_tiers:
        ordered_valid = sorted(valid_tiers, key=lambda item: int(item[1:]))
        eligible = [
            tier
            for tier in ordered_valid
            if int(tier[1:]) >= int(selected_tier[1:])
        ]
        capability_tier = eligible[0] if eligible else ordered_valid[-1]
        route_info["capability_remap"] = {
            "from_tier": selected_tier,
            "to_tier": capability_tier,
            "reason": "request_capability_or_context",
        }
        route_info["route_class"] = _route_class_for_tier(capability_tier)
        selected_tier = capability_tier
    route_info["cost_policy"] = (
        "cost_aware" if cost_aware_enabled else "classifier"
    )
    route_info["thinking_mode"] = _promote_thinking_mode(
        route_info.get("thinking_mode"), selected_tier
    )

    if decision.intent in {"new_task", "task_changed"} or previous is None:
        task_id = identity.task_id_hint or uuid4().hex
        task_anchor = identity.anchor or identity.latest_user_text
    else:
        task_id = previous.task_id
        task_anchor = previous.task_anchor

    return SessionRoutePlan(
        identity=identity,
        previous=previous,
        intent=decision.intent,
        intent_source=decision.source,
        intent_reason=decision.reason,
        similarity=decision.similarity,
        task_id=task_id,
        task_anchor=task_anchor,
        route_info=route_info,
        sticky=decision.intent == "continue" and reusable,
        minimum_tier=minimum_tier,
    )


def _record_session_route(
    plan: SessionRoutePlan, response: JSONResponse | StreamingResponse
) -> None:
    store = _routing_state_store()
    if store is None:
        return
    actual_route = (
        response.headers.get("X-Router-Tier") or plan.route_info["route_class"]
    )
    actual_tier = ROUTE_CLASS_TO_TIER.get(actual_route, actual_route)
    tier_cfg = TIERS.get(actual_tier) or {}
    provider = str(tier_cfg.get("provider") or "")
    provider_cfg = PROVIDERS.get(provider) or {}
    model = response.headers.get("X-Router-Model") or str(
        tier_cfg.get("model") or ""
    )
    if not model or actual_tier not in TIERS:
        return
    try:
        store.put(
            plan.identity,
            task_id=plan.task_id,
            task_anchor=plan.task_anchor,
            tier=actual_tier,
            route_class=_route_class_for_tier(actual_tier),
            model=model,
            provider=provider,
            api_format=str(provider_cfg.get("format") or ""),
            base_url=str(provider_cfg.get("base_url") or ""),
            thinking_mode=_promote_thinking_mode(
                plan.route_info.get("thinking_mode"), actual_tier
            ),
            prompt_policy=str(plan.route_info.get("prompt_policy") or "P1"),
            previous=plan.previous,
        )
        if plan.previous and plan.previous.forced_intent:
            store.clear_forced_intent(
                plan.identity.session_id,
                expected_reason=plan.previous.forced_reason or None,
            )
    except StateConflict:
        logger.warning(
            "routing state conflict for session %s; preserving newer writer",
            plan.identity.session_id,
        )
    except Exception as exc:
        logger.warning("failed to persist routing state: %s", exc)


def _attach_session_headers(
    response: JSONResponse | StreamingResponse, plan: SessionRoutePlan
) -> None:
    response.headers["X-Squilla-Session-Id"] = plan.identity.session_id
    response.headers["X-Squilla-Task-Id"] = plan.task_id
    response.headers["X-Squilla-Intent"] = plan.intent
    response.headers["X-Squilla-Intent-Source"] = plan.intent_source
    response.headers["X-Squilla-Intent-Reason"] = plan.intent_reason
    response.headers["X-Squilla-Sticky"] = "1" if plan.sticky else "0"
    if plan.route_info.get("classifier_tier"):
        response.headers["X-Squilla-Classifier-Tier"] = str(
            plan.route_info["classifier_tier"]
        )
        response.headers["X-Squilla-Cost-Aware"] = (
            "1" if plan.route_info.get("cost_aware_selected") else "0"
        )
    if plan.route_info.get("cost_policy"):
        response.headers["X-Squilla-Cost-Policy"] = str(
            plan.route_info["cost_policy"]
        )
    if plan.route_info.get("large_context_floor"):
        response.headers["X-Squilla-Large-Context-Floor"] = str(
            plan.route_info["large_context_floor"]["to_tier"]
        )
    if plan.similarity is not None:
        response.headers["X-Squilla-Task-Similarity"] = f"{plan.similarity:.4f}"


def _price_rates(tier: str) -> tuple[float, float, float]:
    multiplier = {"c0": 1.0, "c1": 4.0, "c2": 10.0, "c3": 20.0}.get(tier, 20.0)
    cache_hit = float(os.environ.get("ROUTER_PRICE_C0_CACHE_HIT", "1"))
    cache_miss = float(os.environ.get("ROUTER_PRICE_C0_CACHE_MISS", "10"))
    output = float(os.environ.get("ROUTER_PRICE_C0_OUTPUT", "40"))
    return cache_hit * multiplier, cache_miss * multiplier, output * multiplier


def _usage_values(usage: dict[str, Any] | None) -> tuple[int, int, int, int]:
    usage = usage or {}
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(
        usage.get("completion_tokens") or usage.get("output_tokens") or 0
    )
    input_details = (
        usage.get("prompt_tokens_details")
        or usage.get("input_tokens_details")
        or {}
    )
    output_details = (
        usage.get("completion_tokens_details")
        or usage.get("output_tokens_details")
        or {}
    )
    cached_tokens = min(
        input_tokens, int(input_details.get("cached_tokens") or 0)
    )
    reasoning_tokens = int(output_details.get("reasoning_tokens") or 0)
    return input_tokens, cached_tokens, output_tokens, reasoning_tokens


def _cost_units(usage: dict[str, Any] | None, tier: str) -> int:
    input_tokens, cached_tokens, output_tokens, _ = _usage_values(usage)
    hit_rate, miss_rate, output_rate = _price_rates(tier)
    cache_miss = max(0, input_tokens - cached_tokens)
    return int(
        cached_tokens * hit_rate
        + cache_miss * miss_rate
        + output_tokens * output_rate
    )


def _record_usage(
    plan: SessionRoutePlan,
    *,
    tier: str,
    route_class: str,
    model: str,
    provider: str,
    usage: dict[str, Any] | None,
    request_body: dict[str, Any] | None = None,
    estimated_output_tokens: int = 0,
    latency_s: float,
) -> None:
    store = _routing_state_store()
    if store is None:
        return
    input_tokens, cached_tokens, output_tokens, reasoning_tokens = _usage_values(
        usage
    )
    provider_input = input_tokens > 0
    provider_output = output_tokens > 0
    if not provider_input:
        input_tokens = estimate_input_tokens(request_body or {})
    if not provider_output:
        output_tokens = max(0, int(estimated_output_tokens))
    if provider_input and provider_output:
        usage_source = "provider"
    elif provider_input or provider_output:
        usage_source = "mixed"
    else:
        usage_source = "estimated"
    effective_usage = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }
    try:
        store.record_call(
            session_id=plan.identity.session_id,
            task_id=plan.task_id,
            intent=plan.intent,
            tier=tier,
            route_class=route_class,
            model=model,
            provider=provider,
            task_class=str(plan.route_info.get("task_class") or "unknown"),
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            latency_s=latency_s,
            cost_units=_cost_units(effective_usage, tier),
            usage_source=usage_source,
        )
    except Exception as exc:
        logger.warning("failed to persist usage ledger: %s", exc)


def _usage_callback_for(plan: SessionRoutePlan) -> Callable[[dict[str, Any]], None]:
    def callback(payload: dict[str, Any]) -> None:
        route = str(payload.get("tier") or payload.get("route_class") or "")
        tier = ROUTE_CLASS_TO_TIER.get(route, route)
        _record_usage(
            plan,
            tier=tier,
            route_class=_route_class_for_tier(tier),
            model=str(payload.get("model") or ""),
            provider=str(payload.get("provider") or ""),
            usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else {},
            request_body=(
                payload.get("request_body")
                if isinstance(payload.get("request_body"), dict)
                else None
            ),
            estimated_output_tokens=int(
                payload.get("estimated_output_tokens") or 0
            ),
            latency_s=float(payload.get("latency_s") or 0),
        )

    return callback





def _attempt_callback_for(plan: SessionRoutePlan) -> Callable[[dict[str, Any]], None]:
    def callback(payload: dict[str, Any]) -> None:
        route = str(payload.get("route_class") or payload.get("tier") or "")
        tier = ROUTE_CLASS_TO_TIER.get(route, route)
        provider = str(payload.get("provider") or "")
        model = str(payload.get("model") or "")
        status_code = payload.get("status_code")
        status = status_code if isinstance(status_code, int) else None
        success = bool(payload.get("success"))
        error_type = str(payload.get("error_type") or "")
        breaker = _circuit_breaker()
        key = "|".join(
            (
                provider,
                model,
                str(payload.get("base_url") or ""),
                str(payload.get("format") or ""),
            )
        )
        if success:
            breaker.record_success(key)
            cooldown = 0.0
        else:
            cooldown = breaker.record_failure(key, status=status, error=error_type)
        store = _routing_state_store()
        if store is not None:
            try:
                store.record_attempt(
                    session_id=plan.identity.session_id,
                    task_id=plan.task_id,
                    intent=plan.intent,
                    tier=tier,
                    route_class=_route_class_for_tier(tier),
                    model=model,
                    provider=provider,
                    success=success,
                    status_code=status,
                    error_type=error_type,
                    latency_s=float(payload.get("latency_s") or 0),
                    cooldown_s=cooldown,
                )
            except Exception as exc:
                logger.warning("failed to persist attempt ledger: %s", exc)

    return callback

def _quality_callback_for(plan: SessionRoutePlan) -> Callable[[str], None]:
    def callback(reason: str) -> None:
        store = _routing_state_store()
        if store is None:
            return
        try:
            store.mark_forced_intent(
                plan.identity.session_id,
                "redo",
                f"automatic_quality_failure:{reason}",
            )
        except Exception as exc:
            logger.warning("failed to mark automatic quality failure: %s", exc)

    return callback


def _repeat_callback_for(plan: SessionRoutePlan) -> Callable[[str], None]:
    threshold = int(os.environ.get("ROUTER_REPEAT_TOOL_CALL_THRESHOLD", "3"))

    def callback(signature: str) -> None:
        store = _routing_state_store()
        if store is None:
            return
        try:
            count = store.record_tool_signature(plan.identity.session_id, signature)
            if signature and threshold > 0 and count >= threshold:
                store.mark_forced_intent(
                    plan.identity.session_id,
                    "redo",
                    f"repeated_tool_call_no_progress:{count}",
                )
        except Exception as exc:
            logger.warning("failed to track repeated tool calls: %s", exc)

    return callback


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    _auth: RouterPrincipal = Depends(_verify_auth),
):
    """OpenAI Chat Completions: classify, pick model, replace model, forward raw to /chat/completions."""
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools", None)
    user_text = _extract_user_text(messages)
    has_image = _detect_image_content(messages)
    has_video = _detect_video_content(messages)
    is_vision = has_image or has_video
    valid_tiers = _matching_tiers(
        body,
        require_image=has_image,
        require_video=has_video,
    )
    if not valid_tiers:
        return JSONResponse(
            content={
                "error": {
                    "message": "no configured backend satisfies request capabilities",
                    "code": "NO_CAPABLE_BACKEND",
                }
            },
            status_code=503,
        )

    if is_vision:
        plan = _prepare_session_route(
            request, body, messages, tools, valid_tiers=valid_tiers
        )
        route_info = plan.route_info
        route_class = route_info["route_class"]
        backend = _get_backend(route_class)
        body["model"] = backend["model"]
        chain = _fallback_chain(
            route_class,
            route_info,
            vision=True,
            video=has_video,
            minimum_tier=plan.minimum_tier,
            valid_tiers=valid_tiers,
        )
        result = await _call_backend(
            body,
            backend,
            route_class,
            inbound_fmt="openai_chat",
            vision=True,
            video=has_video,
            thinking_mode=route_info.get("thinking_mode"),
            fallback_backends=[b for b, _ in chain[1:]],
            fallback_route_classes=[r for _, r in chain[1:]],
            usage_callback=_usage_callback_for(plan),
            attempt_callback=_attempt_callback_for(plan),
            quality_callback=_quality_callback_for(plan),
            repeat_callback=_repeat_callback_for(plan),
        )
        _record_session_route(plan, result)
        _attach_session_headers(result, plan)
        return result

    plan = _prepare_session_route(
        request,
        body,
        messages,
        tools,
        valid_tiers=valid_tiers,
    )
    route_info = plan.route_info
    route_class = route_info["route_class"]
    backend = _get_backend(route_class)
    body["model"] = backend["model"]
    chain = _fallback_chain(
        route_class,
        route_info,
        minimum_tier=plan.minimum_tier,
        valid_tiers=valid_tiers,
    )
    resp = await _call_backend(
        body, backend, route_class, inbound_fmt="openai_chat",
        thinking_mode=route_info.get("thinking_mode"),
        fallback_backends=[b for b, _ in chain[1:]],
        fallback_route_classes=[r for _, r in chain[1:]],
        usage_callback=_usage_callback_for(plan),
        attempt_callback=_attempt_callback_for(plan),
        quality_callback=_quality_callback_for(plan),
        repeat_callback=_repeat_callback_for(plan),
    )
    _record_session_route(plan, resp)
    _attach_session_headers(resp, plan)
    # Merge classification metadata
    if hasattr(resp, "body"):
        import json as _json
        try:
            payload = _json.loads(resp.body)
            if isinstance(payload, dict):
                if "_router" in payload and isinstance(payload["_router"], dict):
                    payload["_router"].update({
                        "raw_route_class": route_info.get("raw_route_class"),
                        "confidence": route_info.get("confidence"),
                        "thinking_mode": route_info.get("thinking_mode"),
                        "material_tokens": route_info.get("material_tokens"),
                        "probabilities": route_info.get("probabilities"),
                    })
                resp.body = _json.dumps(payload).encode("utf-8")
                resp.headers["content-length"] = str(len(resp.body))
        except Exception:
            pass
    return resp


@app.post("/v1/responses")
async def responses_endpoint(
    request: Request,
    _auth: RouterPrincipal = Depends(_verify_auth),
):
    """OpenAI Responses API: classify, pick model, replace model, forward raw to /responses."""
    body = await request.json()
    # Extract user text from Responses 'input'
    user_text = ""
    inp = body.get("input")
    if isinstance(inp, str):
        user_text = inp
    elif isinstance(inp, list):
        parts = []
        for item in inp:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                c = item.get("content", "")
                if isinstance(c, str):
                    parts.append(c)
                elif isinstance(c, list):
                    for p in c:
                        if isinstance(p, dict) and p.get("type") == "input_text":
                            parts.append(p.get("text", ""))
        user_text = " ".join(p for p in parts if p)

    # Detect image in Responses format
    is_vision = False
    has_image = False
    has_video = False
    if isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict):
                c = item.get("content")
                if isinstance(c, list):
                    for p in c:
                        if isinstance(p, dict) and p.get("type") in ("input_image", "image_url", "image", "input_video", "video", "video_url"):
                            is_vision = True
                            if p.get("type") in (
                                "input_image",
                                "image_url",
                                "image",
                            ):
                                has_image = True
                            if p.get("type") in ("input_video", "video", "video_url"):
                                has_video = True
    # For classification, feed a pseudo message so V4 classifier sees the user text
    msgs_for_classify = [{"role": "user", "content": user_text}] if user_text else []
    tools = body.get("tools", None) or body.get("tool", None)
    valid_tiers = _matching_tiers(
        body,
        require_image=has_image,
        require_video=has_video,
    )
    if not valid_tiers:
        return JSONResponse(
            content={
                "error": {
                    "message": "no configured backend satisfies request capabilities",
                    "code": "NO_CAPABLE_BACKEND",
                }
            },
            status_code=503,
        )

    if is_vision:
        plan = _prepare_session_route(
            request, body, msgs_for_classify, tools, valid_tiers=valid_tiers
        )
        route_info = plan.route_info
        route_class = route_info["route_class"]
        backend = _get_backend(route_class)
        body["model"] = backend["model"]
        chain = _fallback_chain(
            route_class,
            route_info,
            vision=True,
            video=has_video,
            minimum_tier=plan.minimum_tier,
            valid_tiers=valid_tiers,
        )
        result = await _call_backend(
            body,
            backend,
            route_class,
            inbound_fmt="openai_responses",
            vision=True,
            video=has_video,
            thinking_mode=route_info.get("thinking_mode"),
            fallback_backends=[b for b, _ in chain[1:]],
            fallback_route_classes=[r for _, r in chain[1:]],
            usage_callback=_usage_callback_for(plan),
            attempt_callback=_attempt_callback_for(plan),
            quality_callback=_quality_callback_for(plan),
            repeat_callback=_repeat_callback_for(plan),
        )
        _record_session_route(plan, result)
        _attach_session_headers(result, plan)
        return result

    plan = _prepare_session_route(
        request,
        body,
        msgs_for_classify,
        tools,
        valid_tiers=valid_tiers,
    )
    route_info = plan.route_info
    route_class = route_info["route_class"]
    backend = _get_backend(route_class)
    body["model"] = backend["model"]
    chain = _fallback_chain(
        route_class,
        route_info,
        minimum_tier=plan.minimum_tier,
        valid_tiers=valid_tiers,
    )
    result = await _call_backend(
        body, backend, route_class, inbound_fmt="openai_responses",
        thinking_mode=route_info.get("thinking_mode"),
        fallback_backends=[b for b, _ in chain[1:]],
        fallback_route_classes=[r for _, r in chain[1:]],
        usage_callback=_usage_callback_for(plan),
        attempt_callback=_attempt_callback_for(plan),
        quality_callback=_quality_callback_for(plan),
        repeat_callback=_repeat_callback_for(plan),
    )
    _record_session_route(plan, result)
    _attach_session_headers(result, plan)
    return result


@app.post("/v1/messages")
async def anthropic_messages(
    request: Request,
    _auth: RouterPrincipal = Depends(_verify_auth),
):
    """Anthropic Messages: classify, pick model, replace model, forward raw to /messages."""
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools", None)
    user_text = _extract_user_text(messages)
    has_image = _detect_image_content(messages)
    has_video = _detect_video_content(messages)
    is_vision = has_image or has_video
    valid_tiers = _matching_tiers(
        body,
        require_image=has_image,
        require_video=has_video,
    )
    if not valid_tiers:
        return JSONResponse(
            content={
                "error": {
                    "message": "no configured backend satisfies request capabilities",
                    "code": "NO_CAPABLE_BACKEND",
                }
            },
            status_code=503,
        )

    if is_vision:
        plan = _prepare_session_route(
            request, body, messages, tools, valid_tiers=valid_tiers
        )
        route_info = plan.route_info
        route_class = route_info["route_class"]
        backend = _get_backend(route_class)
        body["model"] = backend["model"]
        chain = _fallback_chain(
            route_class,
            route_info,
            vision=True,
            video=has_video,
            minimum_tier=plan.minimum_tier,
            valid_tiers=valid_tiers,
        )
        result = await _call_backend(
            body, backend, route_class, inbound_fmt="anthropic", vision=True,
            video=has_video, thinking_mode=route_info.get("thinking_mode"),
            fallback_backends=[b for b, _ in chain[1:]],
            fallback_route_classes=[r for _, r in chain[1:]],
            usage_callback=_usage_callback_for(plan),
            attempt_callback=_attempt_callback_for(plan),
            quality_callback=_quality_callback_for(plan),
            repeat_callback=_repeat_callback_for(plan),
        )
        _record_session_route(plan, result)
        _attach_session_headers(result, plan)
        return result

    plan = _prepare_session_route(
        request,
        body,
        messages,
        tools,
        valid_tiers=valid_tiers,
    )
    route_info = plan.route_info
    route_class = route_info["route_class"]
    backend = _get_backend(route_class)
    body["model"] = backend["model"]
    chain = _fallback_chain(
        route_class,
        route_info,
        minimum_tier=plan.minimum_tier,
        valid_tiers=valid_tiers,
    )
    result = await _call_backend(
        body, backend, route_class, inbound_fmt="anthropic",
        thinking_mode=route_info.get("thinking_mode"),
        fallback_backends=[b for b, _ in chain[1:]],
        fallback_route_classes=[r for _, r in chain[1:]],
        usage_callback=_usage_callback_for(plan),
        attempt_callback=_attempt_callback_for(plan),
        quality_callback=_quality_callback_for(plan),
        repeat_callback=_repeat_callback_for(plan),
    )
    _record_session_route(plan, result)
    _attach_session_headers(result, plan)
    return result


@app.get("/health")

async def health():

    return {"status": "ok", "ml_ready": _core is not None}



@app.post("/admin/reload-config")

async def reload_config_endpoint(
    _auth: RouterPrincipal = Depends(_verify_admin_auth),
):

    """Force a config reload from .env (no restart).  Auth required."""

    _reload_config()

    return {

        "status": "reloaded",

        "providers": {

            name: {"base_url": p.get("base_url", ""), "format": p.get("format", ""),

                   "configured": bool(p.get("api_key"))}

            for name, p in PROVIDERS.items()

        },

        "tiers": {

            t: {"provider": c.get("provider", ""), "model": c.get("model", ""),

                "supports_image": c.get("supports_image", False),

                "supports_video": c.get("supports_video", False),

                "context_length": c.get("context_length")}

            for t, c in TIERS.items()

        },

        "route_class_to_tier": ROUTE_CLASS_TO_TIER,

    }


@app.get("/admin/routing-state")
async def routing_state_endpoint(
    session_id: str,
    _auth: RouterPrincipal = Depends(_verify_admin_auth),
):
    store = _routing_state_store()
    state = store.get(session_id) if store else None
    if state is None:
        return JSONResponse(
            content={"error": {"message": "session state not found"}},
            status_code=404,
        )
    return asdict(state)


@app.delete("/admin/routing-state")
async def delete_routing_state_endpoint(
    session_id: str,
    _auth: RouterPrincipal = Depends(_verify_admin_auth),
):
    store = _routing_state_store()
    deleted = store.delete(session_id) if store else False
    return {"deleted": deleted, "session_id": session_id}


@app.post("/admin/purge-state")
async def purge_routing_state_endpoint(
    _auth: RouterPrincipal = Depends(_verify_admin_auth),
):
    store = _routing_state_store()
    return {"purged": store.purge_expired() if store else 0}


@app.get("/admin/circuit-breakers")
async def circuit_breakers_endpoint(
    _auth: RouterPrincipal = Depends(_verify_admin_auth),
):
    breaker = _circuit_breaker()
    snapshot = breaker.snapshot()
    now = time.time() if breaker.state_path is not None else time.monotonic()
    return {
        "enabled": breaker.enabled,
        "persistent": breaker.state_path is not None,
        "count": len(snapshot),
        "backends": {
            key: {
                "consecutive_failures": health.consecutive_failures,
                "cooldown_remaining_seconds": round(
                    max(0.0, health.cooldown_until - now),
                    3,
                ),
                "last_status": health.last_status,
                "last_error": health.last_error,
            }
            for key, health in snapshot.items()
        },
    }


@app.get("/admin/routing-usage")
async def routing_usage_endpoint(
    session_id: str,
    _auth: RouterPrincipal = Depends(_verify_admin_auth),
):
    store = _routing_state_store()
    calls = store.list_calls(session_id) if store else []
    attempts = store.list_attempts(session_id) if store else []
    totals = {
        "calls": len(calls),
        "input_tokens": sum(call.input_tokens for call in calls),
        "cached_tokens": sum(call.cached_tokens for call in calls),
        "output_tokens": sum(call.output_tokens for call in calls),
        "reasoning_tokens": sum(call.reasoning_tokens for call in calls),
        "latency_s": round(sum(call.latency_s for call in calls), 6),
        "cost_units": sum(call.cost_units for call in calls),
        "estimated_usage_calls": sum(
            1 for call in calls if call.usage_source != "provider"
        ),
        "mixed_usage_calls": sum(
            1 for call in calls if call.usage_source == "mixed"
        ),
        "attempts": len(attempts),
        "failed_attempts": sum(1 for attempt in attempts if not attempt.success),
        "cooldown_seconds": round(
            sum(attempt.cooldown_s for attempt in attempts), 6
        ),
    }
    return {
        "session_id": session_id,
        "totals": totals,
        "calls": [asdict(call) for call in calls],
        "attempts_detail": [asdict(attempt) for attempt in attempts],
    }




@app.get("/router-status")

async def router_status():

    """Report current tier configs (no keys leaked)."""

    tiers_out = {}
    for tier, cfg in TIERS.items():
        provider = PROVIDERS.get(str(cfg.get("provider") or ""), {})
        tiers_out[tier] = {
            "model": cfg.get("model", ""),
            "provider": cfg.get("provider", ""),
            "base_url": provider.get("base_url", ""),
            "supports_image": cfg.get("supports_image", False),
            "supports_video": cfg.get("supports_video", False),
            "context_length": cfg.get("context_length"),
            "reasoning": cfg.get("reasoning"),
            "configured": bool(cfg.get("model") and provider.get("api_key")),
        }
    return {
        "ml_ready": _core is not None,
        "reasoning_routing": {
            "enabled": ROUTER_REASONING_ENABLED,
            "priority": ROUTER_REASONING_PRIORITY,
            "thinking_effort_map": _thinking_effort_map(),
        },
        "session_routing": {
            "sticky_enabled": ROUTER_SESSION_STICKY,
            "task_change_similarity": ROUTER_TASK_CHANGE_SIMILARITY,
            "cost_aware_enabled": ROUTER_COST_AWARE_ENABLED,
            "min_profile_observations": ROUTER_MIN_PROFILE_OBSERVATIONS,
        },
        "providers": {name: {"base_url": p.get("base_url",""), "configured": bool(p.get("api_key"))} for name, p in PROVIDERS.items()},
        "tiers": tiers_out,
        "route_class_to_tier": ROUTE_CLASS_TO_TIER,
    }
