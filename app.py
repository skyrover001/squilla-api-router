"""Squilla API Router - one HTTP service.



Receives OpenAI-compatible chat completion requests, classifies the user

message using the V4 Phase 3 ML classifier (same as OpenSquilla), routes

to the corresponding backend model, and returns a standard OpenAI response.



One Python process. No Docker, no LiteLLM, no external classifier service.

"""



from __future__ import annotations



import os
import logging

from pathlib import Path

from typing import Any



import httpx

import yaml

from dotenv import load_dotenv

from fastapi import FastAPI, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from fastapi.responses import JSONResponse, StreamingResponse



from squilla_api_router.conversation_context import ConversationContext, RouteDecisionContext

from squilla_api_router.final_policy import apply_final_policy



load_dotenv()



app = FastAPI(title="Squilla API Router", version="0.1.0", docs_url=None)

logger = logging.getLogger("squilla.router")



ROUTER_API_KEY = os.environ.get("ROUTER_API_KEY", "")


# Rosetta conversion gateway (optional). If set, classified requests are
# forwarded to rosetta (model rewritten to `<provider>/<model>`) which converts
# format to each backend's native API.
ROSETTA_URL = os.environ.get("ROSETTA_URL", "")
ROSETTA_API_KEY = os.environ.get("ROSETTA_API_KEY", "sk-proxy-test")

_security = HTTPBearer(auto_error=False)


def _verify_auth(credentials: HTTPAuthorizationCredentials | None = Depends(_security)):
    """Verify the caller presents the configured router API key (if set)."""
    if not ROUTER_API_KEY:
        # No key configured -> allow (open mode)
        return True
    if credentials is None:
        from fastapi import HTTPException, status as http_status
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    provided = credentials.credentials or ""
    # Bearer token may be prefixed with nothing; header already strips "Bearer "
    if provided != ROUTER_API_KEY:
        from fastapi import HTTPException, status as http_status
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Invalid router API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True



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
        tiers[tier] = {
            "provider": os.environ.get(f"{prefix}_PROVIDER", "tianhe").lower(),
            "model": os.environ.get(f"{prefix}_MODEL", ""),
            "supports_image": os.environ.get(f"{prefix}_SUPPORTS_IMAGE", "0") == "1",
            "supports_video": os.environ.get(f"{prefix}_SUPPORTS_VIDEO", "0") == "1",
        }
    return tiers


def _build_route_map(tiers: dict[str, dict]) -> dict[str, str]:
    # Route class R0..R(n-1) -> c0..c(n-1), one tier per route class.
    ordered = sorted(tiers.keys(), key=lambda t: int(t[1:]))
    return {f"R{i}": tier for i, tier in enumerate(ordered)}


PROVIDERS = _discover_providers()
TIERS = _discover_tiers()
ROUTE_CLASS_TO_TIER = _build_route_map(TIERS)


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


def _vision_valid_tiers() -> list[str]:
    """Return tiers whose model supports image (and video if requested)."""
    return [name for name, cfg in TIERS.items() if cfg.get("supports_image")]


def _is_failure(result: dict, status: int, outbound_format: str = "openai_chat") -> bool:
    """Return True if the backend response should trigger failover."""
    if status >= 500 or status == 429 or status >= 400:
        return True
    if outbound_format == "openai_responses":
        # Responses backends return `output` (list) + `status`.
        if result.get("status") == "incomplete":
            return True
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
    if not (msg.get("content") or msg.get("reasoning")):
        return True
    return False


async def _call_backend(
    body: dict,
    backend: dict[str, str],
    route_class: str,
    *,
    inbound_fmt: str = "openai_chat",
    vision: bool = False,
    video: bool = False,
    fallback_backends: list[dict[str, str]] | None = None,
    fallback_route_classes: list[str] | None = None,
):
    from squilla_api_router._formats import (
        FORMAT_TO_PATH,
        convert_request,
        convert_response,
        convert_stream,
        cd_available,
    )

    can_convert = cd_available()
    is_stream = bool(body.get("stream", False))

    attempts: list[str] = []

    candidates: list[tuple[dict[str, str], str]] = [(backend, route_class)]
    for fb, frc in zip(fallback_backends or [], fallback_route_classes or []):
        candidates.append((fb, frc))

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
        logger.info("attempt %d/%d tier=%s model=%s fmt=%s stream=%s url=%s",
                    idx + 1, len(candidates), cand_route, cand_backend.get("model", ""),
                    outbound_fmt, is_stream, url)
        try:
            outbound_body = body
            _conv_used = False
            if can_convert and inbound_fmt != outbound_fmt:
                try:
                    outbound_body = convert_request(body, inbound_fmt, outbound_fmt)
                    _conv_used = True
                except Exception:
                    outbound_body = body

            if is_stream:
                # Pre-read the upstream stream: some backends (e.g. starfire
                # /responses) return an immediate empty completion.  If the
                # first buffered content is empty/failed, fall through to the
                # next candidate instead of returning an empty stream.
                # Use async-with context managers (like httpx recommends);
                # manually __aenter__ing the stream without __aexit__ causes
                # ReadError after the first chunk.
                client_ctx = httpx.AsyncClient(timeout=120.0, trust_env=False)
                stream_ctx = client_ctx.stream("POST", url, json=outbound_body, headers=headers)
                client = await client_ctx.__aenter__()
                resp = await stream_ctx.__aenter__()

                buffered: list[bytes] = []
                upstream_error = False
                got_content = False
                try:
                    if resp.status_code != 200:
                        logger.warning("stream candidate %s returned HTTP %d",
                                       cand_route, resp.status_code)
                        upstream_error = True
                    else:
                        # Buffer up to ~4 KB of the converted stream to decide
                        # whether this candidate actually produced content.
                        async for out_chunk in convert_stream(resp.aiter_bytes(), inbound_fmt, outbound_fmt):
                            buffered.append(out_chunk)
                            text = out_chunk.decode("utf-8", errors="replace")
                            if "data:" in text and not text.endswith("\n\n"):
                                # partial frame, keep buffering until complete
                                continue
            # Real content: a non-empty delta value (not just the key —
            # tianhe's first chunk is role-only with content:"" and that
            # caused the empty-reply regression).
                            # Buffer the whole stream (up to a byte cap) so the
                            # full reply is in memory; httpx can only read it
                            # once, and stopping at the first content frame
                            # truncated the reply.
                            import json as _tmpjson
                            has_real_content = False
                            for frame in text.split("\n"):
                                if not frame.startswith("data: ") or "[DONE]" in frame:
                                    continue
                                try:
                                    _obj = _tmpjson.loads(frame[6:])
                                except Exception:
                                    continue
                                _delta = (_obj.get("choices") or [{}])[0].get("delta") or {}
                                if (_delta.get("content") or _delta.get("reasoning")
                                        or _delta.get("reasoning_content")):
                                    has_real_content = True
                                    break
                                _evt = _obj.get("type") or ""
                                if _evt in ("response.output_text.delta", "response.reasoning_text.delta"):
                                    has_real_content = True
                                    break
                                if _evt == "content_block_delta":
                                    _d = _obj.get("delta") or {}
                                    if _d.get("type") == "text_delta" and _d.get("text"):
                                        has_real_content = True
                                        break
                            if has_real_content:
                                got_content = True
                            total_bytes = sum(len(b) for b in buffered)
                            if total_bytes > 1024 * 1024:  # 1 MB cap
                                got_content = True
                                break
                            # never break on content alone — keep reading until
                            # the stream is done or the cap is reached, so the
                            # full reply is buffered for the agent.
                except Exception as exc:
                    logger.warning("stream candidate %s pre-read failed: %s", cand_route, exc)
                    upstream_error = True

                if upstream_error or not got_content:
                    logger.warning("stream candidate %s produced no content; falling back",
                                   cand_route)
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
                ):
                    try:
                        for chunk in _buffered:
                            yield chunk
                        # httpx responses can only be streamed once.  The
                        # pre-read loop consumed the upstream stream to detect
                        # content, so we replay only what we buffered.  The
                        # pre-read cap (64 frames) bounds the maximum reply we
                        # can relay; most short replies complete within it.
                        # For longer replies, the upstream content is already
                        # fully buffered (the cap triggers got_content=True).
                    finally:
                        try:
                            await _resp.aclose()
                        except Exception:
                            pass
                        try:
                            await _client.aclose()
                        except Exception:
                            pass

                logger.info("stream candidate %s accepted (buffered=%d bytes)",
                            cand_route, sum(len(b) for b in buffered))
                return StreamingResponse(
                    _stream(),
                    media_type="text/event-stream",
                    headers={
                        "X-Router-Tier": cand_route,
                        "X-Router-Model": cand_backend["model"],
                        "X-Router-Converted": "1" if _conv_used else "0",
                    },
                )

            resp = httpx.post(url, json=outbound_body, headers=headers, timeout=120.0, trust_env=False)
            result = resp.json()
            status = resp.status_code
            raw_failed = _is_failure(result, status, outbound_fmt)
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
                out_resp["_router"] = {
                    "tier": cand_route,
                    "model": cand_backend["model"],
                    "vision": vision,
                    "video": video,
                    "source": "vision_route" if vision else "ml_route",
                    "endpoint": outbound_path,
                    "outbound_format": outbound_fmt,
                    "converted": _conv_used and inbound_fmt != outbound_fmt,
                    "attempts": attempts,
                    "fallback_used": idx > 0,
                }
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

def _fallback_chain(route_class: str, route_info: dict, vision: bool = False,
                    tools: list | None = None) -> list[tuple[dict, str]]:
    """Build failover chain: selected tier first, then remaining tiers by
    canonical order (c0..c3). For vision, only include vision-capable tiers.

    When the request carries tools, skip candidates whose outbound format is
    anthropic: in the current setup that is the DeepSeek Anthropic-compatible
    endpoint, which only accepts its own web_search tools and returns 422 for
    standard OpenAI function tools. Skipping it here avoids a guaranteed
    failure before the first real attempt."""
    current = route_class
    # Get raw route class (R0..R3) to determine order for text vs vision remap
    tiers_in_order = sorted(TIERS.keys(), key=lambda t: int(t[1:]))
    if vision:
        tiers_in_order = _vision_valid_tiers()
    # remove current, build chain
    rest = [t for t in tiers_in_order if t != current]
    chain = []
    try:
        cand = _get_backend(current)
        if not (tools and cand.get("format") == "anthropic"):
            chain.append((cand, current))
    except Exception:
        pass
    for t in rest:
        try:
            cand = _get_backend(t)
            if tools and cand.get("format") == "anthropic":
                continue
            chain.append((cand, t))
        except Exception:
            continue
    return chain


def _classify(user_text: str, messages: list[dict[str, Any]], tools: list | None, valid_tiers: list[str] | None = None) -> dict:

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



        context_tokens_est = max(0, (len(user_text) + sum(len(t) for t in history_user_texts) + len(prev_assistant_text or "")) // 4)



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

        material_tokens = material_chars // 4

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

        if material_tokens >= 80000:

            if route_class != highest:

                final_route_class = highest

        elif material_tokens >= 25000:

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





@app.post("/v1/chat/completions")
async def chat_completions(request: Request, _auth: bool = Depends(_verify_auth)):
    """OpenAI Chat Completions: classify, pick model, replace model, forward raw to /chat/completions."""
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools", None)
    user_text = _extract_user_text(messages)
    is_vision = _detect_image_content(messages) or _detect_video_content(messages)
    has_video = _detect_video_content(messages)

    if is_vision:
        valid_tiers = _vision_valid_tiers()
        if not valid_tiers:
            backend = {"model": VISION_MODEL, "api_key": VISION_API_KEY, "base_url": VISION_BASE_URL}
            body["model"] = VISION_MODEL
            return await _call_backend(body, backend, "vision", inbound_fmt="openai_chat", vision=True, video=has_video)
        route_info = _classify(user_text, messages, tools, valid_tiers=valid_tiers)
        route_class = route_info["route_class"]
        backend = _get_backend(route_class)
        body["model"] = backend["model"]
        chain = _fallback_chain(route_class, route_info, vision=True, tools=tools)
        return await _call_backend(body, backend, route_class, inbound_fmt="openai_chat", vision=True, video=has_video,
                                   fallback_backends=[b for b, _ in chain[1:]],
                                   fallback_route_classes=[r for _, r in chain[1:]])

    route_info = _classify(user_text, messages, tools)
    route_class = route_info["route_class"]
    backend = _get_backend(route_class)
    body["model"] = backend["model"]
    chain = _fallback_chain(route_class, route_info, tools=tools)
    resp = await _call_backend(
        body, backend, route_class, inbound_fmt="openai_chat",
        fallback_backends=[b for b, _ in chain[1:]],
        fallback_route_classes=[r for _, r in chain[1:]],
    )
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
async def responses_endpoint(request: Request, _auth: bool = Depends(_verify_auth)):
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
    has_video = False
    if isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict):
                c = item.get("content")
                if isinstance(c, list):
                    for p in c:
                        if isinstance(p, dict) and p.get("type") in ("input_image", "image_url", "image", "input_video", "video", "video_url"):
                            is_vision = True
                            if p.get("type") in ("input_video", "video", "video_url"):
                                has_video = True
    # For classification, feed a pseudo message so V4 classifier sees the user text
    msgs_for_classify = [{"role": "user", "content": user_text}] if user_text else []
    tools = body.get("tools", None) or body.get("tool", None)

    if is_vision:
        valid_tiers = _vision_valid_tiers()
        if not valid_tiers:
            backend = {"model": VISION_MODEL, "api_key": VISION_API_KEY, "base_url": VISION_BASE_URL}
            body["model"] = VISION_MODEL
            return await _call_backend(body, backend, "vision", inbound_fmt="openai_responses", vision=True, video=has_video)
        route_info = _classify(user_text, msgs_for_classify, tools, valid_tiers=valid_tiers)
        route_class = route_info["route_class"]
        backend = _get_backend(route_class)
        body["model"] = backend["model"]
        chain = _fallback_chain(route_class, route_info, vision=True, tools=tools)
        return await _call_backend(body, backend, route_class, inbound_fmt="openai_responses", vision=True, video=has_video,
                                   fallback_backends=[b for b, _ in chain[1:]],
                                   fallback_route_classes=[r for _, r in chain[1:]])

    route_info = _classify(user_text, msgs_for_classify, tools)
    route_class = route_info["route_class"]
    backend = _get_backend(route_class)
    body["model"] = backend["model"]
    chain = _fallback_chain(route_class, route_info, tools=tools)
    return await _call_backend(
        body, backend, route_class, inbound_fmt="openai_responses",
        fallback_backends=[b for b, _ in chain[1:]],
        fallback_route_classes=[r for _, r in chain[1:]],
    )


@app.post("/v1/messages")
async def anthropic_messages(request: Request, _auth: bool = Depends(_verify_auth)):
    """Anthropic Messages: classify, pick model, replace model, forward raw to /messages."""
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools", None)
    user_text = _extract_user_text(messages)
    is_vision = _detect_image_content(messages) or _detect_video_content(messages)
    has_video = _detect_video_content(messages)

    if is_vision:
        valid_tiers = _vision_valid_tiers()
        if not valid_tiers:
            backend = {"model": VISION_MODEL, "api_key": VISION_API_KEY, "base_url": VISION_BASE_URL}
            body["model"] = VISION_MODEL
            return await _call_backend(body, backend, "vision", inbound_fmt="anthropic", vision=True, video=has_video)
        route_info = _classify(user_text, messages, tools, valid_tiers=valid_tiers)
        route_class = route_info["route_class"]
        backend = _get_backend(route_class)
        body["model"] = backend["model"]
        chain = _fallback_chain(route_class, route_info, vision=True, tools=tools)
        return await _call_backend(body, backend, route_class, inbound_fmt="anthropic", vision=True, video=has_video,
                                   fallback_backends=[b for b, _ in chain[1:]],
                                   fallback_route_classes=[r for _, r in chain[1:]])

    route_info = _classify(user_text, messages, tools)
    route_class = route_info["route_class"]
    backend = _get_backend(route_class)
    body["model"] = backend["model"]
    chain = _fallback_chain(route_class, route_info, tools=tools)
    return await _call_backend(
        body, backend, route_class, inbound_fmt="anthropic",
        fallback_backends=[b for b, _ in chain[1:]],
        fallback_route_classes=[r for _, r in chain[1:]],
    )


@app.get("/health")

async def health():

    return {"status": "ok", "ml_ready": _core is not None}





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
            "configured": bool(cfg.get("model") and provider.get("api_key")),
        }
    return {
        "ml_ready": _core is not None,
        "providers": {name: {"base_url": p.get("base_url",""), "configured": bool(p.get("api_key"))} for name, p in PROVIDERS.items()},
        "tiers": tiers_out,
        "route_class_to_tier": ROUTE_CLASS_TO_TIER,
    }
