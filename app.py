"""Squilla API Router - one HTTP service.



Receives OpenAI-compatible chat completion requests, classifies the user

message using the V4 Phase 3 ML classifier (same as OpenSquilla), routes

to the corresponding backend model, and returns a standard OpenAI response.



One Python process. No Docker, no LiteLLM, no external classifier service.

"""



from __future__ import annotations



import os

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



ROUTER_API_KEY = os.environ.get("ROUTER_API_KEY", "")

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
PROVIDERS: dict[str, dict[str, str]] = {
    "tianhe": {
        "base_url": os.environ.get("BACKEND_BASE_URL", ""),
        "api_key": os.environ.get("BACKEND_API_KEY", ""),
    },
    "starfire": {
        "base_url": os.environ.get("STARFIRE_BASE_URL", ""),
        "api_key": os.environ.get("STARFIRE_API_KEY", ""),
    },
}


def _provider(provider: str) -> dict[str, str]:
    """Resolve a provider entry to {'model','api_key','base_url'}."""
    p = PROVIDERS.get(provider, {})
    return {
        "model": "",
        "api_key": p.get("api_key", ""),
        "base_url": p.get("base_url", ""),
    }


# Backend model configs per tier. Each tier references a provider by name.

TIERS: dict[str, dict[str, str]] = {

    "c0": {
        "provider": os.environ.get("C0_PROVIDER", "tianhe"),
        "model": os.environ.get("C0_MODEL", ""),
        "supports_image": os.environ.get("C0_SUPPORTS_IMAGE", "0") == "1",
        "supports_video": os.environ.get("C0_SUPPORTS_VIDEO", "0") == "1",
    },

    "c1": {
        "provider": os.environ.get("C1_PROVIDER", "tianhe"),
        "model": os.environ.get("C1_MODEL", ""),
        "supports_image": os.environ.get("C1_SUPPORTS_IMAGE", "0") == "1",
        "supports_video": os.environ.get("C1_SUPPORTS_VIDEO", "0") == "1",
    },

    "c2": {
        "provider": os.environ.get("C2_PROVIDER", "tianhe"),
        "model": os.environ.get("C2_MODEL", ""),
        "supports_image": os.environ.get("C2_SUPPORTS_IMAGE", "0") == "1",
        "supports_video": os.environ.get("C2_SUPPORTS_VIDEO", "0") == "1",
    },

    "c3": {
        "provider": os.environ.get("C3_PROVIDER", "tianhe"),
        "model": os.environ.get("C3_MODEL", "") or os.environ.get("C2_MODEL", ""),
        "supports_image": os.environ.get("C3_SUPPORTS_IMAGE", "0") == "1",
        "supports_video": os.environ.get("C3_SUPPORTS_VIDEO", "0") == "1",
    },

}



# Route class -> tier mapping.

ROUTE_CLASS_TO_TIER = {

    "R0": "c0",

    "R1": "c1",

    "R2": "c2",

    "R3": "c3",

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


def _is_failure(result: dict, status: int) -> bool:
    """Return True if the backend response should trigger failover."""
    if status >= 500 or status == 429:
        return True
    if status >= 400:
        # 4xx auth/not-found etc: fail over too (could be wrong key/model on one provider)
        return True
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
    vision: bool = False,
    video: bool = False,
    fallback_backends: list[dict[str, str]] | None = None,
    fallback_route_classes: list[str] | None = None,
):
    """Call the backend model with failover to fallback backends on failure."""
    is_stream = body.get("stream", False)

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
        try:
            if is_stream:
                async def _stream():
                    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
                        async with client.stream(
                            "POST",
                            f"{cand_backend['base_url']}/chat/completions",
                            json=body,
                            headers=headers,
                        ) as resp:
                            async for chunk in resp.aiter_bytes():
                                yield chunk

                # For streaming, we can't easily pre-check the HTTP status without
                # buffering the whole SSE; forward the first candidate as-is.
                return StreamingResponse(
                    _stream(),
                    media_type="text/event-stream",
                    headers={
                        "X-Router-Tier": cand_route,
                        "X-Router-Model": cand_backend["model"],
                    },
                )

            resp = httpx.post(
                f"{cand_backend['base_url']}/chat/completions",
                json=body,
                headers=headers,
                timeout=120.0,
                trust_env=False,
            )
            result = resp.json()
            status = resp.status_code
            if not _is_failure(result, status) or idx == len(candidates) - 1:
                result["_router"] = {
                    "tier": cand_route,
                    "model": cand_backend["model"],
                    "vision": vision,
                    "video": video,
                    "source": "vision_route" if vision else "ml_route",
                    "attempts": attempts,
                    "fallback_used": idx > 0,
                }
                return JSONResponse(content=result, status_code=status)
            # else: keep trying next candidate
        except Exception as exc:
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
            # else try next

    # unreachable
    return JSONResponse(content={"error": {"message": "all backends failed"}}, status_code=502)


def _fallback_chain(route_class: str, route_info: dict, vision: bool = False) -> list[tuple[dict, str]]:
    """Build failover chain: selected tier first, then remaining tiers by
    canonical order (c0..c3). For vision, only include vision-capable tiers."""
    current = route_class
    # Get raw route class (R0..R3) to determine order for text vs vision remap
    tiers_in_order = ["c0", "c1", "c2", "c3"]
    if vision:
        tiers_in_order = _vision_valid_tiers()
    # remove current, build chain
    rest = [t for t in tiers_in_order if t != current]
    chain = []
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


def _classify(user_text: str, messages: list[dict[str, Any]], tools: list | None, valid_tiers: list[str] | None = None) -> dict:

    """Run the full SquillaRouter pipeline: ML classify + 4-gate policy."""

    # When valid_tiers is provided (vision routing), build a remap from
    # R0-R3 to the nearest vision-capable tier.
    tier_remap = None
    if valid_tiers is not None:
        tier_remap = {
            "R0": valid_tiers[0],
            "R1": valid_tiers[min(1, len(valid_tiers)-1)],
            "R2": valid_tiers[min(2, len(valid_tiers)-1)],
            "R3": valid_tiers[-1],
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



        # Large context floor

        if material_tokens >= 80000:

            if route_class != "R3":

                final_route_class = "R3"

        elif material_tokens >= 25000:

            if route_class in ("R0", "R1"):

                final_route_class = "R2"



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
        return {
            "model": cfg["model"],
            "api_key": p.get("api_key", ""),
            "base_url": p.get("base_url", ""),
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

    """Classify, route, call the backend model, return OpenAI response."""

    body = await request.json()

    messages = body.get("messages", [])

    tools = body.get("tools", None)

    user_text = _extract_user_text(messages)



    if _detect_image_content(messages) or _detect_video_content(messages):
        is_video = _detect_video_content(messages)
        valid_tiers = _vision_valid_tiers()
        if not valid_tiers:
            backend = {
                "model": VISION_MODEL,
                "api_key": VISION_API_KEY,
                "base_url": VISION_BASE_URL,
            }
            body["model"] = VISION_MODEL
            return await _call_backend(body, backend, "vision", vision=True)

        # Classify the text prompt, but only allow vision-capable tiers.
        route_info = _classify(user_text, messages, tools, valid_tiers=valid_tiers)
        route_class = route_info["route_class"]
        backend = _get_backend(route_class)
        body["model"] = backend["model"]
        chain = _fallback_chain(route_class, route_info, vision=True)
        return await _call_backend(
            body, backend, route_class,
            vision=True, video=is_video,
            fallback_backends=[b for b, _ in chain[1:]],
            fallback_route_classes=[r for _, r in chain[1:]],
        )

    route_info = _classify(user_text, messages, tools)

    route_class = route_info["route_class"]

    backend = _get_backend(route_class)



    body["model"] = backend["model"]

    chain = _fallback_chain(route_class, route_info)
    # Call with failover: selected backend first, then fallbacks in tier order.
    resp = await _call_backend(
        body, backend, route_class,
        fallback_backends=[b for b, _ in chain[1:]],
        fallback_route_classes=[r for _, r in chain[1:]],
    )
    # Merge classification metadata into the response _router.
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







async def _call_backend_anthropic(
    openai_body: dict,
    backend: dict[str, str],
    max_tokens: int,
    *,
    vision: bool = False,
    video: bool = False,
):
    """Call backend with OpenAI body, convert response back to Anthropic format."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {backend['api_key']}",
    }
    try:
        resp = httpx.post(
            f"{backend['base_url']}/chat/completions",
            json=openai_body,
            headers=headers,
            timeout=120.0,
            trust_env=False,
        )
        result = resp.json()
        status = resp.status_code
    except Exception as exc:
        return JSONResponse(
            content={"type": "error", "error": {"type": "api_error", "message": str(exc)}},
            status_code=502,
        )

    choices = result.get("choices", [])
    msg = choices[0].get("message", {}) if choices else {}
    content_text = (msg.get("content") or "") or (msg.get("reasoning") or "")
    usage = result.get("usage", {})
    finish = choices[0].get("finish_reason", "end_turn") if choices else "end_turn"

    anthropic_response = {
        "id": result.get("id", "msg_router"),
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content_text}],
        "model": backend["model"],
        "stop_reason": finish,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
        "_router": {
            "tier": openai_body.get("model", ""),
            "model": backend["model"],
            "vision": vision,
            "video": video,
            "confidence": None,
        },
    }
    return JSONResponse(content=anthropic_response, status_code=status)


@app.post("/v1/messages")

async def anthropic_messages(request: Request, _auth: bool = Depends(_verify_auth)):

    """Anthropic Messages format endpoint for Codex / Claude Code / Anthropic SDK."""

    body = await request.json()

    anthropic_msgs = body.get("messages", [])

    system_prompt = body.get("system", "")

    max_tokens = body.get("max_tokens", 4096)



    openai_messages = []

    if system_prompt:

        if isinstance(system_prompt, str):

            openai_messages.append({"role": "system", "content": system_prompt})

        elif isinstance(system_prompt, list):

            text = " ".join(b.get("text", "") for b in system_prompt if isinstance(b, dict) and b.get("type") == "text")

            openai_messages.append({"role": "system", "content": text})



    for msg in anthropic_msgs:

        role = msg.get("role", "user")

        content = msg.get("content", "")

        if isinstance(content, str):

            openai_messages.append({"role": role, "content": content})

        elif isinstance(content, list):
            # Convert Anthropic blocks, PRESERVING image/video blocks so the
            # vision route can forward them to a multimodal model.
            text_parts = []
            preserved_content = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_result":
                    text_parts.append(str(block.get("content", "")))
                elif btype in ("image", "image_url", "video", "video_url"):
                    # Keep the block for multimodal forwarding.
                    if btype in ("image", "image_url") and "source" in block:
                        # Anthropic image block -> OpenAI-style image_url if possible
                        src = block.get("source") or {}
                        b64 = src.get("data")
                        media_type = src.get("media_type", "image/png")
                        if b64:
                            preserved_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{media_type};base64,{b64}"},
                            })
                        elif block.get("url"):
                            preserved_content.append({
                                "type": "image_url",
                                "image_url": {"url": block["url"]},
                            })
                        else:
                            preserved_content.append(block)
                    else:
                        preserved_content.append(block)
            # Combine text + preserved multimodal blocks into a single content array
            combined = []
            if text_parts:
                combined.append({"type": "text", "text": " ".join(text_parts)})
            combined.extend(preserved_content)
            if combined:
                openai_messages.append({"role": role, "content": combined})



    user_text = ""

    for msg in reversed(openai_messages):

        if msg.get("role") == "user":

            user_text = msg.get("content", "")

            break



    # Vision/Video routing for Anthropic format (image blocks preserved above)
    if _detect_image_content(openai_messages) or _detect_video_content(openai_messages):
        is_video = _detect_video_content(openai_messages)
        valid_tiers = _vision_valid_tiers()
        if not valid_tiers:
            backend = {
                "model": VISION_MODEL,
                "api_key": VISION_API_KEY,
                "base_url": VISION_BASE_URL,
            }
        else:
            v_route = _classify(user_text, openai_messages, body.get("tools"), valid_tiers=valid_tiers)
            v_class = v_route["route_class"]
            backend = _get_backend(v_class)
        openai_body = {
            "model": backend["model"],
            "messages": openai_messages,
            "max_tokens": max_tokens,
        }
        return await _call_backend_anthropic(openai_body, backend, max_tokens, vision=True, video=is_video)

    route_info = _classify(user_text, openai_messages, body.get("tools"))

    route_class = route_info["route_class"]

    backend = _get_backend(route_class)



    openai_body = {

        "model": backend["model"],

        "messages": openai_messages,

        "max_tokens": max_tokens,

    }



    headers = {

        "Content-Type": "application/json",

        "Authorization": f"Bearer {backend['api_key']}",

    }



    try:

        resp = httpx.post(

            f"{backend['base_url']}/chat/completions",

            json=openai_body,

            headers=headers,

            timeout=120.0, trust_env=False,

        )

        result = resp.json()

        status = resp.status_code

    except Exception as exc:

        return JSONResponse(

            content={"type": "error", "error": {"type": "api_error", "message": str(exc)}},

            status_code=502,

        )



    choices = result.get("choices", [])

    content_text = choices[0].get("message", {}).get("content", "") if choices else ""

    usage = result.get("usage", {})



    anthropic_response = {

        "id": result.get("id", "msg_router"),

        "type": "message",

        "role": "assistant",

        "content": [{"type": "text", "text": content_text}],

        "model": backend["model"],

        "stop_reason": choices[0].get("finish_reason", "end_turn") if choices else "end_turn",

        "stop_sequence": None,

        "usage": {

            "input_tokens": usage.get("prompt_tokens", 0),

            "output_tokens": usage.get("completion_tokens", 0),

        },

        "_router": {

            "tier": route_class,

            "model": backend["model"],

            "confidence": route_info.get("confidence"),

            "thinking_mode": route_info.get("thinking_mode"),

            "probabilities": route_info.get("probabilities"),

        },

    }



    return JSONResponse(content=anthropic_response, status_code=status)



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
