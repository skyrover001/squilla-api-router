"""Format conversion for three API shapes: OpenAI Chat / Responses / Anthropic.

The router may receive any of the three formats from an agent and forward to a
backend that speaks a different native format.  All request, response and
stream conversions live here so the endpoint layer stays simple.

Responses <-> Chat request/response conversion reuses codex-deepseek
(`_cd_translate` / `_cd_sse`), which is Apache-2.0 and bundled in this repo.
Everything else (Anthropic, streams) is a compact hand-written bridge.
"""

from __future__ import annotations

import json as _json
from typing import Any, AsyncIterator

# Format keys used in config and endpoints
FMT_CHAT = "openai_chat"
FMT_RESPONSES = "openai_responses"
FMT_ANTHROPIC = "anthropic"

# Map endpoint path prefix -> inbound format key
PATH_TO_FORMAT = {
    "/chat/completions": FMT_CHAT,
    "/responses": FMT_RESPONSES,
    "/messages": FMT_ANTHROPIC,
}

# Map format key -> upstream HTTP path
FORMAT_TO_PATH = {
    FMT_CHAT: "/chat/completions",
    FMT_RESPONSES: "/responses",
    FMT_ANTHROPIC: "/messages",
}


# ---------------------------------------------------------------------------
# codex-deepseek translators (responses <-> chat). Zero deps, bundled locally.
# ---------------------------------------------------------------------------
_CD_TRANSLATE = None
_CD_SSE = None


def _load_cd():
    global _CD_TRANSLATE, _CD_SSE
    if _CD_TRANSLATE is None:
        from squilla_api_router._cd_translate import translate_messages
        _CD_TRANSLATE = translate_messages
    if _CD_SSE is None:
        from squilla_api_router._cd_sse import SseTranslator
        _CD_SSE = SseTranslator
    return _CD_TRANSLATE, _CD_SSE


def cd_available() -> bool:
    try:
        _load_cd()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Request conversion
# ---------------------------------------------------------------------------
def _copied(body: dict, out: dict, keys=("temperature", "top_p", "stop", "stream")) -> dict:
    for k in keys:
        if k in body:
            out[k] = body[k]
    return out


def convert_request(body: dict, inbound_fmt: str, outbound_fmt: str) -> dict:
    """Convert a non-streaming request from the agent format to the backend
    format.  Responses -> Chat uses codex-deepseek (no empty-messages bug)."""
    if inbound_fmt == outbound_fmt:
        return body

    if inbound_fmt == FMT_RESPONSES and outbound_fmt == FMT_CHAT:
        translate_messages, _ = _load_cd()
        messages = translate_messages(body.get("input"), {"multimodal": True})["messages"]
        out = {"model": body.get("model", ""), "messages": messages}
        if "stream" in body:
            out["stream"] = body["stream"]
        if "max_output_tokens" in body and "max_tokens" not in out:
            out["max_tokens"] = body["max_output_tokens"]
        return _copied(body, out)

    if inbound_fmt == FMT_CHAT:
        chat = body
    else:
        # responses / anthropic -> chat shape
        if inbound_fmt == FMT_RESPONSES:
            messages = _responses_to_chat_messages(body.get("input"))
            chat = {"model": body.get("model", ""), "messages": messages}
            if body.get("max_output_tokens"):
                chat["max_tokens"] = body["max_output_tokens"]
        else:
            chat = _anthropic_to_chat_request(body)
    if outbound_fmt == FMT_CHAT:
        return chat
    if outbound_fmt == FMT_RESPONSES:
        out = {"model": chat.get("model", ""), "input": chat.get("messages", [])}
        if chat.get("max_tokens"):
            out["max_output_tokens"] = chat["max_tokens"]
        return _copied(chat, out)
    return _chat_to_anthropic_request(chat)


def _responses_to_chat_messages(input_data: Any) -> list[dict]:
    """Best-effort conversion of a Responses `input` into chat messages."""
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]
    if not isinstance(input_data, list):
        return []
    out: list[dict] = []
    for item in input_data:
        if isinstance(item, str):
            out.append({"role": "user", "content": item})
        elif isinstance(item, dict) and item.get("role"):
            role = "system" if item.get("role") == "developer" else item.get("role")
            content = item.get("content")
            if isinstance(content, list):
                content = "".join(
                    (p.get("text") or "") for p in content
                    if isinstance(p, dict) and p.get("type") in ("input_text", "text")
                )
            out.append({"role": role, "content": content or ""})
    return out


def _anthropic_to_chat_request(body: dict) -> dict:
    out = {"model": body.get("model", ""), "messages": body.get("messages", [])}
    if body.get("system"):
        out["messages"] = [{"role": "system", "content": body["system"]}] + out["messages"]
    if "max_tokens" in body:
        out["max_tokens"] = body["max_tokens"]
    return _copied(body, out)


def _chat_to_anthropic_request(body: dict) -> dict:
    messages = list(body.get("messages", []))
    system = ""
    out_messages = []
    for m in messages:
        if m.get("role") == "system":
            system += (m.get("content") or "") + "\n"
        else:
            c = m.get("content")
            if isinstance(c, list):
                # OpenAI multimodal content parts -> anthropic blocks
                blocks = []
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "text":
                        blocks.append({"type": "text", "text": p.get("text", "")})
                    elif isinstance(p, dict) and p.get("type") == "image_url":
                        url = p.get("image_url")
                        url = url.get("url") if isinstance(url, dict) else url
                        blocks.append({"type": "image", "source": {
                            "type": "base64", "media_type": "image/jpeg",
                            "data": (url or "").split(",", 1)[-1] if url and url.startswith("data:") else url or "",
                        }})
                c = blocks
            out_messages.append({**m, "content": c})
    if not out_messages:
        # never send an empty messages array to anthropic
        out_messages = [{"role": "user", "content": ""}]
    max_tokens = body.get("max_tokens") or body.get("max_output_tokens") or body.get("max_completion_tokens") or 8192
    out = {"model": body.get("model", ""), "messages": out_messages, "max_tokens": max_tokens}
    if system:
        out["system"] = system.strip()
    return _copied(body, out)


# ---------------------------------------------------------------------------
# Non-streaming response conversion
# ---------------------------------------------------------------------------
def _output_text_from_items(output: Any) -> str:
    """Pull the assistant text out of a Responses `output` list."""
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            content = item.get("content") or []
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                    return part.get("text", "")
        elif item.get("type") == "reasoning":
            content = item.get("content") or []
            for part in content:
                if isinstance(part, dict):
                    return part.get("text", "")
    return ""


def _chat_response_from(fields: dict) -> dict:
    return {
        "id": fields.get("id", "chatcmpl-1"),
        "object": "chat.completion",
        "created": int(fields.get("created", 0)),
        "model": fields.get("model", ""),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": fields.get("content")},
            "finish_reason": fields.get("finish_reason", "stop"),
        }],
        "usage": {
            "prompt_tokens": fields.get("prompt_tokens", 0),
            "completion_tokens": fields.get("completion_tokens", 0),
            "total_tokens": fields.get("total_tokens", 0),
        },
    }


def convert_response(result: dict, inbound_fmt: str, outbound_fmt: str) -> dict:
    """Convert a non-streaming backend response (outbound format) back to the
    format the agent (inbound_fmt) expects."""
    if inbound_fmt == outbound_fmt:
        return result
    if inbound_fmt == FMT_CHAT:
        if outbound_fmt == FMT_RESPONSES:
            usage = result.get("usage") or {}
            return _chat_response_from({
                "id": result.get("id", "chatcmpl-1"),
                "created": result.get("created_at", 0),
                "model": result.get("model", ""),
                "content": _output_text_from_items(result.get("output")),
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            })
        if outbound_fmt == FMT_ANTHROPIC:
            usage = result.get("usage") or {}
            text = ""
            for part in result.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    break
            return _chat_response_from({
                "id": result.get("id", "chatcmpl-1"),
                "created": 0,
                "model": result.get("model", ""),
                "content": text,
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            })
    if inbound_fmt == FMT_RESPONSES:
        content = ""
        usage = {}
        if outbound_fmt == FMT_CHAT:
            message = (result.get("choices") or [{}])[0].get("message", {}) or {}
            content = message.get("content")
            if not content and message.get("reasoning"):
                # Some backends (tianhe Qwen thinking mode) only return
                # `reasoning` with content=null; surface it as the answer.
                content = message.get("reasoning")
            content = content or ""
            usage = result.get("usage") or {}
            usage = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        elif outbound_fmt == FMT_ANTHROPIC:
            for part in result.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "text":
                    content = part.get("text", "")
                    break
            usage = {
                "input_tokens": (result.get("usage") or {}).get("input_tokens", 0),
                "output_tokens": (result.get("usage") or {}).get("output_tokens", 0),
            }
        output = []
        if content:
            output.append({
                "id": "msg_" + str(result.get("id", "resp")),
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            })
        return {
            "id": str(result.get("id", "resp_") + "_router"),
            "object": "response",
            "created_at": int(result.get("created", result.get("created_at", 0))),
            "status": "completed",
            "model": result.get("model", ""),
            "output": output,
            "usage": usage,
        }
    # inbound == anthropic
    if outbound_fmt == FMT_CHAT:
        message = (result.get("choices") or [{}])[0].get("message", {}) or {}
        content = message.get("content") or message.get("reasoning") or ""
        usage = result.get("usage") or {}
    else:
        content = _output_text_from_items(result.get("output"))
        usage = result.get("usage") or {}
    return {
        "id": str(result.get("id", "msg_router")),
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "model": result.get("model", ""),
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# Streaming conversion
# ---------------------------------------------------------------------------
async def _iter_json(stream: AsyncIterator[bytes]):
    """Yield parsed SSE `data:` payloads from a raw byte stream."""
    buf = b""
    async for chunk in stream:
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            if line.startswith(b"data: "):
                payload = line[6:]
            elif line == b"data:":
                continue
            else:
                continue
            if payload.strip() == b"[DONE]":
                continue
            try:
                yield _json.loads(payload)
            except Exception:
                continue


def _chat_chunk(text: str, *, role: str | None = None, finish: str | None = None,
                usage: dict | None = None, model: str = "") -> dict:
    delta: dict = {}
    if role:
        delta["role"] = role
    if text:
        delta["content"] = text
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage:
        chunk["usage"] = usage
    return chunk


def _sse(obj: dict, *, event: str | None = None) -> bytes:
    data = _json.dumps(obj, ensure_ascii=False)
    if event:
        return f"event: {event}\ndata: {data}\n\n".encode()
    return f"data: {data}\n\n".encode()


async def _chat_to_responses_stream(stream) -> AsyncIterator[bytes]:
    from squilla_api_router._cd_sse import SseTranslator

    translator = SseTranslator()
    started = False
    seen_model = False
    async for chunk in _iter_json(stream):
        if not seen_model:
            model = chunk.get("model")
            if model:
                translator.model = model
                seen_model = True
        if not started:
            started = True
            yield translator._ensure_started().encode()
        out = translator.feed(chunk)
        if out:
            yield out.encode()
    # Always close cleanly; upstream may not send usage.
    yield translator.done().encode()


async def _responses_to_chat_stream(stream) -> AsyncIterator[bytes]:
    usage: dict = {}
    text_parts: list[str] = []
    resp_id = "resp_1"
    model = ""

    async for chunk in _iter_json(stream):
        evt = chunk.get("type") or ""
        if evt == "response.created":
            resp = chunk.get("response") or {}
            resp_id = resp.get("id", resp_id)
            model = resp.get("model", model)
            yield _sse(_chat_chunk("", role="assistant", model=model))
        elif evt in ("response.output_text.delta", "response.reasoning_text.delta"):
            text = chunk.get("delta") or ""
            if text:
                text_parts.append(text)
                yield _sse(_chat_chunk(text, model=model))
        elif evt == "response.completed":
            usage = (chunk.get("response") or {}).get("usage") or {}
        elif evt == "done":
            break

    if not text_parts:
        text_parts = [""]
    usage_map = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
    yield _sse(_chat_chunk("", finish="stop", usage=usage_map, model=model))
    yield b"data: [DONE]\n\n"


async def _chat_to_anthropic_stream(stream) -> AsyncIterator[bytes]:
    import uuid

    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    started = False
    text_parts: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0}

    async for chunk in _iter_json(stream):
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        text = delta.get("content") or ""
        if chunk.get("usage"):
            usage = chunk["usage"]
        if not started:
            started = True
            yield _sse({
                "type": "message_start",
                "message": {
                    "id": msg_id, "type": "message", "role": "assistant",
                    "content": [], "model": chunk.get("model", ""),
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                              "output_tokens": usage.get("completion_tokens", 0)},
                },
            }, event="message_start")
            yield _sse({"type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""}}, event="content_block_start")
        if text:
            text_parts.append(text)
            yield _sse({"type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": text}}, event="content_block_delta")

    if not started:
        yield _sse({"type": "message_start", "message": {
            "id": msg_id, "type": "message", "role": "assistant", "content": [],
            "model": "", "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }}, event="message_start")

    yield _sse({"type": "content_block_stop", "index": 0}, event="content_block_stop")
    yield _sse({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": usage.get("completion_tokens", 0)}}, event="message_delta")
    yield _sse({"type": "message_stop"}, event="message_stop")


async def _anthropic_to_chat_stream(stream) -> AsyncIterator[bytes]:
    usage = {"input_tokens": 0, "output_tokens": 0}
    model = ""
    text_parts: list[str] = []

    async for chunk in _iter_json(stream):
        evt = chunk.get("type") or ""
        if evt == "message_start":
            model = (chunk.get("message") or {}).get("model", model)
            usage = (chunk.get("message") or {}).get("usage") or usage
            yield _sse(_chat_chunk("", role="assistant", model=model))
        elif evt == "content_block_delta":
            delta = chunk.get("delta") or {}
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                text_parts.append(text)
                yield _sse(_chat_chunk(text, model=model))
        elif evt == "message_delta":
            usage = {**usage, **((chunk.get("usage") or {}))}
        elif evt == "message_stop":
            break

    if not text_parts:
        text_parts = [""]
    yield _sse(_chat_chunk("", finish="stop", usage={
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
    }, model=model))
    yield b"data: [DONE]\n\n"


async def _anthropic_to_responses_stream(stream) -> AsyncIterator[bytes]:
    from squilla_api_router._cd_sse import SseTranslator, _rand_id

    driver = SseTranslator()
    driver.response_id = _rand_id("resp")
    started = False
    msg_item_added = False

    async for chunk in _iter_json(stream):
        evt = chunk.get("type") or ""
        if evt == "message_start":
            started = True
            yield driver._ensure_started().encode()
            driver.content_so_far = ""
            driver.text_started = False
        elif evt == "content_block_start":
            # begin accumulating a text block into the translator state
            driver.text_started = True
            if not msg_item_added:
                msg_item_added = True
                oi = driver.output_item_count
                driver.output_item_count += 1
                driver.output_items.append({
                    "index": oi,
                    "type": "message",
                    "itemId": driver.message_item_id,
                })
            yield driver._emit("response.content_part.added", {
                "type": "response.content_part.added",
                "response_id": driver.response_id,
                "item_id": driver.message_item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            }).encode()
        elif evt == "content_block_delta":
            delta = chunk.get("delta") or {}
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                driver.content_so_far += text
                yield driver._emit("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "response_id": driver.response_id,
                    "item_id": driver.message_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text,
                }).encode()
        elif evt == "message_stop":
            break

    if not started:
        yield driver._ensure_started().encode()
    yield driver.done().encode()


async def _responses_to_anthropic_stream(stream) -> AsyncIterator[bytes]:
    import uuid

    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    started = False
    model = ""
    usage = {"input_tokens": 0, "output_tokens": 0}
    output_tokens = 0

    async for chunk in _iter_json(stream):
        evt = chunk.get("type") or ""
        if evt == "response.created":
            resp = chunk.get("response") or {}
            model = resp.get("model", model)
            if not started:
                started = True
                yield _sse({
                    "type": "message_start",
                    "message": {
                        "id": msg_id, "type": "message", "role": "assistant",
                        "content": [], "model": model,
                        "stop_reason": None, "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                }, event="message_start")
                yield _sse({"type": "content_block_start", "index": 0,
                            "content_block": {"type": "text", "text": ""}}, event="content_block_start")
        elif evt in ("response.output_text.delta", "response.reasoning_text.delta"):
            text = chunk.get("delta") or ""
            if text:
                if not started:
                    started = True
                    yield _sse({
                        "type": "message_start",
                        "message": {
                            "id": msg_id, "type": "message", "role": "assistant",
                            "content": [], "model": model,
                            "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    }, event="message_start")
                    yield _sse({"type": "content_block_start", "index": 0,
                                "content_block": {"type": "text", "text": ""}}, event="content_block_start")
                yield _sse({"type": "content_block_delta", "index": 0,
                            "delta": {"type": "text_delta", "text": text}}, event="content_block_delta")
        elif evt == "response.completed":
            usage = (chunk.get("response") or {}).get("usage") or usage
            output_tokens = usage.get("output_tokens", output_tokens)
        elif evt == "done":
            break

    if not started:
        yield _sse({
            "type": "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant", "content": [],
                "model": model, "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }, event="message_start")
        yield _sse({"type": "content_block_start", "index": 0,
                    "content_block": {"type": "text", "text": ""}}, event="content_block_start")

    yield _sse({"type": "content_block_stop", "index": 0}, event="content_block_stop")
    yield _sse({"type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": output_tokens}}, event="message_delta")
    yield _sse({"type": "message_stop"}, event="message_stop")


async def convert_stream(stream, inbound_fmt: str, outbound_fmt: str) -> AsyncIterator[bytes]:
    """Convert a streaming response produced in outbound format back to the
    agent's ingress format (inbound_fmt)."""
    if inbound_fmt == outbound_fmt:
        async for chunk in stream:
            yield chunk
        return

    if inbound_fmt == FMT_CHAT and outbound_fmt == FMT_RESPONSES:
        try:
            async for out in _responses_to_chat_stream(stream):
                yield out
            return
        except Exception:
            pass
    elif inbound_fmt == FMT_CHAT and outbound_fmt == FMT_ANTHROPIC:
        try:
            async for out in _anthropic_to_chat_stream(stream):
                yield out
            return
        except Exception:
            pass
    elif inbound_fmt == FMT_RESPONSES and outbound_fmt == FMT_CHAT:
        try:
            async for out in _chat_to_responses_stream(stream):
                yield out
            return
        except Exception:
            pass
    elif inbound_fmt == FMT_RESPONSES and outbound_fmt == FMT_ANTHROPIC:
        try:
            async for out in _anthropic_to_responses_stream(stream):
                yield out
            return
        except Exception:
            pass
    elif inbound_fmt == FMT_ANTHROPIC and outbound_fmt == FMT_CHAT:
        try:
            async for out in _chat_to_anthropic_stream(stream):
                yield out
            return
        except Exception:
            pass
    elif inbound_fmt == FMT_ANTHROPIC and outbound_fmt == FMT_RESPONSES:
        try:
            async for out in _responses_to_anthropic_stream(stream):
                yield out
            return
        except Exception:
            pass

    # Fallback: raw passthrough (better than dropping the stream).
    async for chunk in stream:
        yield chunk
