"""Format conversion for three API shapes using Oaklight/llm-rosetta.

Same public interface as `_formats.py` so `app.py` can swap backend without
signature changes:
  - FMT_CHAT / FMT_RESPONSES / FMT_ANTHROPIC / PATH_TO_FORMAT / FORMAT_TO_PATH
  - convert_request(body, inbound_fmt, outbound_fmt)
  - convert_response(result, inbound_fmt, outbound_fmt)
  - convert_stream(stream, inbound_fmt, outbound_fmt)  [async iterator]
  - cd_available() -> bool

Internally delegates to llm-rosetta (hub-and-spoke IR).  The original
hand-written bridges are kept in `_formats.py` and can be restored by
importing that module instead.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

# Format keys used in config and endpoints (kept identical to _formats.py)
FMT_CHAT = "openai_chat"
FMT_RESPONSES = "openai_responses"
FMT_ANTHROPIC = "anthropic"

PATH_TO_FORMAT = {
    "/chat/completions": FMT_CHAT,
    "/responses": FMT_RESPONSES,
    "/messages": FMT_ANTHROPIC,
}

FORMAT_TO_PATH = {
    FMT_CHAT: "/chat/completions",
    FMT_RESPONSES: "/responses",
    FMT_ANTHROPIC: "/messages",
}

# Normalize legacy name 'open_responses' -> 'openai_responses'
_ALIASES = {
    "open_responses": FMT_RESPONSES,
}


def _prov(name: str) -> str:
    return _ALIASES.get(name, name)


_LLM_ROSATTA = None


def _load_lr():
    """Lazy-load llm-rosetta; returns module or None if unavailable."""
    global _LLM_ROSATTA
    if _LLM_ROSATTA is not None:
        return _LLM_ROSATTA
    try:
        import llm_rosetta  # type: ignore
        _LLM_ROSATTA = llm_rosetta
    except Exception:
        _LLM_ROSATTA = False
    return _LLM_ROSATTA


def cd_available() -> bool:
    return _load_lr() is not False


def convert_request(body: dict, inbound_fmt: str, outbound_fmt: str) -> dict:
    """Convert agent request (inbound format) to backend format (outbound)."""
    if inbound_fmt == outbound_fmt:
        return body
    lr = _load_lr()
    if not lr:
        from squilla_api_router._formats_handwritten import convert_request as _h
        return _h(body, inbound_fmt, outbound_fmt)
    return lr.convert(body, _prov(outbound_fmt), _prov(inbound_fmt))


def convert_response(result: dict, inbound_fmt: str, outbound_fmt: str) -> dict:
    """Convert backend response (outbound format) back to agent format (inbound)."""
    if inbound_fmt == outbound_fmt:
        return result
    lr = _load_lr()
    if not lr:
        from squilla_api_router._formats_handwritten import convert_response as _h
        return _h(result, inbound_fmt, outbound_fmt)
    # llm-rosetta's convert_response expects (upstream_response, request_body,
    # source_provider, target_provider).  We don't carry the request body here,
    # so we pass the response itself as a best-effort request; the pipeline
    # primarily needs it for tool-call context, and plain text still converts.
    request_body = _infer_request_from_response(result, outbound_fmt)
    return lr.convert_response(
        result,
        request_body,
        _prov(inbound_fmt),
        _prov(outbound_fmt),
    )


def _infer_request_from_response(result: dict, outbound_fmt: str) -> dict:
    """Build a minimal request body used only to satisfy llm-rosetta's
    convert_response signature (tool context is not needed for plain text)."""
    model = result.get("model", "")
    if outbound_fmt == FMT_CHAT:
        # Backend returned chat; the source client sent something.  We can't
        # know the original, but a minimal user message keeps IR valid.
        return {"model": model, "messages": [{"role": "user", "content": ""}]}
    if outbound_fmt == FMT_RESPONSES:
        return {"model": model, "input": []}
    if outbound_fmt == FMT_ANTHROPIC:
        return {"model": model, "messages": [{"role": "user", "content": ""}]}
    return {"model": model}


async def convert_stream(
    stream: AsyncIterator[bytes],
    inbound_fmt: str,
    outbound_fmt: str,
    *,
    request_body: dict[str, Any] | None = None,
) -> AsyncIterator[bytes]:
    """Convert a streaming response from backend (outbound) back to agent
    (inbound) format.

    Uses llm-rosetta's ConversionPipeline for stateful per-chunk translation
    when available; otherwise falls back to the handwritten bridges.
    """
    if inbound_fmt == outbound_fmt:
        async for chunk in stream:
            yield chunk
        return

    lr = _load_lr()

    frames = _iter_sse_frames(stream)

    source = _prov(inbound_fmt)   # agent-side format
    target = _prov(outbound_fmt)  # backend-side format (upstream)
    error = None
    try:
        pipeline = lr.pipeline.ConversionPipeline(source, target)
        # request_body must be in SOURCE (agent/inbound) format.
        req_body = request_body if request_body is not None else _dummy_source_request(source)
        pipeline.convert_request(req_body)
        processor = pipeline.create_stream_processor()
    except Exception as exc:
        error = exc

    if error is not None:
        # Fall back to handwritten bridges so streaming still works.
        from squilla_api_router._formats_handwritten import convert_stream as _h
        async for out in _h(stream, inbound_fmt, outbound_fmt):
            yield out
        return

    saw_any = False
    async for payload in frames:
        try:
            out_chunks = processor.process_chunk(payload)
        except Exception as exc:
            continue
        if not out_chunks:
            continue
        saw_any = True
        for out_chunk in out_chunks if isinstance(out_chunks, (list, tuple)) else [out_chunks]:
            data = json.dumps(out_chunk, ensure_ascii=False)
            yield f"data: {data}\n\n".encode()

    if not saw_any:
        yield b"data: [DONE]\n\n"
    else:
        # A Responses SSE stream must terminate with the [DONE] sentinel.
        yield b"data: [DONE]\n\n"


def _dummy_source_request(source_provider: str) -> dict[str, Any]:
    """Minimal request body for llm-rosetta when the original request isn't
    available.  Preferred path passes the real `request_body` instead."""
    if source_provider == FMT_RESPONSES:
        return {"model": "", "input": [{"role": "user", "content": ""}]}
    if source_provider == FMT_ANTHROPIC:
        return {"model": "", "messages": [{"role": "user", "content": ""}]}
    return {"model": "", "messages": [{"role": "user", "content": ""}]}


async def _iter_sse_frames(stream: AsyncIterator[bytes]) -> AsyncIterator[dict]:
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
                yield json.loads(payload)
            except Exception:
                continue
