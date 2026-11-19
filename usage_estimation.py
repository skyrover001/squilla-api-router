"""Approximate output-token accounting when a backend omits usage."""

from __future__ import annotations

import json
from typing import Any


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 2) // 3)


def _json_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _chat_response_text(result: dict[str, Any]) -> str:
    parts: list[str] = []
    for choice in result.get("choices") or []:
        message = choice.get("message") or {}
        parts.append(_json_text(message.get("content")))
        parts.append(_json_text(message.get("reasoning")))
        parts.append(_json_text(message.get("reasoning_content")))
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            parts.append(_json_text(function.get("name")))
            parts.append(_json_text(function.get("arguments")))
    return "\n".join(part for part in parts if part)


def _responses_response_text(result: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in result.get("output") or []:
        if not isinstance(item, dict):
            continue
        parts.append(_json_text(item.get("name")))
        parts.append(_json_text(item.get("arguments")))
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            parts.append(_json_text(content.get("text")))
            parts.append(_json_text(content.get("refusal")))
    parts.append(_json_text(result.get("output_text")))
    return "\n".join(part for part in parts if part)


def _anthropic_response_text(result: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in result.get("content") or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            parts.append(_json_text(block.get("text")))
        elif block_type == "thinking":
            parts.append(_json_text(block.get("thinking")))
        elif block_type == "tool_use":
            parts.append(_json_text(block.get("name")))
            parts.append(_json_text(block.get("input")))
    return "\n".join(part for part in parts if part)


def estimate_response_output_tokens(result: dict[str, Any], fmt: str) -> int:
    if fmt == "openai_responses":
        text = _responses_response_text(result)
    elif fmt == "anthropic":
        text = _anthropic_response_text(result)
    else:
        text = _chat_response_text(result)
    return estimate_text_tokens(text)


class StreamOutputTokenEstimator:
    """Accumulate visible output deltas without retaining the full stream."""

    def __init__(self) -> None:
        self._chars = 0

    def feed(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type.startswith("response.") or event_type in {
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        }:
            delta = event.get("delta")
            if isinstance(delta, str):
                self._chars += len(delta)
            elif isinstance(delta, dict):
                for key in ("text", "thinking", "partial_json", "arguments"):
                    value = delta.get(key)
                    if isinstance(value, str):
                        self._chars += len(value)
            if event_type in {
                "response.output_item.added",
                "response.output_item.done",
            }:
                self._chars += len(_json_text(event.get("item")))
            return

        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            for key in ("content", "reasoning", "reasoning_content"):
                value = delta.get(key)
                if isinstance(value, str):
                    self._chars += len(value)
            for call in delta.get("tool_calls") or []:
                function = call.get("function") or {}
                for key in ("name", "arguments"):
                    value = function.get(key)
                    if isinstance(value, str):
                        self._chars += len(value)

    @property
    def tokens(self) -> int:
        return max(1, (self._chars + 2) // 3) if self._chars else 0
