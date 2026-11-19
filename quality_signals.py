from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def _invalid_arguments(arguments: Any) -> bool:
    if isinstance(arguments, dict):
        return False
    if arguments in (None, "", "{}"):
        return False
    if not isinstance(arguments, str):
        return True
    try:
        value = json.loads(arguments)
    except Exception:
        return True
    return not isinstance(value, dict)


def response_quality_failure(result: dict[str, Any], fmt: str) -> str | None:
    if fmt == "openai_chat":
        message = (result.get("choices") or [{}])[0].get("message") or {}
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if not function.get("name"):
                return "tool_call_missing_name"
            if _invalid_arguments(function.get("arguments")):
                return "tool_call_invalid_json"
        return None
    if fmt == "openai_responses":
        for item in result.get("output") or []:
            if item.get("type") == "function_call":
                if not item.get("name"):
                    return "tool_call_missing_name"
                if _invalid_arguments(item.get("arguments")):
                    return "tool_call_invalid_json"
        return None
    if fmt == "anthropic":
        for block in result.get("content") or []:
            if block.get("type") == "tool_use":
                if not block.get("name"):
                    return "tool_call_missing_name"
                if _invalid_arguments(block.get("input")):
                    return "tool_call_invalid_json"
    return None


@dataclass
class StreamingToolCall:
    name: str = ""
    arguments: str = ""


class StreamToolCallTracker:
    def __init__(self) -> None:
        self._calls: dict[str, StreamingToolCall] = {}

    def feed(self, event: dict[str, Any]) -> None:
        choice = (event.get("choices") or [{}])[0]
        for position, call in enumerate(
            (choice.get("delta") or {}).get("tool_calls") or []
        ):
            key = str(call.get("index", call.get("id", position)))
            current = self._calls.setdefault(key, StreamingToolCall())
            function = call.get("function") or {}
            if function.get("name"):
                current.name += str(function["name"])
            if function.get("arguments"):
                current.arguments += str(function["arguments"])

        event_type = event.get("type")
        if event_type == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                key = str(block.get("id") or event.get("index", 0))
                self._calls[key] = StreamingToolCall(
                    name=str(block.get("name") or ""),
                    arguments=json.dumps(
                        block.get("input") or {}, ensure_ascii=False
                    ),
                )
        if event_type == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "input_json_delta":
                key = str(event.get("index", 0))
                current = self._calls.setdefault(key, StreamingToolCall())
                current.arguments += str(delta.get("partial_json") or "")

    def failure_reason(self) -> str | None:
        for call in self._calls.values():
            if not call.name:
                return "tool_call_missing_name"
            if _invalid_arguments(call.arguments):
                return "tool_call_invalid_json"
        return None
