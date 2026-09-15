from __future__ import annotations

from dataclasses import dataclass, field

from squilla_api_router.contracts import (
    ImageURLContentPart,
    Message,
    RouteRequest,
    TextContentPart,
)

_HISTORY_USER_MAX_CHARS = 8_000
_HISTORY_USER_MAX_TURNS = 4
_PREVIOUS_ASSISTANT_MAX_CHARS = 8_000
_METRIC_FIELDS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "duration_ms",
)


@dataclass(frozen=True)
class RouteDecisionContext:
    route_class: str
    difficulty: float
    margin: float


@dataclass(frozen=True)
class ConversationContext:
    current_user_text: str | None
    current_user_has_image: bool = False
    tool_calling_required: bool = False
    history_user_texts: list[str] = field(default_factory=list)
    previous_assistant_text: str | None = None
    previous_response_metrics: dict[str, int | float] | None = None
    route_history: list[RouteDecisionContext] = field(default_factory=list)
    feature_availability: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def material_estimated_tokens(self) -> int:
        material_chars = len(self.current_user_text or "")
        material_chars += sum(len(text) for text in self.history_user_texts)
        material_chars += len(self.previous_assistant_text or "")
        return material_chars // 4

    @property
    def requirements(self) -> dict[str, object]:
        return {
            "vision": self.current_user_has_image,
            "tool_calling": self.tool_calling_required,
            "minimum_context_tokens": self.material_estimated_tokens,
        }


def _message_text(message: Message) -> str | None:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        return "\n".join(
            part.text
            for part in message.content
            if isinstance(part, TextContentPart)
        )
    return None


def _current_user_index(request: RouteRequest) -> int:
    return next(
        index
        for index in range(len(request.messages) - 1, -1, -1)
        if request.messages[index].role == "user"
    )


def build_conversation_context(request: RouteRequest) -> ConversationContext:
    current_index = _current_user_index(request)
    current_message = request.messages[current_index]
    current_has_image = isinstance(current_message.content, list) and any(
        isinstance(part, ImageURLContentPart) for part in current_message.content
    )
    current_user_text = _message_text(current_message)
    if current_user_text == "":
        current_user_text = None

    history_user_texts: list[str] = []
    previous_assistant_text: str | None = None
    for message in request.messages[:current_index]:
        if message.role == "user":
            text = _message_text(message)
            if text is not None and text.strip():
                history_user_texts.append(text.strip()[-_HISTORY_USER_MAX_CHARS:])
        elif message.role == "assistant":
            text = _message_text(message)
            if text is not None and text.strip():
                previous_assistant_text = text.strip()[
                    -_PREVIOUS_ASSISTANT_MAX_CHARS:
                ]
    history_user_texts = history_user_texts[-_HISTORY_USER_MAX_TURNS:]

    metrics = request.previous_response_metrics
    metric_availability = {
        field_name: metrics is not None and getattr(metrics, field_name) is not None
        for field_name in _METRIC_FIELDS
    }
    previous_response_metrics = (
        {
            field_name: value
            for field_name in _METRIC_FIELDS
            if (value := getattr(metrics, field_name)) is not None
        }
        if metrics is not None
        else None
    )
    route_history = [
        RouteDecisionContext(
            route_class=entry.final_route_class,
            difficulty=entry.difficulty,
            margin=entry.margin,
        )
        for entry in sorted(request.route_history or [], key=lambda item: item.sequence)
    ]
    feature_availability: dict[str, object] = {
        "current_user_text": current_user_text is not None,
        "history_user_texts": bool(history_user_texts),
        "previous_assistant_text": previous_assistant_text is not None,
        "route_history": {
            "final_route_class": bool(route_history),
            "difficulty": bool(route_history),
            "margin": bool(route_history),
        },
        "previous_response_metrics": metric_availability,
    }
    warnings: list[str] = []
    if not history_user_texts:
        warnings.append("history_user_texts_missing")
    if previous_assistant_text is None:
        warnings.append("previous_assistant_text_missing")
    if not route_history:
        warnings.append("route_history_missing")
    warnings.extend(
        f"previous_response_metrics.{field_name}_missing"
        for field_name in _METRIC_FIELDS
        if not metric_availability[field_name]
    )

    return ConversationContext(
        current_user_text=current_user_text,
        current_user_has_image=current_has_image,
        tool_calling_required=bool(request.tools),
        history_user_texts=history_user_texts,
        previous_assistant_text=previous_assistant_text,
        previous_response_metrics=previous_response_metrics,
        route_history=route_history,
        feature_availability=feature_availability,
        warnings=warnings,
    )

