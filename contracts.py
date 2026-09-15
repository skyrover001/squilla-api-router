from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from squilla_api_router.limits import (
    MAX_CONTENT_PARTS,
    MAX_DESCRIPTION_CHARS,
    MAX_IDENTIFIER_CHARS,
    MAX_MESSAGES,
    MAX_ROUTE_HISTORY,
    MAX_TEXT_CHARS,
    MAX_TOOL_CALLS,
    MAX_TOOLS,
    MAX_URL_CHARS,
)

LimitedText = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CHARS)]
LimitedIdentifier = Annotated[str, Field(min_length=1, max_length=MAX_IDENTIFIER_CHARS)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TextContentPart(StrictModel):
    type: Literal["text"]
    text: LimitedText


class ImageURL(StrictModel):
    url: str = Field(min_length=1, max_length=MAX_URL_CHARS)
    detail: Literal["auto", "low", "high"] = "auto"


class ImageURLContentPart(StrictModel):
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Annotated[TextContentPart | ImageURLContentPart, Field(discriminator="type")]
MessageContent = (
    LimitedText
    | Annotated[list[ContentPart], Field(min_length=1, max_length=MAX_CONTENT_PARTS)]
    | None
)
RouteClass = Literal["R0", "R1", "R2", "R3"]


class FunctionCall(StrictModel):
    name: LimitedIdentifier
    arguments: str = Field(max_length=MAX_TEXT_CHARS)


class ToolCall(StrictModel):
    id: LimitedIdentifier
    type: Literal["function"]
    function: FunctionCall


class FunctionTool(StrictModel):
    name: LimitedIdentifier
    description: Annotated[str, Field(max_length=MAX_DESCRIPTION_CHARS)] | None = None
    parameters: dict[str, object] = Field(default_factory=dict)

    @field_validator("parameters")
    @classmethod
    def require_finite_numbers(cls, parameters: dict[str, object]) -> dict[str, object]:
        pending_values: list[object] = list(parameters.values())
        while pending_values:
            value = pending_values.pop()
            if isinstance(value, float) and not isfinite(value):
                raise ValueError("tool parameters require finite numbers")
            if isinstance(value, dict):
                pending_values.extend(value.values())
            elif isinstance(value, list):
                pending_values.extend(value)
        return parameters


class ToolDefinition(StrictModel):
    type: Literal["function"]
    function: FunctionTool


class Message(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: MessageContent = None
    name: LimitedIdentifier | None = None
    tool_call_id: LimitedIdentifier | None = None
    tool_calls: Annotated[list[ToolCall], Field(min_length=1, max_length=MAX_TOOL_CALLS)] | None = (
        None
    )

    @model_validator(mode="after")
    def validate_role_fields(self) -> Self:
        if isinstance(self.content, str) and not self.content:
            raise ValueError("message text content must not be empty")
        if self.role in {"system", "user"} and self.content is None:
            raise ValueError(f"{self.role} messages require content")
        if self.role == "assistant" and self.content is None and not self.tool_calls:
            raise ValueError("assistant messages require content or tool_calls")
        if self.role == "tool" and (self.content is None or not self.tool_call_id):
            raise ValueError("tool messages require content and tool_call_id")
        if self.role != "assistant" and self.tool_calls is not None:
            raise ValueError("only assistant messages may contain tool_calls")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("only tool messages may contain tool_call_id")
        if self.role != "user" and isinstance(self.content, list):
            if any(isinstance(part, ImageURLContentPart) for part in self.content):
                raise ValueError("only user messages may contain image content")
        return self


class RouteHistoryEntry(StrictModel):
    sequence: int = Field(ge=0)
    route_class: RouteClass
    final_route_class: RouteClass
    difficulty: float = Field(ge=0, le=3, allow_inf_nan=False)
    margin: float = Field(ge=0, le=1, allow_inf_nan=False)


class PreviousResponseMetrics(StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    duration_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class RouteRequest(StrictModel):
    messages: list[Message] = Field(min_length=1, max_length=MAX_MESSAGES)
    tools: list[ToolDefinition] = Field(default_factory=list, max_length=MAX_TOOLS)
    route_history: (
        Annotated[list[RouteHistoryEntry], Field(max_length=MAX_ROUTE_HISTORY)] | None
    ) = None
    previous_response_metrics: PreviousResponseMetrics | None = None

    @model_validator(mode="after")
    def require_user_message(self) -> Self:
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("messages require at least one user message")
        return self

