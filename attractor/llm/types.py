"""Core data model types for the Unified LLM Client.

Implements: Message, ContentPart, ContentKind, Role, Request, Response,
Usage, FinishReason, StreamEvent, ToolDefinition, ToolCall, ToolResult, etc.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Role(enum.Enum):
    """Message roles covering all major provider semantics."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"


class ContentKind(enum.Enum):
    """Discriminator tag for ContentPart."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    REDACTED_THINKING = "redacted_thinking"


class StreamEventType(enum.Enum):
    """Event types emitted during streaming."""
    STREAM_START = "stream_start"
    TEXT_DELTA = "text_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    THINKING_DELTA = "thinking_delta"
    FINISH = "finish"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Content data structures
# ---------------------------------------------------------------------------

@dataclass
class ImageData:
    """Image content — URL or inline base64."""
    url: str | None = None
    base64_data: str | None = None
    media_type: str = "image/png"


@dataclass
class AudioData:
    """Audio content."""
    url: str | None = None
    base64_data: str | None = None
    media_type: str = "audio/wav"


@dataclass
class DocumentData:
    """Document content (PDF, etc.)."""
    url: str | None = None
    base64_data: str | None = None
    media_type: str = "application/pdf"
    title: str = ""


@dataclass
class ToolCallData:
    """Tool invocation request from the model."""
    id: str = ""
    name: str = ""
    arguments: str = ""  # JSON string


@dataclass
class ToolResultData:
    """Result of executing a tool."""
    tool_call_id: str = ""
    content: str = ""
    is_error: bool = False


@dataclass
class ThinkingData:
    """Reasoning/thinking content from the model."""
    text: str = ""
    signature: str | None = None
    redacted: bool = False


# ---------------------------------------------------------------------------
# ContentPart — tagged union
# ---------------------------------------------------------------------------

@dataclass
class ContentPart:
    """A single part of a message's content. Uses tagged-union pattern."""
    kind: ContentKind | str = ContentKind.TEXT
    text: str | None = None
    image: ImageData | None = None
    audio: AudioData | None = None
    document: DocumentData | None = None
    tool_call: ToolCallData | None = None
    tool_result: ToolResultData | None = None
    thinking: ThinkingData | None = None

    @staticmethod
    def text_part(text: str) -> ContentPart:
        return ContentPart(kind=ContentKind.TEXT, text=text)

    @staticmethod
    def image_part(url: str | None = None, base64_data: str | None = None,
                   media_type: str = "image/png") -> ContentPart:
        return ContentPart(kind=ContentKind.IMAGE,
                           image=ImageData(url=url, base64_data=base64_data,
                                           media_type=media_type))

    @staticmethod
    def tool_call_part(call_id: str, name: str, arguments: str) -> ContentPart:
        return ContentPart(kind=ContentKind.TOOL_CALL,
                           tool_call=ToolCallData(id=call_id, name=name,
                                                  arguments=arguments))

    @staticmethod
    def tool_result_part(tool_call_id: str, content: str,
                         is_error: bool = False) -> ContentPart:
        return ContentPart(kind=ContentKind.TOOL_RESULT,
                           tool_result=ToolResultData(
                               tool_call_id=tool_call_id,
                               content=content, is_error=is_error))

    @staticmethod
    def thinking_part(text: str, signature: str | None = None,
                      redacted: bool = False) -> ContentPart:
        return ContentPart(kind=ContentKind.THINKING,
                           thinking=ThinkingData(text=text, signature=signature,
                                                 redacted=redacted))


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """The fundamental unit of conversation."""
    role: Role = Role.USER
    content: list[ContentPart] = field(default_factory=list)
    name: str | None = None
    tool_call_id: str | None = None

    @property
    def text(self) -> str:
        """Concatenated text from all text content parts."""
        return "".join(
            p.text for p in self.content
            if p.kind == ContentKind.TEXT and p.text is not None
        )

    # Convenience constructors
    @staticmethod
    def system(text: str) -> Message:
        return Message(role=Role.SYSTEM,
                       content=[ContentPart.text_part(text)])

    @staticmethod
    def user(text: str) -> Message:
        return Message(role=Role.USER,
                       content=[ContentPart.text_part(text)])

    @staticmethod
    def assistant(text: str) -> Message:
        return Message(role=Role.ASSISTANT,
                       content=[ContentPart.text_part(text)])

    @staticmethod
    def tool_result(tool_call_id: str, content: str,
                    is_error: bool = False) -> Message:
        return Message(
            role=Role.TOOL,
            content=[ContentPart.tool_result_part(tool_call_id, content,
                                                  is_error)],
            tool_call_id=tool_call_id,
        )


# ---------------------------------------------------------------------------
# Tool types
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    """Definition of a tool that the model can call."""
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    arguments_raw: str = ""  # Raw JSON string


@dataclass
class ToolResult:
    """Result of executing a tool call."""
    tool_call_id: str = ""
    content: str = ""
    is_error: bool = False


class ToolChoice(enum.Enum):
    """Control how the model selects tools."""
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------

@dataclass
class ResponseFormat:
    """Controls output format."""
    type: str = "text"  # "text", "json", "json_schema"
    schema: dict[str, Any] | None = None


@dataclass
class Usage:
    """Token usage information."""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class FinishReason:
    """Why generation stopped."""
    reason: str = "stop"  # "stop", "tool_calls", "length", "content_filter"
    raw: str = ""  # Provider-specific raw value


@dataclass
class Warning:
    """Non-fatal issue in a response."""
    message: str = ""
    code: str = ""


@dataclass
class RateLimitInfo:
    """Rate limit metadata from response headers."""
    limit: int | None = None
    remaining: int | None = None
    reset_seconds: float | None = None


@dataclass
class Request:
    """The single input type for complete() and stream()."""
    model: str = ""
    messages: list[Message] = field(default_factory=list)
    provider: str | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice | str | None = None
    response_format: ResponseFormat | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    reasoning_effort: str | None = None
    metadata: dict[str, str] | None = None
    provider_options: dict[str, Any] | None = None


@dataclass
class Response:
    """LLM response."""
    id: str = ""
    model: str = ""
    provider: str = ""
    message: Message = field(default_factory=Message)
    finish_reason: FinishReason = field(default_factory=FinishReason)
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] | None = None
    warnings: list[Warning] = field(default_factory=list)
    rate_limit: RateLimitInfo | None = None

    @property
    def text(self) -> str:
        """Concatenated text from all text parts."""
        return self.message.text

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Extract tool calls from the response message."""
        calls = []
        for part in self.message.content:
            if part.kind == ContentKind.TOOL_CALL and part.tool_call:
                import json
                try:
                    args = json.loads(part.tool_call.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                calls.append(ToolCall(
                    id=part.tool_call.id,
                    name=part.tool_call.name,
                    arguments=args,
                    arguments_raw=part.tool_call.arguments,
                ))
        return calls

    @property
    def reasoning(self) -> str | None:
        """Concatenated reasoning/thinking text."""
        parts = [
            p.thinking.text
            for p in self.message.content
            if p.kind in (ContentKind.THINKING, ContentKind.REDACTED_THINKING)
            and p.thinking and p.thinking.text
        ]
        return "\n".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """A typed event emitted during streaming."""
    type: StreamEventType = StreamEventType.TEXT_DELTA
    delta: str = ""
    tool_call: ToolCall | None = None
    usage: Usage | None = None
    finish_reason: FinishReason | None = None
    response: Response | None = None
    error: str | None = None
