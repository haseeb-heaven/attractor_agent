"""Anthropic provider adapter using the Messages API."""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

from attractor.llm.adapters.base import ProviderAdapter
from attractor.llm.errors import (
    AuthenticationError,
    ConfigurationError,
    error_from_status,
)
from attractor.llm.types import (
    ContentKind,
    ContentPart,
    FinishReason,
    Message,
    Request,
    Response,
    Role,
    StreamEvent,
    StreamEventType,
    ThinkingData,
    ToolCallData,
    Usage,
)


class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic's Messages API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ConfigurationError("ANTHROPIC_API_KEY not set")
        self._base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        self._timeout = timeout
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
                kwargs: dict[str, Any] = {
                    "api_key": self._api_key,
                    "timeout": self._timeout,
                }
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                self._client = anthropic.Anthropic(**kwargs)
            except ImportError:
                raise ConfigurationError(
                    "anthropic package not installed. Install with: pip install anthropic"
                )
        return self._client

    @property
    def provider_id(self) -> str:
        return "anthropic"

    def _extract_system(self, messages: list[Message]) -> tuple[str, list[Message]]:
        """Extract system messages (Anthropic uses system parameter, not a role)."""
        system_parts: list[str] = []
        non_system: list[Message] = []
        for msg in messages:
            if msg.role in (Role.SYSTEM, Role.DEVELOPER):
                system_parts.append(msg.text)
            else:
                non_system.append(msg)
        return "\n\n".join(system_parts), non_system

    def _build_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert unified Messages to Anthropic format."""
        result: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == Role.TOOL:
                # Tool results go as user messages with tool_result blocks
                content_blocks: list[dict[str, Any]] = []
                for p in msg.content:
                    if p.kind == ContentKind.TOOL_RESULT and p.tool_result:
                        content_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": p.tool_result.tool_call_id,
                            "content": p.tool_result.content,
                            **({"is_error": True} if p.tool_result.is_error else {}),
                        })
                result.append({"role": "user", "content": content_blocks})
                continue

            role = "assistant" if msg.role == Role.ASSISTANT else "user"

            # Build content blocks
            content_blocks = []
            for p in msg.content:
                if p.kind == ContentKind.TEXT and p.text:
                    content_blocks.append({"type": "text", "text": p.text})
                elif p.kind == ContentKind.IMAGE and p.image:
                    if p.image.base64_data:
                        content_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": p.image.media_type,
                                "data": p.image.base64_data,
                            },
                        })
                    elif p.image.url:
                        content_blocks.append({
                            "type": "image",
                            "source": {"type": "url", "url": p.image.url},
                        })
                elif p.kind == ContentKind.TOOL_CALL and p.tool_call:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": p.tool_call.id,
                        "name": p.tool_call.name,
                        "input": json.loads(p.tool_call.arguments)
                        if isinstance(p.tool_call.arguments, str)
                        else p.tool_call.arguments,
                    })
                elif p.kind == ContentKind.THINKING and p.thinking:
                    block: dict[str, Any] = {
                        "type": "thinking",
                        "thinking": p.thinking.text,
                    }
                    if p.thinking.signature:
                        block["signature"] = p.thinking.signature
                    content_blocks.append(block)

            if not content_blocks:
                content_blocks = [{"type": "text", "text": ""}]

            result.append({"role": role, "content": content_blocks})

        return result

    def _build_tools(self, request: Request) -> list[dict[str, Any]] | None:
        if not request.tools:
            return None
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in request.tools
        ]

    def complete(self, request: Request) -> Response:
        client = self._ensure_client()

        system_text, non_system = self._extract_system(request.messages)

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": self._build_messages(non_system),
            "max_tokens": request.max_tokens or 4096,
        }

        if system_text:
            kwargs["system"] = system_text

        tools = self._build_tools(request)
        if tools:
            kwargs["tools"] = tools
        if request.tool_choice:
            tc = request.tool_choice
            if hasattr(tc, "value"):
                kwargs["tool_choice"] = {"type": tc.value}
            else:
                kwargs["tool_choice"] = {"type": str(tc)}

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop_sequences:
            kwargs["stop_sequences"] = request.stop_sequences

        # Provider options (thinking, beta headers, etc.)
        if request.provider_options and "anthropic" in request.provider_options:
            anthro_opts = request.provider_options["anthropic"]
            if "thinking" in anthro_opts:
                kwargs["thinking"] = anthro_opts["thinking"]
            if "beta_features" in anthro_opts:
                # Beta features are passed as extra headers
                pass

        try:
            raw_resp = client.messages.create(**kwargs)
        except Exception as e:
            self._handle_error(e)

        return self._parse_response(raw_resp)

    def _parse_response(self, raw: Any) -> Response:
        parts: list[ContentPart] = []

        for block in raw.content:
            if block.type == "text":
                parts.append(ContentPart.text_part(block.text))
            elif block.type == "tool_use":
                parts.append(ContentPart(
                    kind=ContentKind.TOOL_CALL,
                    tool_call=ToolCallData(
                        id=block.id,
                        name=block.name,
                        arguments=json.dumps(block.input),
                    ),
                ))
            elif block.type == "thinking":
                sig = getattr(block, "signature", None)
                parts.append(ContentPart(
                    kind=ContentKind.THINKING,
                    thinking=ThinkingData(
                        text=getattr(block, "thinking", ""),
                        signature=sig,
                    ),
                ))

        msg = Message(role=Role.ASSISTANT, content=parts)

        # Finish reason
        stop_map = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        raw_sr = raw.stop_reason or "end_turn"
        fr = FinishReason(
            reason=stop_map.get(raw_sr, raw_sr),
            raw=raw_sr,
        )

        # Usage
        usage = Usage(
            input_tokens=raw.usage.input_tokens if raw.usage else 0,
            output_tokens=raw.usage.output_tokens if raw.usage else 0,
        )
        if raw.usage and hasattr(raw.usage, "cache_read_input_tokens"):
            usage.cache_read_tokens = raw.usage.cache_read_input_tokens or 0
        if raw.usage and hasattr(raw.usage, "cache_creation_input_tokens"):
            usage.cache_write_tokens = raw.usage.cache_creation_input_tokens or 0

        return Response(
            id=raw.id or "",
            model=raw.model or "",
            provider="anthropic",
            message=msg,
            finish_reason=fr,
            usage=usage,
        )

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        client = self._ensure_client()

        system_text, non_system = self._extract_system(request.messages)

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": self._build_messages(non_system),
            "max_tokens": request.max_tokens or 4096,
            "stream": True,
        }
        if system_text:
            kwargs["system"] = system_text

        tools = self._build_tools(request)
        if tools:
            kwargs["tools"] = tools
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        try:
            stream_resp = client.messages.create(**kwargs)
        except Exception as e:
            self._handle_error(e)

        yield StreamEvent(type=StreamEventType.STREAM_START)

        accumulated_text = ""
        with stream_resp as stream:
            for event in stream:
                if hasattr(event, "type"):
                    if event.type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta and hasattr(delta, "text"):
                            accumulated_text += delta.text
                            yield StreamEvent(
                                type=StreamEventType.TEXT_DELTA,
                                delta=delta.text,
                            )

        yield StreamEvent(
            type=StreamEventType.FINISH,
            response=Response(
                id="",
                model=request.model,
                provider="anthropic",
                message=Message.assistant(accumulated_text),
                finish_reason=FinishReason(reason="stop"),
            ),
        )

    def _handle_error(self, e: Exception) -> None:
        try:
            import anthropic as anthropic_mod
            if isinstance(e, anthropic_mod.AuthenticationError):
                raise AuthenticationError(str(e), provider="anthropic", cause=e)
            if isinstance(e, anthropic_mod.APIStatusError):
                raise error_from_status(
                    e.status_code, str(e), provider="anthropic",
                )
        except ImportError:
            pass
        raise e
