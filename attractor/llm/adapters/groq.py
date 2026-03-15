"""Groq provider adapter using the OpenAI-compatible Chat Completions API."""

from __future__ import annotations

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
    ToolCallData,
    Usage,
)

# Groq defaults
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqAdapter(ProviderAdapter):
    """Adapter for Groq's OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self._api_key:
            raise ConfigurationError("GROQ_API_KEY not set")
        self._base_url = base_url or os.environ.get("GROQ_BASE_URL", GROQ_BASE_URL)
        self._timeout = timeout
        self._client: Any = None

    @property
    def provider_id(self) -> str:
        return "groq"

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from groq import Groq

                self._client = Groq(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    timeout=self._timeout,
                )
            except ImportError:
                raise ConfigurationError(
                    "groq package not installed. Install with: pip install groq"
                )
        return self._client

    def _map_role(self, role: Role) -> str:
        mapping = {
            Role.SYSTEM: "system",
            Role.USER: "user",
            Role.ASSISTANT: "assistant",
            Role.TOOL: "tool",
            Role.DEVELOPER: "developer",
        }
        return mapping.get(role, "user")

    def _build_content(self, msg: Message) -> str | list[dict[str, Any]]:
        """Build content field for a message."""
        text_parts = [
            p.text for p in msg.content
            if p.kind == ContentKind.TEXT and p.text is not None
        ]
        image_parts = [p for p in msg.content if p.kind == ContentKind.IMAGE]

        if not image_parts:
            return "\n".join(text_parts) if text_parts else ""

        parts: list[dict[str, Any]] = []
        for t in text_parts:
            parts.append({"type": "text", "text": t})
        for ip in image_parts:
            if ip.image and ip.image.url:
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": ip.image.url},
                })
            elif ip.image and ip.image.base64_data:
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{ip.image.media_type};base64,{ip.image.base64_data}"
                    },
                })
        return parts

    def _build_messages(self, request: Request) -> list[dict[str, Any]]:
        """Convert unified Messages to Groq/OpenAI format."""
        result: list[dict[str, Any]] = []
        for msg in request.messages:
            role_str = self._map_role(msg.role)
            content = self._build_content(msg)

            entry: dict[str, Any] = {"role": role_str}
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id

            if msg.role == Role.TOOL:
                text_parts = [
                    p.tool_result.content if p.tool_result else (p.text or "")
                    for p in msg.content
                ]
                entry["content"] = "\n".join(text_parts)
                result.append(entry)
                continue

            tool_calls = [p for p in msg.content if p.kind == ContentKind.TOOL_CALL]
            if tool_calls:
                tc_list = []
                for p in tool_calls:
                    if p.tool_call:
                        tc_list.append({
                            "id": p.tool_call.id,
                            "type": "function",
                            "function": {
                                "name": p.tool_call.name,
                                "arguments": p.tool_call.arguments,
                            },
                        })
                entry["tool_calls"] = tc_list
                text_parts = [
                    p.text for p in msg.content
                    if p.kind == ContentKind.TEXT and p.text
                ]
                entry["content"] = "\n".join(text_parts) if text_parts else None
            elif isinstance(content, str):
                entry["content"] = content
            else:
                entry["content"] = content

            result.append(entry)
        return result

    def _build_tools(self, request: Request) -> list[dict[str, Any]] | None:
        if not request.tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in request.tools
        ]

    def complete(self, request: Request) -> Response:
        client = self._ensure_client()

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": self._build_messages(request),
        }

        tools = self._build_tools(request)
        if tools:
            kwargs["tools"] = tools
        if request.tool_choice:
            tc = request.tool_choice
            kwargs["tool_choice"] = tc.value if hasattr(tc, "value") else tc

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.stop_sequences:
            kwargs["stop"] = request.stop_sequences
        if request.reasoning_effort:
            kwargs["reasoning_effort"] = request.reasoning_effort

        if request.response_format and request.response_format.type == "json":
            kwargs["response_format"] = {"type": "json_object"}

        if request.provider_options and "groq" in request.provider_options:
            kwargs.update(request.provider_options["groq"])

        try:
            raw_resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            self._handle_error(e)

        return self._parse_response(raw_resp)

    def _parse_response(self, raw: Any) -> Response:
        choice = raw.choices[0] if raw.choices else None
        if not choice:
            return Response(
                id=getattr(raw, "id", "") or "",
                model=getattr(raw, "model", "") or "",
                provider="groq",
                usage=self._parse_usage(getattr(raw, "usage", None)),
            )

        parts: list[ContentPart] = []

        if choice.message.content:
            parts.append(ContentPart.text_part(choice.message.content))

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                parts.append(ContentPart(
                    kind=ContentKind.TOOL_CALL,
                    tool_call=ToolCallData(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                ))

        msg = Message(role=Role.ASSISTANT, content=parts)

        fr_raw = choice.finish_reason or "stop"
        fr_map = {
            "stop": "stop",
            "tool_calls": "tool_calls",
            "length": "length",
            "content_filter": "content_filter",
        }
        fr = FinishReason(reason=fr_map.get(fr_raw, fr_raw), raw=fr_raw)

        return Response(
            id=getattr(raw, "id", "") or "",
            model=getattr(raw, "model", "") or "",
            provider="groq",
            message=msg,
            finish_reason=fr,
            usage=self._parse_usage(getattr(raw, "usage", None)),
        )

    def _parse_usage(self, usage: Any) -> Usage:
        if not usage:
            return Usage()
        reasoning = 0
        if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            reasoning = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
        return Usage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            reasoning_tokens=reasoning,
        )

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        client = self._ensure_client()

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": self._build_messages(request),
            "stream": True,
        }
        tools = self._build_tools(request)
        if tools:
            kwargs["tools"] = tools
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens

        try:
            stream_resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            self._handle_error(e)

        yield StreamEvent(type=StreamEventType.STREAM_START)

        accumulated_text = ""
        for chunk in stream_resp:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                accumulated_text += delta.content
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    delta=delta.content,
                )

        yield StreamEvent(
            type=StreamEventType.FINISH,
            response=Response(
                id="",
                model=request.model,
                provider="groq",
                message=Message.assistant(accumulated_text),
                finish_reason=FinishReason(reason="stop"),
            ),
        )

    def _handle_error(self, e: Exception) -> None:
        """Translate Groq SDK exceptions to our error hierarchy when possible."""
        try:
            import groq as groq_mod

            auth_error = getattr(groq_mod, "AuthenticationError", None)
            status_error = getattr(groq_mod, "APIStatusError", None)

            if auth_error and isinstance(e, auth_error):
                raise AuthenticationError(str(e), provider="groq", cause=e)
            if status_error and isinstance(e, status_error):
                status = getattr(e, "status_code", None) or getattr(e, "status", None)
                raise error_from_status(status, str(e), provider="groq")
        except ImportError:
            pass
        raise e
