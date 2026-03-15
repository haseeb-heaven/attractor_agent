"""OpenAI provider adapter using the Chat Completions / Responses API."""

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
    ToolCallData,
    Usage,
)


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI's API (Chat Completions)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        timeout: float = 120.0,
    ):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ConfigurationError("OPENAI_API_KEY not set")
        self._base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self._organization = organization or os.environ.get("OPENAI_ORG_ID")
        self._timeout = timeout
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    organization=self._organization,
                    timeout=self._timeout,
                )
            except ImportError:
                raise ConfigurationError(
                    "openai package not installed. Install with: pip install openai"
                )
        return self._client

    @property
    def provider_id(self) -> str:
        return "openai"

    def _build_messages(self, request: Request) -> list[dict[str, Any]]:
        """Convert unified Messages to OpenAI format."""
        result: list[dict[str, Any]] = []
        for msg in request.messages:
            role_str = self._map_role(msg.role)
            content = self._build_content(msg)

            entry: dict[str, Any] = {"role": role_str}
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id

            # Tool call results
            if msg.role == Role.TOOL:
                text_parts = [
                    p.tool_result.content if p.tool_result else (p.text or "")
                    for p in msg.content
                ]
                entry["content"] = "\n".join(text_parts)
                result.append(entry)
                continue

            # Check for tool calls in assistant messages
            tool_calls = [
                p for p in msg.content if p.kind == ContentKind.TOOL_CALL
            ]
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
                # Also include text if present
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

    def _build_content(self, msg: Message) -> str | list[dict[str, Any]]:
        """Build content field for a message."""
        text_parts = [
            p.text for p in msg.content
            if p.kind == ContentKind.TEXT and p.text is not None
        ]
        image_parts = [
            p for p in msg.content if p.kind == ContentKind.IMAGE
        ]

        if not image_parts:
            return "\n".join(text_parts) if text_parts else ""

        # Multimodal
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

    def _map_role(self, role: Role) -> str:
        mapping = {
            Role.SYSTEM: "system",
            Role.USER: "user",
            Role.ASSISTANT: "assistant",
            Role.TOOL: "tool",
            Role.DEVELOPER: "developer",
        }
        return mapping.get(role, "user")

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
        """Send a blocking completion request.

        Uses stream=True internally to handle providers (like llmock) that
        always respond with text/event-stream, then reassembles chunks into
        a single Response — identical to non-streaming from the caller's view.
        """
        client = self._ensure_client()

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": self._build_messages(request),
            "stream": True,  # ← works with both real OpenAI and llmock
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
        if request.provider_options and "openai" in request.provider_options:
            kwargs.update(request.provider_options["openai"])

        try:
            stream_resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            self._handle_error(e)

        # ── Consume all SSE chunks and reassemble into a full Response ────────
        accumulated_text = ""
        raw_id = ""
        raw_model = request.model
        tool_calls_map: dict[int, dict[str, Any]] = {}

        try:
            for chunk in stream_resp:
                if not raw_id and hasattr(chunk, "id") and chunk.id:
                    raw_id = chunk.id
                if hasattr(chunk, "model") and chunk.model:
                    raw_model = chunk.model
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if not delta:
                    continue
                # Accumulate text
                if delta.content:
                    accumulated_text += delta.content
                # Accumulate tool calls
                if delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        idx = tc_chunk.index
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_chunk.id:
                        tool_calls_map[idx]["id"] += tc_chunk.id
                    if tc_chunk.function:
                        if tc_chunk.function.name:
                            tool_calls_map[idx]["name"] += tc_chunk.function.name
                        if tc_chunk.function.arguments:
                            tool_calls_map[idx]["arguments"] += tc_chunk.function.arguments
        except Exception as e:
            self._handle_error(e)

        # ── Build unified Response ────────────────────────────────────────────
        parts: list[ContentPart] = []
        if accumulated_text:
            parts.append(ContentPart.text_part(accumulated_text))
        for tc_data in tool_calls_map.values():
            parts.append(ContentPart(
                kind=ContentKind.TOOL_CALL,
                tool_call=ToolCallData(
                    id=tc_data["id"],
                    name=tc_data["name"],
                    arguments=tc_data["arguments"],
                ),
            ))

        return Response(
            id=raw_id,
            model=raw_model,
            provider="openai",
            message=Message(role=Role.ASSISTANT, content=parts),
            finish_reason=FinishReason(reason="stop"),
            usage=Usage(),
        )


    def _parse_response(self, raw: Any) -> Response:
        choice = raw.choices[0] if raw.choices else None
        if not choice:
            return Response(
                id=raw.id or "",
                model=raw.model or "",
                provider="openai",
                usage=self._parse_usage(raw.usage),
            )

        # Build message content
        parts: list[ContentPart] = []

        # Text content
        if choice.message.content:
            parts.append(ContentPart.text_part(choice.message.content))

        # Tool calls
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

        # Finish reason
        fr_raw = choice.finish_reason or "stop"
        fr_map = {
            "stop": "stop",
            "tool_calls": "tool_calls",
            "length": "length",
            "content_filter": "content_filter",
        }
        fr = FinishReason(reason=fr_map.get(fr_raw, fr_raw), raw=fr_raw)

        return Response(
            id=raw.id or "",
            model=raw.model or "",
            provider="openai",
            message=msg,
            finish_reason=fr,
            usage=self._parse_usage(raw.usage),
        )

    def _parse_usage(self, usage: Any) -> Usage:
        if not usage:
            return Usage()
        reasoning = 0
        if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            reasoning = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
        return Usage(
            input_tokens=usage.prompt_tokens or 0,
            output_tokens=usage.completion_tokens or 0,
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
            if not chunk.choices:
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
                provider="openai",
                message=Message.assistant(accumulated_text),
                finish_reason=FinishReason(reason="stop"),
            ),
        )

    def _handle_error(self, e: Exception) -> None:
        """Translate OpenAI SDK exceptions to our error hierarchy."""
        try:
            import openai as openai_mod
            if isinstance(e, openai_mod.AuthenticationError):
                raise AuthenticationError(str(e), provider="openai", cause=e)
            if isinstance(e, openai_mod.APIStatusError):
                raise error_from_status(
                    e.status_code, str(e), provider="openai",
                )
        except ImportError:
            pass
        raise e
