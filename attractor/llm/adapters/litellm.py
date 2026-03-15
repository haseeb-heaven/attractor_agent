"""LiteLLM provider adapter for chat completions and streaming."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

from attractor.llm.adapters.base import ProviderAdapter
from attractor.llm.errors import AuthenticationError, ConfigurationError, error_from_status
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


class LiteLLMAdapter(ProviderAdapter):
    """Adapter for LiteLLM's unified completion API."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str | None = None,
        timeout: float = 120.0,
    ):
        self._api_key = api_key or os.environ.get("LITELLM_API_KEY")
        self._api_base = api_base or os.environ.get("LITELLM_BASE_URL")
        self._default_model = default_model or os.environ.get("LITELLM_MODEL", "")
        self._timeout = timeout
        self._litellm: Any = None

    @property
    def provider_id(self) -> str:
        return "litellm"

    def _ensure_module(self) -> Any:
        if self._litellm is None:
            try:
                import litellm
            except ImportError as exc:
                raise ConfigurationError(
                    "litellm package not installed. Install with: pip install litellm"
                ) from exc
            self._litellm = litellm
        return self._litellm

    def _resolve_model(self, request: Request) -> str:
        model = request.model or self._default_model
        if not model:
            raise ConfigurationError(
                "No model provided. Set Request.model or LITELLM_MODEL."
            )
        # For proxy/openai-compatible endpoints, bare model names should route via openai.
        if self._api_base and "/" not in model:
            return f"openai/{model}"
        return model

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
        text_parts = [
            p.text for p in msg.content if p.kind == ContentKind.TEXT and p.text is not None
        ]
        image_parts = [p for p in msg.content if p.kind == ContentKind.IMAGE]
        if not image_parts:
            return "\n".join(text_parts) if text_parts else ""

        parts: list[dict[str, Any]] = []
        for text in text_parts:
            parts.append({"type": "text", "text": text})
        for image_part in image_parts:
            if image_part.image and image_part.image.url:
                parts.append(
                    {"type": "image_url", "image_url": {"url": image_part.image.url}}
                )
            elif image_part.image and image_part.image.base64_data:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_part.image.media_type};base64,{image_part.image.base64_data}"
                        },
                    }
                )
        return parts

    def _build_messages(self, request: Request) -> list[dict[str, Any]]:
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
                entry["tool_calls"] = [
                    {
                        "id": p.tool_call.id,
                        "type": "function",
                        "function": {
                            "name": p.tool_call.name,
                            "arguments": p.tool_call.arguments,
                        },
                    }
                    for p in tool_calls
                    if p.tool_call
                ]
                text_parts = [
                    p.text for p in msg.content if p.kind == ContentKind.TEXT and p.text
                ]
                entry["content"] = "\n".join(text_parts) if text_parts else None
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
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in request.tools
        ]

    def _base_kwargs(self, request: Request, stream: bool = False) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._resolve_model(request),
            "messages": self._build_messages(request),
            "stream": stream,
            "timeout": self._timeout,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
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

        tools = self._build_tools(request)
        if tools:
            kwargs["tools"] = tools
        if request.tool_choice:
            tool_choice = request.tool_choice
            kwargs["tool_choice"] = (
                tool_choice.value if hasattr(tool_choice, "value") else tool_choice
            )
        if request.response_format and request.response_format.type == "json":
            kwargs["response_format"] = {"type": "json_object"}

        if request.provider_options:
            if "litellm" in request.provider_options and isinstance(
                request.provider_options["litellm"], dict
            ):
                kwargs.update(request.provider_options["litellm"])
            else:
                kwargs.update(request.provider_options)
        return kwargs

    def _safe_get(self, obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _parse_usage(self, usage: Any) -> Usage:
        if not usage:
            return Usage()
        completion_tokens_details = self._safe_get(usage, "completion_tokens_details")
        reasoning_tokens = 0
        if completion_tokens_details:
            reasoning_tokens = (
                self._safe_get(completion_tokens_details, "reasoning_tokens", 0) or 0
            )
        return Usage(
            input_tokens=self._safe_get(usage, "prompt_tokens", 0) or 0,
            output_tokens=self._safe_get(usage, "completion_tokens", 0) or 0,
            reasoning_tokens=reasoning_tokens,
        )

    def _parse_response(self, raw: Any) -> Response:
        choices = self._safe_get(raw, "choices", [])
        choice = choices[0] if choices else None
        if not choice:
            return Response(
                id=self._safe_get(raw, "id", "") or "",
                model=self._safe_get(raw, "model", "") or "",
                provider="litellm",
                usage=self._parse_usage(self._safe_get(raw, "usage")),
            )

        message = self._safe_get(choice, "message")
        parts: list[ContentPart] = []
        content = self._safe_get(message, "content")
        if content:
            parts.append(ContentPart.text_part(content))

        tool_calls = self._safe_get(message, "tool_calls", [])
        for tool_call in tool_calls or []:
            function = self._safe_get(tool_call, "function")
            parts.append(
                ContentPart(
                    kind=ContentKind.TOOL_CALL,
                    tool_call=ToolCallData(
                        id=self._safe_get(tool_call, "id", "") or "",
                        name=self._safe_get(function, "name", "") or "",
                        arguments=self._safe_get(function, "arguments", "") or "",
                    ),
                )
            )

        finish_reason_raw = self._safe_get(choice, "finish_reason", "stop") or "stop"
        finish_reason = FinishReason(reason=finish_reason_raw, raw=finish_reason_raw)

        return Response(
            id=self._safe_get(raw, "id", "") or "",
            model=self._safe_get(raw, "model", "") or "",
            provider="litellm",
            message=Message(role=Role.ASSISTANT, content=parts),
            finish_reason=finish_reason,
            usage=self._parse_usage(self._safe_get(raw, "usage")),
        )

    def complete(self, request: Request) -> Response:
        litellm_mod = self._ensure_module()
        kwargs = self._base_kwargs(request, stream=False)
        try:
            raw_response = litellm_mod.completion(**kwargs)
        except Exception as exc:
            self._handle_error(exc)
        return self._parse_response(raw_response)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        litellm_mod = self._ensure_module()
        kwargs = self._base_kwargs(request, stream=True)
        try:
            stream_response = litellm_mod.completion(**kwargs)
        except Exception as exc:
            self._handle_error(exc)

        yield StreamEvent(type=StreamEventType.STREAM_START)
        accumulated_text = ""

        for chunk in stream_response:
            choices = self._safe_get(chunk, "choices", [])
            if not choices:
                continue
            delta = self._safe_get(choices[0], "delta")
            if not delta:
                continue
            text_delta = self._safe_get(delta, "content")
            if text_delta:
                accumulated_text += text_delta
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, delta=text_delta)

        yield StreamEvent(
            type=StreamEventType.FINISH,
            response=Response(
                model=self._resolve_model(request),
                provider="litellm",
                message=Message.assistant(accumulated_text),
                finish_reason=FinishReason(reason="stop", raw="stop"),
            ),
        )

    def _handle_error(self, exc: Exception) -> None:
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            if status_code == 401:
                raise AuthenticationError(str(exc), provider="litellm", cause=exc)
            raise error_from_status(status_code, str(exc), provider="litellm")
        raise exc
