"""OpenRouter provider adapter — OpenAI-compatible API with model routing.

OpenRouter (https://openrouter.ai) provides a unified gateway to 200+ LLMs
via an OpenAI-compatible Chat Completions endpoint. This adapter extends
the OpenAI adapter pattern but targets OpenRouter's base URL and handles
its specific headers and response metadata.
"""

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

# Default OpenRouter base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default model to use when none is specified
OPENROUTER_DEFAULT_MODEL = "openrouter/free"


class OpenRouterAdapter(ProviderAdapter):
    """Adapter for OpenRouter's OpenAI-compatible API.

    OpenRouter routes requests to 200+ models from OpenAI, Anthropic,
    Google, Meta, Mistral, and more through a single API key.

    Usage:
        adapter = OpenRouterAdapter()  # reads OPENROUTER_API_KEY from env
        response = adapter.complete(Request(model="google/gemini-2.5-pro-exp-03-25:free", ...))
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        app_name: str | None = None,
        app_url: str | None = None,
        timeout: float = 120.0,
    ):
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ConfigurationError("OPENROUTER_API_KEY not set")
        self._base_url = base_url or os.environ.get(
            "OPENROUTER_BASE_URL", OPENROUTER_BASE_URL
        )
        self._default_model = default_model or os.environ.get(
            "OPENROUTER_DEFAULT_MODEL", OPENROUTER_DEFAULT_MODEL
        )
        self._app_name = app_name or os.environ.get("OPENROUTER_APP_NAME", "Attractor")
        self._app_url = app_url or os.environ.get("OPENROUTER_APP_URL", "")
        self._timeout = timeout
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    timeout=self._timeout,
                    default_headers=self._build_extra_headers(),
                )
            except ImportError:
                raise ConfigurationError(
                    "openai package not installed. Install with: pip install openai"
                )
        return self._client

    def _build_extra_headers(self) -> dict[str, str]:
        """Build OpenRouter-specific HTTP headers."""
        headers: dict[str, str] = {}
        if self._app_name:
            headers["X-Title"] = self._app_name
        if self._app_url:
            headers["HTTP-Referer"] = self._app_url
        return headers

    @property
    def provider_id(self) -> str:
        return "openrouter"

    def _build_messages(self, request: Request) -> list[dict[str, Any]]:
        """Convert unified Messages to OpenAI/OpenRouter format."""
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
        client = self._ensure_client()

        # Use default model if none specified
        model = request.model or self._default_model

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._build_messages(request),
			"stream": True,   # ← ADD THIS — llmock always streams
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

        if request.response_format and request.response_format.type == "json":
            kwargs["response_format"] = {"type": "json_object"}

        # Provider options escape hatch
        if request.provider_options and "openrouter" in request.provider_options:
            kwargs.update(request.provider_options["openrouter"])

        try:
            raw_resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            self._handle_error(e)

        return self._parse_response(raw_resp)

    def _parse_response(self, raw: Any) -> Response:
        choice = raw.choices[0] if raw.choices else None
        if not choice:
            return Response(
                id=raw.id or "",
                model=raw.model or "",
                provider="openrouter",
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
            provider="openrouter",
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
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            reasoning_tokens=reasoning,
        )

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        client = self._ensure_client()

        model = request.model or self._default_model

        kwargs: dict[str, Any] = {
            "model": model,
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
                model=model,
                provider="openrouter",
                message=Message.assistant(accumulated_text),
                finish_reason=FinishReason(reason="stop"),
            ),
        )

    def _handle_error(self, e: Exception) -> None:
        """Translate OpenAI SDK exceptions to our error hierarchy."""
        try:
            import openai as openai_mod
            if isinstance(e, openai_mod.AuthenticationError):
                raise AuthenticationError(str(e), provider="openrouter", cause=e)
            if isinstance(e, openai_mod.APIStatusError):
                raise error_from_status(
                    e.status_code, str(e), provider="openrouter",
                )
        except ImportError:
            pass
        raise e
