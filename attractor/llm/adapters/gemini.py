"""Gemini provider adapter using the Google Generative AI SDK."""

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


class GeminiAdapter(ProviderAdapter):
    """Adapter for Google's Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self._api_key = api_key or os.environ.get(
            "GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY")
        )
        if not self._api_key:
            raise ConfigurationError("GEMINI_API_KEY not set")
        self._timeout = timeout
        self._configured = False

    def _ensure_configured(self) -> None:
        if not self._configured:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self._api_key)
                self._configured = True
            except ImportError:
                raise ConfigurationError(
                    "google-generativeai package not installed. "
                    "Install with: pip install google-generativeai"
                )

    @property
    def provider_id(self) -> str:
        return "gemini"

    def _extract_system(self, messages: list[Message]) -> tuple[str, list[Message]]:
        """Extract system messages for systemInstruction."""
        system_parts: list[str] = []
        non_system: list[Message] = []
        for msg in messages:
            if msg.role in (Role.SYSTEM, Role.DEVELOPER):
                system_parts.append(msg.text)
            else:
                non_system.append(msg)
        return "\n\n".join(system_parts), non_system

    def _build_contents(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert unified Messages to Gemini content format."""
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = "model" if msg.role == Role.ASSISTANT else "user"
            parts: list[dict[str, Any]] = []

            for p in msg.content:
                if p.kind == ContentKind.TEXT and p.text:
                    parts.append({"text": p.text})
                elif p.kind == ContentKind.IMAGE and p.image:
                    if p.image.base64_data:
                        parts.append({
                            "inline_data": {
                                "mime_type": p.image.media_type,
                                "data": p.image.base64_data,
                            }
                        })
                elif p.kind == ContentKind.TOOL_CALL and p.tool_call:
                    try:
                        args = json.loads(p.tool_call.arguments) if isinstance(
                            p.tool_call.arguments, str
                        ) else p.tool_call.arguments
                    except json.JSONDecodeError:
                        args = {}
                    parts.append({
                        "functionCall": {
                            "name": p.tool_call.name,
                            "args": args,
                        }
                    })
                elif p.kind == ContentKind.TOOL_RESULT and p.tool_result:
                    parts.append({
                        "functionResponse": {
                            "name": p.tool_result.tool_call_id,
                            "response": {"result": p.tool_result.content},
                        }
                    })

            if not parts:
                parts = [{"text": ""}]

            contents.append({"role": role, "parts": parts})
        return contents

    def _build_tools(self, request: Request) -> list[dict[str, Any]] | None:
        if not request.tools:
            return None
        declarations = []
        for t in request.tools:
            decl: dict[str, Any] = {
                "name": t.name,
                "description": t.description,
            }
            if t.parameters:
                decl["parameters"] = t.parameters
            declarations.append(decl)
        return [{"function_declarations": declarations}]

    def complete(self, request: Request) -> Response:
        self._ensure_configured()
        import google.generativeai as genai

        system_text, non_system = self._extract_system(request.messages)

        gen_config: dict[str, Any] = {}
        if request.temperature is not None:
            gen_config["temperature"] = request.temperature
        if request.top_p is not None:
            gen_config["top_p"] = request.top_p
        if request.max_tokens is not None:
            gen_config["max_output_tokens"] = request.max_tokens
        if request.stop_sequences:
            gen_config["stop_sequences"] = request.stop_sequences

        model_kwargs: dict[str, Any] = {"model_name": request.model}
        if system_text:
            model_kwargs["system_instruction"] = system_text
        if gen_config:
            model_kwargs["generation_config"] = gen_config

        tools = self._build_tools(request)
        if tools:
            model_kwargs["tools"] = tools

        try:
            model = genai.GenerativeModel(**model_kwargs)
            contents = self._build_contents(non_system)
            raw_resp = model.generate_content(contents)
        except Exception as e:
            self._handle_error(e)

        return self._parse_response(raw_resp, request.model)

    def _parse_response(self, raw: Any, model_name: str) -> Response:
        parts: list[ContentPart] = []

        if raw.candidates:
            candidate = raw.candidates[0]
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    parts.append(ContentPart.text_part(part.text))
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    parts.append(ContentPart(
                        kind=ContentKind.TOOL_CALL,
                        tool_call=ToolCallData(
                            id=fc.name,  # Gemini uses name as ID
                            name=fc.name,
                            arguments=json.dumps(dict(fc.args) if fc.args else {}),
                        ),
                    ))

        msg = Message(role=Role.ASSISTANT, content=parts)

        # Usage
        usage = Usage()
        if hasattr(raw, "usage_metadata") and raw.usage_metadata:
            um = raw.usage_metadata
            usage.input_tokens = getattr(um, "prompt_token_count", 0) or 0
            usage.output_tokens = getattr(um, "candidates_token_count", 0) or 0
            usage.reasoning_tokens = getattr(um, "thoughts_token_count", 0) or 0

        # Finish reason
        fr_raw = "stop"
        if raw.candidates:
            fr_map = {1: "stop", 2: "length", 3: "content_filter", 4: "content_filter"}
            raw_fr = getattr(raw.candidates[0], "finish_reason", 1)
            if isinstance(raw_fr, int):
                fr_raw = fr_map.get(raw_fr, "stop")
            else:
                fr_raw = str(raw_fr)

        has_tool_calls = any(
            p.kind == ContentKind.TOOL_CALL for p in parts
        )
        reason = "tool_calls" if has_tool_calls else fr_raw

        return Response(
            id="",
            model=model_name,
            provider="gemini",
            message=msg,
            finish_reason=FinishReason(reason=reason, raw=fr_raw),
            usage=usage,
        )

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        self._ensure_configured()
        import google.generativeai as genai

        system_text, non_system = self._extract_system(request.messages)

        model_kwargs: dict[str, Any] = {"model_name": request.model}
        if system_text:
            model_kwargs["system_instruction"] = system_text

        gen_config: dict[str, Any] = {}
        if request.temperature is not None:
            gen_config["temperature"] = request.temperature
        if request.max_tokens is not None:
            gen_config["max_output_tokens"] = request.max_tokens
        if gen_config:
            model_kwargs["generation_config"] = gen_config

        tools = self._build_tools(request)
        if tools:
            model_kwargs["tools"] = tools

        try:
            model = genai.GenerativeModel(**model_kwargs)
            contents = self._build_contents(non_system)
            stream_resp = model.generate_content(contents, stream=True)
        except Exception as e:
            self._handle_error(e)

        yield StreamEvent(type=StreamEventType.STREAM_START)

        accumulated_text = ""
        for chunk in stream_resp:
            if chunk.text:
                accumulated_text += chunk.text
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    delta=chunk.text,
                )

        yield StreamEvent(
            type=StreamEventType.FINISH,
            response=Response(
                id="",
                model=request.model,
                provider="gemini",
                message=Message.assistant(accumulated_text),
                finish_reason=FinishReason(reason="stop"),
            ),
        )

    def _handle_error(self, e: Exception) -> None:
        try:
            from google.api_core import exceptions as google_exceptions
            if isinstance(e, google_exceptions.Unauthenticated):
                raise AuthenticationError(str(e), provider="gemini", cause=e)
            if isinstance(e, google_exceptions.GoogleAPIError):
                code = getattr(e, "code", 500)
                raise error_from_status(code, str(e), provider="gemini")
        except ImportError:
            pass
        raise e
