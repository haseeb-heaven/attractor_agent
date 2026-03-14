"""High-level generation API — convenience functions wrapping the Client."""

from __future__ import annotations

import json
from typing import Any, Type, TypeVar

from attractor.llm.client import Client, get_default_client
from attractor.llm.errors import NoObjectGeneratedError
from attractor.llm.retry import RetryConfig, retry_sync, RETRY_STANDARD
from attractor.llm.types import (
    Message,
    Request,
    Response,
    ResponseFormat,
    ToolDefinition,
)

T = TypeVar("T")


def generate(
    prompt: str | None = None,
    *,
    messages: list[Message] | None = None,
    model: str = "",
    provider: str | None = None,
    tools: list[ToolDefinition] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    retry: RetryConfig | None = None,
    client: Client | None = None,
    **kwargs: Any,
) -> Response:
    """High-level generation function.

    Either `prompt` (converted to a single user message) or `messages`
    must be provided, but not both.
    """
    if prompt and messages:
        raise ValueError("Provide either 'prompt' or 'messages', not both.")

    if prompt:
        msgs = [Message.user(prompt)]
    elif messages:
        msgs = messages
    else:
        raise ValueError("Either 'prompt' or 'messages' must be provided.")

    request = Request(
        model=model,
        messages=msgs,
        provider=provider,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )

    c = client or get_default_client()
    config = retry or RETRY_STANDARD

    return retry_sync(c.complete, config, request)


def generate_object(
    prompt: str | None = None,
    *,
    messages: list[Message] | None = None,
    model: str = "",
    provider: str | None = None,
    schema: dict[str, Any] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    retry: RetryConfig | None = None,
    client: Client | None = None,
) -> dict[str, Any]:
    """Generate structured output conforming to a JSON schema.

    Returns the parsed JSON object. Raises NoObjectGeneratedError on failure.
    """
    if prompt and messages:
        raise ValueError("Provide either 'prompt' or 'messages', not both.")

    if prompt:
        msgs = [Message.user(prompt)]
    elif messages:
        msgs = messages
    else:
        raise ValueError("Either 'prompt' or 'messages' must be provided.")

    response_format = None
    if schema:
        response_format = ResponseFormat(type="json_schema", schema=schema)
    else:
        response_format = ResponseFormat(type="json")

    request = Request(
        model=model,
        messages=msgs,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )

    c = client or get_default_client()
    config = retry or RETRY_STANDARD

    response = retry_sync(c.complete, config, request)

    try:
        return json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as e:
        raise NoObjectGeneratedError(
            f"Failed to parse structured output: {e}",
            cause=e,
        )
