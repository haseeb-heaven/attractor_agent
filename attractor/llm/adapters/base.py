"""Base provider adapter interface."""

from __future__ import annotations

import abc
from typing import Any, AsyncIterator

from attractor.llm.types import Request, Response, StreamEvent


class ProviderAdapter(abc.ABC):
    """Contract that every provider adapter must implement."""

    @property
    @abc.abstractmethod
    def provider_id(self) -> str:
        """Unique provider identifier (e.g., 'litellm')."""
        ...

    @abc.abstractmethod
    def complete(self, request: Request) -> Response:
        """Send a blocking request and return the full response."""
        ...

    @abc.abstractmethod
    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send a streaming request and yield events."""
        ...
        # Make it an async generator
        yield  # type: ignore[misc]
