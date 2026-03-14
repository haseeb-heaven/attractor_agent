"""Base handler interface and handler registry."""

from __future__ import annotations

import abc
from typing import Any

from attractor.pipeline.context import Context, Outcome
from attractor.pipeline.events import EventEmitter
from attractor.pipeline.graph import Graph, Node


class Handler(abc.ABC):
    """Common interface all node handlers implement."""

    @abc.abstractmethod
    def execute(
        self,
        node: Node,
        context: Context,
        graph: Graph,
        emitter: EventEmitter,
        **kwargs: Any,
    ) -> Outcome:
        """Execute the handler for a given node.

        Args:
            node: The node being executed.
            context: Shared pipeline context.
            graph: The full pipeline graph (for inspecting outgoing edges, etc.).
            emitter: Event emitter for emitting handler events.
            **kwargs: Additional dependencies (interviewer, llm_client, etc.).

        Returns:
            Outcome with status and optional routing hints.
        """
        ...


class HandlerRegistry:
    """Registry mapping handler type strings to Handler instances."""

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}
        self._default: Handler | None = None

    def register(self, handler_type: str, handler: Handler) -> None:
        """Register a handler for a given type string."""
        self._handlers[handler_type] = handler

    def set_default(self, handler: Handler) -> None:
        """Set a default handler for unrecognized types."""
        self._default = handler

    def get(self, handler_type: str) -> Handler | None:
        """Get the handler for a given type. Falls back to default."""
        return self._handlers.get(handler_type, self._default)

    def has(self, handler_type: str) -> bool:
        return handler_type in self._handlers
