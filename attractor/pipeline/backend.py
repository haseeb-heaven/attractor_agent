"""Pipeline backend implementation using the Unified LLM Client."""

from __future__ import annotations

from attractor.llm.client import Client, get_default_client
from attractor.llm.types import Message, Request
from attractor.pipeline.context import Context
from attractor.pipeline.graph import Node


class LLMBackend:
    """Backend for pipeline nodes that uses the Unified LLM Client."""

    def __init__(self, client: Client | None = None):
        self._client = client or get_default_client()

    def generate(self, prompt: str, node: Node, context: Context) -> str:
        """Execute a text generation request."""
        # Build simple message list
        messages = [Message.user(prompt)]
        
        # Create request
        request = Request(
            model=node.attrs.get("model", ""),  # Use model from DOT node if present
            messages=messages,
            temperature=float(node.attrs.get("temperature", 0.7)),
        )
        
        # Execute
        response = self._client.complete(request)
        return response.text
