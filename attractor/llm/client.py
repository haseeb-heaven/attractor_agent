"""Unified LLM Client — routes requests to provider adapters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, AsyncIterator, Callable

# Auto-load .env file from the project root (or CWD)
try:
    from dotenv import load_dotenv as _load_dotenv

    # Walk up from this file to find .env in the project root
    _project_root = Path(__file__).resolve().parents[2]
    _env_file = _project_root / ".env"
    if _env_file.exists():
        _load_dotenv(_env_file)
    else:
        # Fallback: try CWD
        _load_dotenv()
except ImportError:
    pass  # python-dotenv is optional but recommended

from attractor.llm.adapters.base import ProviderAdapter
from attractor.llm.errors import ConfigurationError
from attractor.llm.types import Request, Response, StreamEvent

# Type for middleware: takes request, returns response
Middleware = Callable[[Request, Callable[[Request], Response]], Response]


class Client:
    """Main orchestration layer routing requests to provider adapters.

    Usage:
        client = Client.from_env()
        response = client.complete(Request(model="gpt-4o", messages=[...]))
    """

    def __init__(
        self,
        providers: dict[str, ProviderAdapter] | None = None,
        default_provider: str | None = None,
    ):
        self._providers: dict[str, ProviderAdapter] = providers or {}
        self._default_provider = default_provider
        self._middleware: list[Middleware] = []

        # If no default set, use first provider
        if not self._default_provider and self._providers:
            self._default_provider = next(iter(self._providers))

    @classmethod
    def from_env(cls) -> Client:
        """Create a Client by auto-detecting API keys from environment variables.

        Provider detection order (first found becomes the default):
          1. OpenRouter  (OPENROUTER_API_KEY)
          2. OpenAI      (OPENAI_API_KEY)
          3. Anthropic   (ANTHROPIC_API_KEY)
          4. Gemini      (GEMINI_API_KEY / GOOGLE_API_KEY)
        """
        providers: dict[str, ProviderAdapter] = {}

        # Try OpenRouter first — makes it the default when present
        if os.environ.get("OPENROUTER_API_KEY"):
            try:
                from attractor.llm.adapters.openrouter import OpenRouterAdapter
                providers["openrouter"] = OpenRouterAdapter()
            except Exception:
                pass

        # Try OpenAI
        if os.environ.get("OPENAI_API_KEY"):
            try:
                from attractor.llm.adapters.openai import OpenAIAdapter
                providers["openai"] = OpenAIAdapter()
            except Exception:
                pass

        # Try Anthropic
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                from attractor.llm.adapters.anthropic import AnthropicAdapter
                providers["anthropic"] = AnthropicAdapter()
            except Exception:
                pass

        # Try Gemini
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            try:
                from attractor.llm.adapters.gemini import GeminiAdapter
                providers["gemini"] = GeminiAdapter()
            except Exception:
                pass

        if not providers:
            raise ConfigurationError(
                "No LLM provider API keys found in environment. "
                "Set OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "or GEMINI_API_KEY.  You can place them in a .env file at "
                "the project root."
            )

        return cls(providers=providers)

    def register_provider(self, provider_id: str, adapter: ProviderAdapter) -> None:
        """Register or replace a provider adapter."""
        self._providers[provider_id] = adapter
        if not self._default_provider:
            self._default_provider = provider_id

    def add_middleware(self, middleware: Middleware) -> None:
        """Add a middleware function to the request/response pipeline."""
        self._middleware.append(middleware)

    def _resolve_provider(self, request: Request) -> ProviderAdapter:
        """Determine which provider adapter handles this request."""
        provider_key = request.provider or self._default_provider
        if not provider_key:
            raise ConfigurationError(
                "No provider specified and no default provider configured."
            )
        if provider_key not in self._providers:
            raise ConfigurationError(
                f"Provider '{provider_key}' is not registered. "
                f"Available: {list(self._providers.keys())}"
            )
        return self._providers[provider_key]

    def complete(self, request: Request) -> Response:
        """Send a blocking request and return the full response.

        Routes to the appropriate provider adapter based on the request's
        provider field (or default provider).
        """
        adapter = self._resolve_provider(request)

        # Apply middleware chain
        def base_call(req: Request) -> Response:
            return adapter.complete(req)

        handler = base_call
        for mw in reversed(self._middleware):
            prev_handler = handler
            handler = lambda req, _mw=mw, _ph=prev_handler: _mw(req, _ph)

        return handler(request)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send a streaming request and yield events."""
        adapter = self._resolve_provider(request)
        async for event in adapter.stream(request):
            yield event


# Module-level default client
_default_client: Client | None = None


def set_default_client(client: Client) -> None:
    """Set the module-level default client."""
    global _default_client
    _default_client = client


def get_default_client() -> Client:
    """Get or create the module-level default client."""
    global _default_client
    if _default_client is None:
        _default_client = Client.from_env()
    return _default_client
