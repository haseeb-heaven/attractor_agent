"""Unified LLM client that routes requests to provider adapters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncIterator, Callable

try:
    from dotenv import load_dotenv as _load_dotenv

    _project_root = Path(__file__).resolve().parents[2]
    _env_file = _project_root / ".env"
    if _env_file.exists():
        _load_dotenv(_env_file)
    else:
        _load_dotenv()
except ImportError:
    pass

from attractor.llm.adapters.base import ProviderAdapter
from attractor.llm.errors import ConfigurationError
from attractor.llm.types import Request, Response, StreamEvent

Middleware = Callable[[Request, Callable[[Request], Response]], Response]


class Client:
    """Main orchestration layer routing requests to provider adapters."""

    def __init__(
        self,
        providers: dict[str, ProviderAdapter] | None = None,
        default_provider: str | None = None,
    ):
        self._providers: dict[str, ProviderAdapter] = providers or {}
        self._default_provider = default_provider
        self._middleware: list[Middleware] = []
        if not self._default_provider and self._providers:
            self._default_provider = next(iter(self._providers))

    @classmethod
    def from_env(cls) -> Client:
        """Create a client configured for LiteLLM."""
        providers: dict[str, ProviderAdapter] = {}
        try:
            from attractor.llm.adapters.litellm import LiteLLMAdapter

            providers["litellm"] = LiteLLMAdapter()
        except Exception as exc:
            raise ConfigurationError(
                "Failed to initialize LiteLLM adapter. "
                "Install litellm and configure LITELLM_MODEL and credentials if needed."
            ) from exc

        default_provider = os.environ.get("DEFAULT_LLM_PROVIDER", "").lower()
        if default_provider and default_provider not in providers:
            default_provider = "litellm"
        if not default_provider:
            default_provider = "litellm"
        return cls(providers=providers, default_provider=default_provider)

    def register_provider(self, provider_id: str, adapter: ProviderAdapter) -> None:
        self._providers[provider_id] = adapter
        if not self._default_provider:
            self._default_provider = provider_id

    def add_middleware(self, middleware: Middleware) -> None:
        self._middleware.append(middleware)

    def _resolve_provider(self, request: Request) -> ProviderAdapter:
        provider_key = request.provider or self._default_provider
        if not provider_key:
            raise ConfigurationError("No provider specified and no default provider configured.")
        if provider_key not in self._providers:
            raise ConfigurationError(
                f"Provider '{provider_key}' is not registered. Available: {list(self._providers.keys())}"
            )
        return self._providers[provider_key]

    def complete(self, request: Request) -> Response:
        adapter = self._resolve_provider(request)

        def base_call(req: Request) -> Response:
            return adapter.complete(req)

        handler = base_call
        for middleware in reversed(self._middleware):
            previous_handler = handler

            def wrapped_handler(
                req: Request, _middleware=middleware, _previous=previous_handler
            ) -> Response:
                return _middleware(req, _previous)

            handler = wrapped_handler

        return handler(request)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        adapter = self._resolve_provider(request)
        async for event in adapter.stream(request):
            yield event


_default_client: Client | None = None


def set_default_client(client: Client) -> None:
    global _default_client
    _default_client = client


def get_default_client() -> Client:
    global _default_client
    if _default_client is None:
        _default_client = Client.from_env()
    return _default_client
