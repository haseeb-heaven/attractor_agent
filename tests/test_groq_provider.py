import os

import pytest

from attractor.llm.adapters.groq import GroqAdapter
from attractor.llm.client import Client
from attractor.llm.errors import ConfigurationError


def test_groq_adapter_requires_api_key(monkeypatch):
    """Guard against silent fallback when the key is missing."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        GroqAdapter(api_key=None)


def test_client_from_env_detects_groq(monkeypatch):
    """Ensure Groq is auto-registered when a key is present."""
    for key in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "DEFAULT_LLM_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    client = Client.from_env()

    assert "groq" in client._providers  # noqa: SLF001 - accessing for test clarity
    assert client._default_provider == "groq"
