from types import SimpleNamespace

from attractor.llm.adapters.litellm import LiteLLMAdapter
from attractor.llm.client import Client
from attractor.llm.types import Message, Request


class _FakeLiteLLMModule:
    def __init__(self, response):
        self._response = response

    def completion(self, **kwargs):
        return self._response


def test_client_from_env_registers_litellm(monkeypatch):
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "")
    client = Client.from_env()
    assert "litellm" in client._providers  # noqa: SLF001
    assert client._default_provider == "litellm"  # noqa: SLF001


def test_litellm_adapter_complete_parses_response():
    raw = SimpleNamespace(
        id="resp_1",
        model="openai/gpt-4o-mini",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="hello from litellm", tool_calls=[]),
            )
        ],
    )
    adapter = LiteLLMAdapter(api_key="test-key")
    adapter._litellm = _FakeLiteLLMModule(raw)  # noqa: SLF001

    response = adapter.complete(
        Request(model="openai/gpt-4o-mini", messages=[Message.user("hi")])
    )

    assert response.provider == "litellm"
    assert response.text == "hello from litellm"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 4
