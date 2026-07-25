import sys
import types

import pytest

from hallucide.core_types.exceptions import HallucideError
from hallucide.llm_providers.litellm_provider import LiteLLMModelProvider


def _install_fake_litellm(monkeypatch, response_content: str | None = "result from litellm", raise_error: bool = False):
    fake_module = types.ModuleType("litellm")
    captured: dict[str, object] = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        if raise_error:
            raise RuntimeError("simulated provider failure")

        message = types.SimpleNamespace(content=response_content)
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])

    fake_module.completion = fake_completion
    monkeypatch.setitem(sys.modules, "litellm", fake_module)
    return captured


def test_rejects_forced_tool_choice() -> None:
    provider = LiteLLMModelProvider(api_key="test-key")
    with pytest.raises(HallucideError, match="forced tool calling"):
        provider.generate(messages=[{"role": "system", "content": "foo"}], tools=[], tool_choice="required")


def test_generate_builds_messages_and_extracts_text(monkeypatch) -> None:
    captured = _install_fake_litellm(monkeypatch)
    provider = LiteLLMModelProvider(api_key="test-key", model="mistral/mistral-small-latest")

    response = provider.generate(
        messages=[{"role": "system", "content": "hello"}, {"role": "user", "content": "world"}],
        tools=[],
        tool_choice=None,
    )

    assert response["text"] == "result from litellm"
    assert captured["model"] == "mistral/mistral-small-latest"
    assert captured["api_key"] == "test-key"
    assert captured["messages"] == [
        {"role": "system", "content": "hello"},
        {"role": "user", "content": "world"},
    ]


def test_raises_when_response_has_no_text(monkeypatch) -> None:
    _install_fake_litellm(monkeypatch, response_content=None)
    provider = LiteLLMModelProvider(api_key="test-key")

    with pytest.raises(HallucideError, match="Unable to extract text"):
        provider.generate(messages=[{"role": "user", "content": "hello"}], tools=[], tool_choice=None)


def test_wraps_underlying_errors(monkeypatch) -> None:
    _install_fake_litellm(monkeypatch, raise_error=True)
    provider = LiteLLMModelProvider(api_key="test-key")

    with pytest.raises(HallucideError, match="LiteLLM error"):
        provider.generate(messages=[{"role": "user", "content": "hello"}], tools=[], tool_choice=None)
