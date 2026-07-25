from hallucide import GemmaModelProvider
from hallucide.core_types.exceptions import HallucideError


def test_gemma_provider_rejects_forced_tool_choice() -> None:
    provider = GemmaModelProvider()
    try:
        provider.generate(messages=[{"role": "system", "content": "foo"}], tools=[], tool_choice="required")
        assert False, "Expected HallucideError"
    except HallucideError as exc:
        assert "forced tool calling" in str(exc)


def test_gemma_provider_builds_chat_completions_payload() -> None:
    captured = {}

    class DummyProvider(GemmaModelProvider):
        def _send_request(self, payload):
            captured.update(payload)
            return {"choices": [{"message": {"content": "Bonjour"}}]}

    provider = DummyProvider(base_url="http://localhost:8000/v1", model="google/gemma-4-E4B-it")
    response = provider.generate(
        messages=[{"role": "system", "content": "hello"}, {"role": "user", "content": "world"}],
        tools=[],
        tool_choice=None,
    )

    assert response["text"] == "Bonjour"
    assert captured["model"] == "google/gemma-4-E4B-it"
    assert captured["messages"] == [
        {"role": "system", "content": "hello"},
        {"role": "user", "content": "world"},
    ]


def test_gemma_provider_raises_when_response_has_no_choices() -> None:
    class DummyProvider(GemmaModelProvider):
        def _send_request(self, payload):
            return {"choices": []}

    provider = DummyProvider()
    try:
        provider.generate(messages=[{"role": "user", "content": "hello"}], tools=[], tool_choice=None)
        assert False, "Expected HallucideError"
    except HallucideError:
        assert True


def test_gemma_provider_defaults_are_configurable_not_hardcoded() -> None:
    # Règle NVIDIA (§7) : URL/modèle doivent être configurables pour basculer
    # Ollama <-> vLLM sans changer le code, jamais en dur.
    provider = GemmaModelProvider(base_url="http://example-vllm-host:9000/v1", model="custom/model")
    assert provider.base_url == "http://example-vllm-host:9000/v1"
    assert provider.model == "custom/model"
