from hallucide import MistralModelProvider
from hallucide.core_types.exceptions import HallucideError


def test_mistral_provider_rejects_forced_tool_choice() -> None:
    provider = MistralModelProvider(api_key="test-key")
    try:
        provider.generate(messages=[{"role": "system", "content": "foo"}], tools=[], tool_choice="required")
        assert False, "Expected HallucideError"
    except HallucideError as exc:
        assert "forced tool calling" in str(exc)


def test_mistral_provider_builds_chat_completions_payload() -> None:
    captured = {}

    class DummyProvider(MistralModelProvider):
        def _send_request(self, payload):
            captured.update(payload)
            return {"choices": [{"message": {"content": "result from mistral"}}]}

    provider = DummyProvider(api_key="test-key", model="mistral-small-latest")
    response = provider.generate(
        messages=[{"role": "system", "content": "hello"}, {"role": "user", "content": "world"}],
        tools=[],
        tool_choice=None,
    )

    assert response["text"] == "result from mistral"
    assert captured["model"] == "mistral-small-latest"
    assert captured["messages"] == [
        {"role": "system", "content": "hello"},
        {"role": "user", "content": "world"},
    ]


def test_mistral_provider_raises_when_response_has_no_choices() -> None:
    class DummyProvider(MistralModelProvider):
        def _send_request(self, payload):
            return {"choices": []}

    provider = DummyProvider(api_key="test-key")
    try:
        provider.generate(messages=[{"role": "user", "content": "hello"}], tools=[], tool_choice=None)
        assert False, "Expected HallucideError"
    except HallucideError:
        assert True
