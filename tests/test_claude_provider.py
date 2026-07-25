from hallucide import ClaudeModelProvider
from hallucide.core_types.exceptions import HallucideError


def test_claude_provider_rejects_forced_tool_choice() -> None:
    provider = ClaudeModelProvider(api_key="test-key")
    try:
        provider.generate(messages=[{"role": "system", "content": "foo"}], tools=[], tool_choice="required")
        assert False, "Expected HallucideError"
    except HallucideError as exc:
        assert "forced tool calling" in str(exc)


def test_claude_provider_separates_system_from_messages() -> None:
    captured = {}

    class DummyProvider(ClaudeModelProvider):
        def _send_request(self, payload):
            captured.update(payload)
            return {"content": [{"type": "text", "text": "result from claude"}]}

    provider = DummyProvider(api_key="test-key")
    response = provider.generate(
        messages=[
            {"role": "system", "content": "Tu es un assistant juridique."},
            {"role": "user", "content": "Quelle est la règle ?"},
        ],
        tools=[],
        tool_choice=None,
    )

    assert response["text"] == "result from claude"
    # Le prompt système part dans le champ de tête `system`, pas dans `messages`.
    assert captured["system"] == "Tu es un assistant juridique."
    assert captured["messages"] == [{"role": "user", "content": "Quelle est la règle ?"}]
    # max_tokens est obligatoire côté API Anthropic.
    assert captured["max_tokens"] == provider.max_output_tokens


def test_claude_provider_default_model_is_opus() -> None:
    provider = ClaudeModelProvider(api_key="test-key")
    assert provider.model == "claude-opus-4-8"


def test_claude_provider_raises_when_no_text_block() -> None:
    class DummyProvider(ClaudeModelProvider):
        def _send_request(self, payload):
            # Réponse sans bloc texte (ex. refus) -> extraction impossible.
            return {"content": []}

    provider = DummyProvider(api_key="test-key")
    try:
        provider.generate(messages=[{"role": "user", "content": "hello"}], tools=[], tool_choice=None)
        assert False, "Expected HallucideError"
    except HallucideError:
        assert True
