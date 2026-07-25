from hallucide import GeminiModelProvider
from hallucide.core_types.exceptions import HallucideError


def test_gemini_provider_rejects_forced_tool_choice() -> None:
    provider = GeminiModelProvider(api_key="test-key")
    try:
        provider.generate(messages=[{"role": "system", "content": "foo"}], tools=[], tool_choice="required")
        assert False, "Expected HallucideError"
    except HallucideError as exc:
        assert "forced tool calling" in str(exc)


def test_gemini_provider_separates_system_instruction_from_contents() -> None:
    captured = {}

    class DummyProvider(GeminiModelProvider):
        def _send_request(self, payload):
            captured.update(payload)
            return {"candidates": [{"content": {"parts": [{"text": "result from gemini"}]}}]}

    provider = DummyProvider(api_key="test-key")
    response = provider.generate(
        messages=[
            {"role": "system", "content": "Tu es un assistant juridique."},
            {"role": "user", "content": "Quelle est la règle ?"},
        ],
        tools=[],
        tool_choice=None,
    )

    assert response["text"] == "result from gemini"
    assert captured["systemInstruction"]["parts"][0]["text"] == "Tu es un assistant juridique."
    assert captured["contents"] == [{"role": "user", "parts": [{"text": "Quelle est la règle ?"}]}]


def test_gemini_provider_raises_when_response_has_no_candidates() -> None:
    class DummyProvider(GeminiModelProvider):
        def _send_request(self, payload):
            return {"candidates": []}

    provider = DummyProvider(api_key="test-key")
    try:
        provider.generate(messages=[{"role": "user", "content": "hello"}], tools=[], tool_choice=None)
        assert False, "Expected HallucideError"
    except HallucideError:
        assert True
