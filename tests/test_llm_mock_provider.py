from hallucide import MockModelProvider, PromptBasedDecomposer, PromptBasedIntentGenerator
from hallucide.core_types.exceptions import HallucideError
from hallucide.core_types.types import ClaimStatus, Intent, Passage


def test_mock_model_provider_forced_tool_choice_rejected() -> None:
    provider = MockModelProvider(responses={"decompose": "[]"}, supports_forced_tool_calling=False)
    try:
        provider.generate(messages=[{"role": "system", "content": "Découpe le message"}], tools=[], tool_choice="required")
        assert False, "Expected HallucideError"
    except HallucideError:
        assert True


def test_mock_model_provider_returns_default_response() -> None:
    provider = MockModelProvider(responses={"default": "[]"})
    assert provider.generate(messages=[{"role": "system", "content": "Unknown prompt"}], tools=[], tool_choice=None)["text"] == "[]"


def test_prompt_based_decomposer_uses_mock_provider() -> None:
    responses = {"decompose": '[{"id": "1", "question": "Quelle est la règle ?"}]'}
    provider = MockModelProvider(responses=responses)
    decomposer = PromptBasedDecomposer(provider)
    intents = decomposer.decompose("Quelle est la règle ?")

    assert intents == [Intent(id="1", question="Quelle est la règle ?")]


def test_prompt_based_intent_generator_uses_mock_provider() -> None:
    responses = {"claims": '[{"ref": "Passage authentique.", "status": "AUTHENTIFIÉ"}]'}
    provider = MockModelProvider(responses=responses)
    generator = PromptBasedIntentGenerator(provider)
    passage = Passage(source_id="doc1", source_type="normatif", opposable=True, text="Passage authentique.", metadata={})
    claims = generator.generate_claims(Intent(id="1", question="Quelle est la règle ?"), passage)

    assert claims[0].status == ClaimStatus.AUTHENTIFIÉ
    assert claims[0].ref == "Passage authentique."
