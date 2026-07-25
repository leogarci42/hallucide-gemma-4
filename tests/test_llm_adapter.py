from hallucide import PromptBasedDecomposer, PromptBasedIntentGenerator
from hallucide.core_types.exceptions import HallucideError
from hallucide.decomposition.llm import ModelProvider
from hallucide.core_types.types import ClaimStatus, Intent, Passage


class DummyModelProvider:
    supports_forced_tool_calling = False

    def generate(self, messages, tools=None, tool_choice=None):
        system_message = next(m for m in messages if m["role"] == "system")["content"]
        if "Découpe le message" in system_message or "Break down the following user message" in system_message:
            return {"text": '[{"id": "1", "question": "Quelle est la règle ?"}]'}
        if "CITATION vs REFORMULATION" in system_message or "Generate claims that answer" in system_message:
            return {"text": '[{"ref": "Passage authentique.", "status": "AUTHENTIFIÉ"}]'}
        return {"text": "[]"}


def test_prompt_based_decomposer_returns_intent() -> None:
    provider = DummyModelProvider()
    decomposer = PromptBasedDecomposer(provider)
    intents = decomposer.decompose("Quelle est la règle ?")

    assert intents == [Intent(id="1", question="Quelle est la règle ?")]


def test_prompt_based_intent_generator_returns_claims() -> None:
    provider = DummyModelProvider()
    generator = PromptBasedIntentGenerator(provider)
    passage = Passage(
        source_id="doc1",
        source_type="normatif",
        opposable=True,
        text="Passage authentique.",
        metadata={},
    )
    claims = generator.generate_claims(Intent(id="1", question="Quelle est la règle ?"), passage)

    assert claims[0].status == ClaimStatus.AUTHENTIFIÉ
    assert claims[0].ref == "Passage authentique."


def test_prompt_based_decomposer_raises_on_invalid_json() -> None:
    class InvalidProvider(DummyModelProvider):
        def generate(self, messages, tools=None, tool_choice=None):
            return {"text": "not json"}

    decomposer = PromptBasedDecomposer(InvalidProvider())
    try:
        decomposer.decompose("Quelle est la règle ?")
        assert False, "Expected HallucideError"
    except HallucideError:
        assert True
