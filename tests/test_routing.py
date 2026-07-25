from hallucide import DomainRouter
from hallucide.core_types.exceptions import HallucideError

DOMAINS = {
    "Clinical_Trials": "Essais cliniques randomisés",
    "Oncology_Trials": "Essais cliniques en oncologie",
}


class _DummyModelProvider:
    supports_forced_tool_calling = False

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_messages = None

    def generate(self, messages, tools=None, tool_choice=None):
        self.last_messages = messages
        return {"text": self.reply}


class _FailingModelProvider:
    supports_forced_tool_calling = False

    def generate(self, messages, tools=None, tool_choice=None):
        raise RuntimeError("network down")


def test_router_returns_matching_domain() -> None:
    provider = _DummyModelProvider("Oncology_Trials")
    router = DomainRouter(provider, DOMAINS)
    assert router.route("Le traitement réduit-il le risque de rechute après un cancer ?") == "Oncology_Trials"


def test_router_returns_none_on_explicit_aucun() -> None:
    provider = _DummyModelProvider("AUCUN")
    router = DomainRouter(provider, DOMAINS)
    assert router.route("Quelle est la recette du gâteau au chocolat ?") is None


def test_router_forces_out_of_list_answer_to_none() -> None:
    # Garde-fou non négociable (§5) : Gemma ne peut jamais faire apparaître
    # un dataset qui n'a pas été listé, même s'il en invente un plausible.
    provider = _DummyModelProvider("Neurology_Studies")
    router = DomainRouter(provider, DOMAINS)
    assert router.route("Une question quelconque") is None


def test_router_strips_stray_punctuation_from_reply() -> None:
    provider = _DummyModelProvider('"Clinical_Trials".')
    router = DomainRouter(provider, DOMAINS)
    assert router.route("Une question sur les essais cliniques") == "Clinical_Trials"


def test_router_returns_none_when_no_domains_configured() -> None:
    router = DomainRouter(_DummyModelProvider("whatever"), {})
    assert router.route("peu importe") is None


def test_router_raises_hallucide_error_on_technical_failure() -> None:
    # Distinction cruciale (§5) : une panne technique du routage n'est PAS un
    # refus légitime -- l'appelant doit pouvoir la distinguer (HallucideError)
    # d'un simple retour None, pour retomber sur le dataset fixe plutôt que
    # de refuser à tort de répondre.
    router = DomainRouter(_FailingModelProvider(), DOMAINS)
    try:
        router.route("une question")
        assert False, "Expected HallucideError"
    except HallucideError:
        assert True


def test_router_prompt_lists_only_domain_keys_not_arbitrary_text() -> None:
    provider = _DummyModelProvider("Clinical_Trials")
    router = DomainRouter(provider, DOMAINS)
    router.route("une question")
    system_prompt = provider.last_messages[0]["content"]
    assert "Clinical_Trials" in system_prompt
    assert "Oncology_Trials" in system_prompt
    assert "AUCUN" in system_prompt
