from hallucide.retrieval.interventions import InterventionsRetrievalProvider
from hallucide.core_types.exceptions import RetrievalError
from hallucide.core_types.types import Intent, RetrievalState


class FakeClient:
    """Client MCP mocké : réponses canned par nom d'outil."""

    def __init__(self, acteurs=None, interventions=None):
        self._acteurs = acteurs or {"results": []}
        self._interventions = interventions or {"results": []}
        self.calls = []

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "search_acteurs":
            return self._acteurs
        if name == "search_interventions":
            return self._interventions
        raise AssertionError(f"outil inattendu: {name}")


def test_intervention_returns_verbatim_text():
    itv = {"results": [
        {"uid": "i1", "texte": "<p>Le Gouvernement est favorable à cet amendement.</p>",
         "orateur": "M. Gérald Darmanin", "dateSeance": "2024-05-01", "reunionRefUid": "RU1",
         "acteurRefUid": "PA607846"},
    ]}
    prov = InterventionsRetrievalProvider(client=FakeClient(interventions=itv))
    p = prov.retrieve(Intent(id="1", question="position de Darmanin ?"),
                      RetrievalState(), {"route": "intervention", "search": "amendement favorable"})
    assert p.text == "Le Gouvernement est favorable à cet amendement."   # HTML nettoyé
    assert p.opposable is False                                          # débat, non opposable
    assert p.metadata["orateur"] == "M. Gérald Darmanin"
    assert p.metadata["pertinence_non_garantie"] is True


def test_intervention_resolves_orateur_to_acteur_ref():
    fc = FakeClient(
        acteurs={"results": [{"uid": "PA607846", "nom": "Darmanin"}]},
        interventions={"results": [{"uid": "i1", "texte": "<p>Texte.</p>", "orateur": "Darmanin"}]},
    )
    prov = InterventionsRetrievalProvider(client=fc)
    prov.retrieve(Intent(id="1", question="?"),
                  RetrievalState(), {"route": "intervention", "search": "sujet", "orateur": "Darmanin"})
    # search_acteurs appelé, puis acteurRefUid transmis à search_interventions
    assert fc.calls[0][0] == "search_acteurs"
    itv_call = [c for c in fc.calls if c[0] == "search_interventions"][0]
    assert itv_call[1].get("acteurRefUid") == "PA607846"


def test_intervention_reranks_best_match_first():
    itv = {"results": [
        {"uid": "loin", "texte": "<p>Question sans rapport sur l'agriculture.</p>", "orateur": "X"},
        {"uid": "proche", "texte": "<p>Le congé de paternité est allongé à 28 jours.</p>", "orateur": "Y"},
    ]}
    prov = InterventionsRetrievalProvider(client=FakeClient(interventions=itv))
    p = prov.retrieve(Intent(id="1", question="?"),
                      RetrievalState(), {"route": "intervention", "search": "congé paternité allongé"})
    assert p.source_id == "proche"   # reranking déterministe


def test_intervention_no_result_raises():
    prov = InterventionsRetrievalProvider(client=FakeClient(interventions={"results": []}))
    try:
        prov.retrieve(Intent(id="1", question="?"),
                      RetrievalState(), {"route": "intervention", "search": "rien"})
        assert False, "attendu RetrievalError"
    except RetrievalError as e:
        assert "Aucune intervention" in str(e)
