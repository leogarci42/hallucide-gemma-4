from hallucide.retrieval.moulineuse import MoulineuseRetrievalProvider, _strip_html
from hallucide.core_types.exceptions import RetrievalError
from hallucide.core_types.types import Intent, RetrievalState


class FakeClient:
    """Client MCP mocké : renvoie des lignes SQL canned pour query_sql."""

    def __init__(self, rows):
        self.rows = rows
        self.last_args = None

    def call_tool(self, name, arguments):
        self.last_args = (name, arguments)
        return self.rows


def _provider(rows):
    return MoulineuseRetrievalProvider(client=FakeClient(rows))


def test_strip_html_removes_tags_and_entities():
    assert _strip_html("<p>Supprimer cet article.&#160;</p>") == "Supprimer cet article."
    assert _strip_html("<p>a</p><p>b</p>") == "a b"
    assert _strip_html(None) == ""
    assert _strip_html(123) == ""


def test_amendement_returns_dispositif_as_verbatim_passage():
    rows = [{
        "uid": "AMANR5L17PO123B456N245",
        "numero": "245",
        "sort": "Rejeté",
        "type_auteur": "Député",
        "dispositif": "<p>Supprimer l'alin&#233;a 3.</p>",
        "expose": "<p>Cet amendement vise &#224; supprimer.</p>",
        "texte_ref": "PRJLANR5L17B0123",
        "legislature": 17,
    }]
    p = _provider(rows).retrieve(Intent(id="1", question="amendement 245 ?"),
                                 RetrievalState(), {"route": "amendement", "numero": "245"})
    assert p.text == "Supprimer l'alinéa 3."          # HTML nettoyé, verbatim
    assert p.opposable is False                        # acte parlementaire
    assert p.metadata["sort"] == "Rejeté"
    assert p.metadata["numero"] == "245"
    assert p.metadata["expose_sommaire"] == "Cet amendement vise à supprimer."
    assert p.metadata["selection_ambiguous"] is False  # un seul texte de rattachement


def test_amendement_flags_ambiguous_when_multiple_textes():
    rows = [
        {"uid": "A1", "numero": "245", "sort": "Adopté", "dispositif": "<p>Texte A.</p>",
         "expose": "", "texte_ref": "PRJL_A", "legislature": 17},
        {"uid": "A2", "numero": "245", "sort": "Rejeté", "dispositif": "<p>Texte B.</p>",
         "expose": "", "texte_ref": "PRJL_B", "legislature": 16},
    ]
    p = _provider(rows).retrieve(Intent(id="1", question="amendement 245 ?"),
                                 RetrievalState(), {"route": "amendement", "numero": "245"})
    # numéro partagé entre deux textes distincts -> sélection ambiguë signalée
    assert p.metadata["selection_ambiguous"] is True
    assert p.metadata["candidate_count"] == 2


def test_amendement_requires_numero():
    try:
        _provider([]).retrieve(Intent(id="1", question="?"),
                               RetrievalState(), {"route": "amendement"})
        assert False, "attendu RetrievalError"
    except RetrievalError as e:
        assert "numero" in str(e)


def test_amendement_no_candidate_raises():
    try:
        _provider([]).retrieve(Intent(id="1", question="?"),
                               RetrievalState(), {"route": "amendement", "numero": "999999"})
        assert False, "attendu RetrievalError"
    except RetrievalError as e:
        assert "No amendment" in str(e)


def test_amendement_legislature_filter_passed_to_sql():
    fc = FakeClient([{"uid": "A", "numero": "245", "sort": "Adopté",
                      "dispositif": "<p>X.</p>", "expose": "", "texte_ref": "T", "legislature": 17}])
    prov = MoulineuseRetrievalProvider(client=fc)
    prov.retrieve(Intent(id="1", question="?"), RetrievalState(),
                  {"route": "amendement", "numero": "245", "legislature": "17"})
    name, args = fc.last_args
    assert name == "query_sql"
    assert args["params"] == ["245", 17]   # législature transmise en param
