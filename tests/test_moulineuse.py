import pytest

from hallucide.core_types.exceptions import RetrievalError
from hallucide.retrieval.moulineuse import MoulineuseRetrievalProvider
from hallucide.core_types.types import Intent, RetrievalState


class FakeMcpClient:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []

    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self.calls.append((name, arguments))
        return self.responses[name]


def _state() -> RetrievalState:
    return RetrievalState(hop_count=0, visited_documents=set(), remaining_budget=5)


def test_code_article_route_marks_vigueur_as_opposable() -> None:
    client = FakeMcpClient(
        {
            "query_sql": [
                {
                    "id": "LEGIARTI000032040777",
                    "num": "1103",
                    "date_debut": "2016-10-01",
                    "date_fin": "2999-01-01",
                    "etat": "VIGUEUR",
                    "titre_texte": "Code civil",
                    "contenu": "Les contrats légalement formés tiennent lieu de loi à ceux qui les ont faits.",
                }
            ]
        }
    )
    provider = MoulineuseRetrievalProvider(client=client)
    passage = provider.retrieve(
        Intent(id="1", question="Que dit l'article 1103 du code civil ?"),
        _state(),
        {"route": "code_article", "article": "1103", "code": "code civil"},
    )

    assert passage.source_id == "LEGIARTI000032040777"
    assert passage.opposable is True
    assert passage.text == "Les contrats légalement formés tiennent lieu de loi à ceux qui les ont faits."
    assert passage.metadata["liens"] == []  # absent de la fixture -> normalisé en liste vide

    name, arguments = client.calls[0]
    assert name == "query_sql"
    assert arguments["schema"] == "legifrance"
    assert arguments["params"][0] == "1103"
    assert arguments["params"][1] == "%code civil%"


def test_code_article_route_exposes_liens_for_multi_hop() -> None:
    # §4ter : les renvois bruts du texte officiel sont exposés tels quels,
    # pour être suivis par multi_hop.py -- jamais résolus automatiquement ici.
    client = FakeMcpClient(
        {
            "query_sql": [
                {
                    "id": "LEGIARTI000032040777",
                    "num": "1103",
                    "etat": "VIGUEUR",
                    "titre_texte": "Code civil",
                    "contenu": "Les contrats légalement formés tiennent lieu de loi à ceux qui les ont faits.",
                    "liens": [
                        {
                            "@num": "L422-2-1",
                            "@cidtexte": "LEGITEXT000006074096",
                            "@typelien": "CITATION",
                            "@naturetexte": "CODE",
                        }
                    ],
                }
            ]
        }
    )
    provider = MoulineuseRetrievalProvider(client=client)
    passage = provider.retrieve(
        Intent(id="1", question="Que dit l'article 1103 du code civil ?"),
        _state(),
        {"route": "code_article", "article": "1103", "code": "code civil"},
    )

    assert len(passage.metadata["liens"]) == 1
    assert passage.metadata["liens"][0]["@cidtexte"] == "LEGITEXT000006074096"


def test_code_article_route_normalizes_single_lien_object_to_list() -> None:
    # legifrance.article->LIENS->LIEN est un objet unique (pas une liste)
    # quand il n'y a qu'un seul lien -- artefact JSONB observé en direct.
    client = FakeMcpClient(
        {
            "query_sql": [
                {
                    "id": "LEGIARTI000032040777",
                    "num": "1103",
                    "etat": "VIGUEUR",
                    "titre_texte": "Code civil",
                    "contenu": "Texte.",
                    "liens": {
                        "@num": "1134",
                        "@cidtexte": "LEGITEXT000006070721",
                        "@typelien": "CONCORDANCE",
                        "@naturetexte": "CODE",
                    },
                }
            ]
        }
    )
    provider = MoulineuseRetrievalProvider(client=client)
    passage = provider.retrieve(
        Intent(id="1", question="?"), _state(), {"route": "code_article", "article": "1103", "code": "code civil"}
    )

    assert isinstance(passage.metadata["liens"], list)
    assert len(passage.metadata["liens"]) == 1


def test_code_article_route_marks_slot_as_copied_when_in_question() -> None:
    client = FakeMcpClient(
        {
            "query_sql": [
                {
                    "id": "LEGIARTI000032040777",
                    "num": "1103",
                    "etat": "VIGUEUR",
                    "titre_texte": "Code civil",
                    "contenu": "Les contrats légalement formés tiennent lieu de loi à ceux qui les ont faits.",
                }
            ]
        }
    )
    provider = MoulineuseRetrievalProvider(client=client)
    passage = provider.retrieve(
        Intent(id="1", question="Que dit l'article 1103 du code civil ?"),
        _state(),
        {"route": "code_article", "article": "1103", "code": "code civil"},
    )

    assert passage.metadata["slot_inferred"] is False
    assert passage.metadata["article_slot_copied"] is True
    assert passage.metadata["code_slot_copied"] is True


def test_code_article_route_flags_inferred_article_per_piege_a3() -> None:
    # Piège A3 (§10) : la question ne mentionne aucun numéro d'article,
    # mais query["article"] a été deviné par le LLM en amont.
    client = FakeMcpClient(
        {
            "query_sql": [
                {
                    "id": "LEGIARTI000032040777",
                    "num": "1103",
                    "etat": "VIGUEUR",
                    "titre_texte": "Code civil",
                    "contenu": "Les contrats légalement formés tiennent lieu de loi à ceux qui les ont faits.",
                }
            ]
        }
    )
    provider = MoulineuseRetrievalProvider(client=client)
    passage = provider.retrieve(
        Intent(id="1", question="Quelle est la règle sur la force obligatoire des contrats ?"),
        _state(),
        {"route": "code_article", "article": "1103", "code": "code civil"},
    )

    assert passage.metadata["slot_inferred"] is True
    assert passage.metadata["article_slot_copied"] is False


def test_code_article_route_marks_non_vigueur_as_not_opposable() -> None:
    client = FakeMcpClient(
        {
            "query_sql": [
                {
                    "id": "LEGIARTI000006436088",
                    "num": "1103",
                    "date_debut": "1804-03-21",
                    "date_fin": "2016-10-01",
                    "etat": "MODIFIE",
                    "titre_texte": "Code civil",
                    "contenu": "Texte abrogé.",
                }
            ]
        }
    )
    provider = MoulineuseRetrievalProvider(client=client)
    passage = provider.retrieve(
        Intent(id="1", question="?"),
        _state(),
        {"route": "code_article", "article": "1103", "code": "code civil"},
    )

    assert passage.opposable is False


def test_code_article_route_refuses_when_no_candidate() -> None:
    client = FakeMcpClient({"query_sql": []})
    provider = MoulineuseRetrievalProvider(client=client)

    with pytest.raises(RetrievalError):
        provider.retrieve(
            Intent(id="1", question="?"),
            _state(),
            {"route": "code_article", "article": "9999-inexistant", "code": "code civil"},
        )


def test_code_article_route_requires_article_and_code() -> None:
    provider = MoulineuseRetrievalProvider(client=FakeMcpClient({}))

    with pytest.raises(RetrievalError):
        provider.retrieve(
            Intent(id="1", question="?"),
            _state(),
            {"route": "code_article", "article": "1103"},
        )


def test_pastille_route_refuses_without_document_identifier() -> None:
    provider = MoulineuseRetrievalProvider(client=FakeMcpClient({}))

    with pytest.raises(RetrievalError):
        provider.retrieve(
            Intent(id="1", question="?"),
            _state(),
            {"route": "pastille", "chambre": "assemblee", "article": "1er"},
        )


def test_pastille_route_returns_non_opposable_by_default() -> None:
    client = FakeMcpClient({"get_pastilled_article": {"text": "Texte de l'amendement."}})
    provider = MoulineuseRetrievalProvider(client=client)

    passage = provider.retrieve(
        Intent(id="1", question="?"),
        _state(),
        {
            "route": "pastille",
            "chambre": "assemblee",
            "article": "1er",
            "documentUid": "DOC123",
        },
    )

    assert passage.opposable is False
    assert passage.source_id == "DOC123"


def test_pastille_route_ignores_opposable_override_from_query() -> None:
    # §6ter/INV-010 : l'opposabilité dérive du type de document, jamais d'un
    # flag de la requête -- un override piloté par la query rendrait INV-010
    # contournable le jour où la formulation de requête est déléguée à un LLM.
    client = FakeMcpClient({"get_pastilled_article": {"text": "Texte de l'amendement."}})
    provider = MoulineuseRetrievalProvider(client=client)

    passage = provider.retrieve(
        Intent(id="1", question="?"),
        _state(),
        {
            "route": "pastille",
            "chambre": "assemblee",
            "article": "1er",
            "documentUid": "DOC123",
            "opposable_override": "true",
        },
    )

    assert passage.opposable is False


def test_code_article_route_flags_ambiguous_selection_across_distinct_codes() -> None:
    # Piège A3 (variante) : LIKE '%civil%' matche le Code civil ET le Code de
    # procédure civile -- rows[0] est une sélection silencieuse, signalée pour
    # que l'orchestration élève le risque.
    client = FakeMcpClient(
        {
            "query_sql": [
                {"id": "LEGIARTI-1", "num": "16", "etat": "VIGUEUR", "titre_texte": "Code civil", "contenu": "Texte A."},
                {"id": "LEGIARTI-2", "num": "16", "etat": "VIGUEUR", "titre_texte": "Code de procédure civile", "contenu": "Texte B."},
            ]
        }
    )
    provider = MoulineuseRetrievalProvider(client=client)
    passage = provider.retrieve(
        Intent(id="1", question="Que dit l'article 16 du code civil ?"),
        _state(),
        {"route": "code_article", "article": "16", "code": "civil"},
    )

    assert passage.metadata["selection_ambiguous"] is True
    assert passage.metadata["candidate_count"] == 2


def test_code_article_route_does_not_flag_multiple_versions_of_same_code() -> None:
    # Plusieurs versions du MÊME texte ne sont pas une ambiguïté de sélection :
    # l'ORDER BY date_debut DESC choisit la version la plus récente applicable.
    client = FakeMcpClient(
        {
            "query_sql": [
                {"id": "LEGIARTI-NEW", "num": "1103", "etat": "VIGUEUR", "titre_texte": "Code civil", "contenu": "Version actuelle."},
                {"id": "LEGIARTI-OLD", "num": "1103", "etat": "MODIFIE", "titre_texte": "Code civil", "contenu": "Version antérieure."},
            ]
        }
    )
    provider = MoulineuseRetrievalProvider(client=client)
    passage = provider.retrieve(
        Intent(id="1", question="Que dit l'article 1103 du code civil ?"),
        _state(),
        {"route": "code_article", "article": "1103", "code": "code civil"},
    )

    assert passage.metadata["selection_ambiguous"] is False
    assert passage.text == "Version actuelle."


def test_texte_libre_route_flags_ambiguous_selection_when_multiple_hits() -> None:
    client = FakeMcpClient(
        {
            "search_legal_texts": {
                "hits": [
                    {"document": {"uid": "T1", "autocompletion": "Premier titre."}},
                    {"document": {"uid": "T2", "autocompletion": "Second titre."}},
                ]
            }
        }
    )
    provider = MoulineuseRetrievalProvider(client=client)
    passage = provider.retrieve(
        Intent(id="1", question="?"),
        _state(),
        {"route": "texte_libre", "query": "recherche large"},
    )

    assert passage.metadata["selection_ambiguous"] is True


def test_unknown_route_is_refused() -> None:
    provider = MoulineuseRetrievalProvider(client=FakeMcpClient({}))

    with pytest.raises(RetrievalError):
        provider.retrieve(Intent(id="1", question="?"), _state(), {"route": "bogus"})


def test_texte_libre_route_flags_pertinence_non_garantie() -> None:
    # Piège C1 (§10), reproduit en direct dans la spec : "congé menstruel"
    # -> ordonnances 1943-1962 sur le congé militaire, réelles mais hors-sujet.
    client = FakeMcpClient(
        {
            "search_legal_texts": {
                "hits": [
                    {
                        "document": {
                            "uid": "JORFTEXT000000889059",
                            "autocompletion": "Ordonnance n°62-91 du 26 janvier 1962 RELATIVE AU CONGE SPECIAL DE CERTAINS FONCTIONNAIRES",
                            "page_path": "/legifrance/textes/JORFTEXT000000889059",
                            "badge": "ORDONNANCE",
                        }
                    }
                ]
            }
        }
    )
    provider = MoulineuseRetrievalProvider(client=client)
    passage = provider.retrieve(
        Intent(id="1", question="congé menstruel"),
        _state(),
        {"route": "texte_libre", "query": "congé menstruel"},
    )

    assert passage.source_id == "JORFTEXT000000889059"
    assert passage.opposable is False
    assert passage.metadata["pertinence_non_garantie"] is True
    assert "CONGE SPECIAL" in passage.text


def test_texte_libre_route_refuses_when_no_hits() -> None:
    client = FakeMcpClient({"search_legal_texts": {"hits": []}})
    provider = MoulineuseRetrievalProvider(client=client)

    with pytest.raises(RetrievalError):
        provider.retrieve(
            Intent(id="1", question="?"),
            _state(),
            {"route": "texte_libre", "query": "introuvable"},
        )


def test_texte_libre_route_requires_query() -> None:
    provider = MoulineuseRetrievalProvider(client=FakeMcpClient({}))

    with pytest.raises(RetrievalError):
        provider.retrieve(Intent(id="1", question="?"), _state(), {"route": "texte_libre"})
