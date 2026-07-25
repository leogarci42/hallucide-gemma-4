from hallucide.retrieval.multi_hop import build_hop_query, extract_followable_hops, select_next_hop
from hallucide.core_types.types import Passage, RetrievalState

_LIENS_1103 = [
    {
        "@id": "LEGIARTI000032006591",
        "@num": "2",
        "#text": "Ordonnance n°2016-131 du 10 février 2016 - art. 2",
        "@sens": "cible",
        "@cidtexte": "JORFTEXT000032004939",
        "@typelien": "MODIFIE",
        "@naturetexte": "ORDONNANCE",
    },
    {
        "@id": "LEGIARTI000006436298",
        "@num": "1134",
        "#text": "Code civil - art. 1134, alinéa 1er (M)",
        "@sens": "source",
        "@cidtexte": "LEGITEXT000006070721",
        "@typelien": "CONCORDANCE",
        "@naturetexte": "CODE",
    },
    {
        "@id": "LEGIARTI000033282941",
        "@num": "L422-2-1",
        "#text": "Code de la construction et de l'habitation. - art. L422-2-1 (VT)",
        "@sens": "cible",
        "@cidtexte": "LEGITEXT000006074096",
        "@typelien": "CITATION",
        "@naturetexte": "CODE",
    },
    {"#text": "Loi 1804-02-07", "@sens": "source", "@typelien": "CODIFICATION"},
]


def _passage(liens) -> Passage:
    return Passage(
        source_id="LEGIARTI000032040777",
        source_type="normatif",
        opposable=True,
        text="Les contrats légalement formés tiennent lieu de loi à ceux qui les ont faits.",
        metadata={"liens": liens},
    )


def test_extract_followable_hops_filters_by_type_and_nature() -> None:
    hops = extract_followable_hops(_passage(_LIENS_1103))

    # MODIFIE (vers une ordonnance) et CODIFICATION (sans @cidtexte) exclus.
    assert len(hops) == 2
    assert all(h.lien_type in ("CITATION", "CONCORDANCE") for h in hops)
    assert {h.article_num for h in hops} == {"1134", "L422-2-1"}


def test_extract_followable_hops_handles_missing_liens() -> None:
    passage = Passage(source_id="d1", source_type="normatif", opposable=True, text="texte", metadata={})
    assert extract_followable_hops(passage) == []


def test_extract_followable_hops_ignores_malformed_entries() -> None:
    passage = _passage(["not a dict", 42, None])
    assert extract_followable_hops(passage) == []


def test_select_next_hop_skips_already_visited() -> None:
    passage = _passage(_LIENS_1103)
    state = RetrievalState(visited_documents={"LEGITEXT000006070721"})

    hop = select_next_hop(passage, state)

    assert hop is not None
    assert hop.code_cid == "LEGITEXT000006074096"


def test_select_next_hop_returns_none_when_all_visited() -> None:
    passage = _passage(_LIENS_1103)
    state = RetrievalState(visited_documents={"LEGITEXT000006070721", "LEGITEXT000006074096"})

    assert select_next_hop(passage, state) is None


def test_select_next_hop_returns_none_when_no_followable_links() -> None:
    passage = _passage([])
    assert select_next_hop(passage, RetrievalState()) is None


def test_build_hop_query_uses_structured_route() -> None:
    hops = extract_followable_hops(_passage(_LIENS_1103))
    target = next(h for h in hops if h.article_num == "L422-2-1")

    query = build_hop_query(target, code_title_hint="construction et de l'habitation")

    assert query == {
        "route": "code_article",
        "article": "L422-2-1",
        "code": "construction et de l'habitation",
    }
