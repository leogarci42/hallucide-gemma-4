from hallucide.retrieval.moulineuse import _rerank_hits, _hit_title, _hit_date


def _hit(title: str, date: str | None = None) -> dict:
    doc = {"autocompletion": title}
    if date is not None:
        doc["date_texte"] = date
    return {"document": doc}


def test_rerank_pertinence_puts_best_match_first() -> None:
    hits = [
        _hit("Décret sur les marchés publics des PME"),
        _hit("Congé de paternité et d'accueil de l'enfant"),
        _hit("Loi sur la caisse nationale des marchés de l'État"),
    ]
    ordered, ambiguous = _rerank_hits(hits, "congé paternité accueil enfant", "pertinence")

    assert _hit_title(ordered[0]) == "Congé de paternité et d'accueil de l'enfant"
    assert ambiguous is False  # un candidat se détache nettement


def test_rerank_is_deterministic() -> None:
    hits = [_hit("alpha beta"), _hit("beta gamma"), _hit("gamma delta")]
    a, _ = _rerank_hits(hits, "beta gamma delta", "pertinence")
    b, _ = _rerank_hits(hits, "beta gamma delta", "pertinence")
    assert [_hit_title(h) for h in a] == [_hit_title(h) for h in b]


def test_rerank_flags_ambiguous_on_near_tie() -> None:
    # Deux titres quasi identiques (à la ponctuation près) -> égalité serrée -> ambigu.
    hits = [
        _hit("Congé de paternité des agents publics"),
        _hit("Congé de paternité des agents publics."),
    ]
    _, ambiguous = _rerank_hits(hits, "congé de paternité agents publics", "pertinence")
    assert ambiguous is True


def test_rerank_single_hit_is_never_ambiguous() -> None:
    ordered, ambiguous = _rerank_hits([_hit("un seul texte")], "peu importe", "pertinence")
    assert len(ordered) == 1
    assert ambiguous is False


def test_rerank_recent_sorts_by_date_desc() -> None:
    hits = [
        _hit("ancien", date="2001-01-01"),
        _hit("récent", date="2023-06-01"),
        _hit("moyen", date="2015-03-03"),
    ]
    ordered, ambiguous = _rerank_hits(hits, "peu importe", "recent")
    assert [_hit_title(h) for h in ordered] == ["récent", "moyen", "ancien"]
    # Tri explicite par date : pas d'ambiguïté de pertinence.
    assert ambiguous is False


def test_rerank_recent_falls_back_to_pertinence_without_dates() -> None:
    # Aucune date -> repli sur le classement par pertinence.
    hits = [_hit("marchés publics"), _hit("congé paternité")]
    ordered, _ = _rerank_hits(hits, "congé paternité", "recent")
    assert _hit_title(ordered[0]) == "congé paternité"


def test_hit_date_reads_any_date_key() -> None:
    assert _hit_date(_hit("t", date="2020-02-02")) == "2020-02-02"
    assert _hit_date(_hit("t")) is None
