from hallucide.retrieval.moulineuse import MoulineuseRetrievalProvider, _date_seule
from hallucide.core_types.exceptions import RetrievalError
from hallucide.core_types.types import Intent, RetrievalState


class FakeClient:
    """Client MCP mocké : réponses SQL canned selon la requête."""

    def __init__(self, uid_rows=None, commission_rows=None):
        self.uid_rows = uid_rows
        self.commission_rows = commission_rows or []
        self.calls = []

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        sql = arguments.get("query", "")
        if "SELECT uid FROM assemblee.acteurs" in sql:
            return self.uid_rows if self.uid_rows is not None else []
        return self.commission_rows


def _intent():
    return Intent(id="1", question="commissions de X ?")


def test_date_seule():
    assert _date_seule("2022-07-12T00:00:00+02:00") == "2022-07-12"
    assert _date_seule(None) is None
    assert _date_seule("") is None


def test_commissions_lists_memberships_with_dates():
    rows = [
        {"libelle": "Commission des lois", "code_type": "COMPER", "qualite": "Président",
         "date_debut": "2022-07-12T00:00:00+02:00", "date_fin": "2024-06-01T00:00:00+02:00"},
        {"libelle": "Commission des finances", "code_type": "COMPER", "qualite": None,
         "date_debut": "2024-07-07T00:00:00+02:00", "date_fin": None},
    ]
    prov = MoulineuseRetrievalProvider(client=FakeClient(commission_rows=rows))
    p = prov.retrieve(_intent(), RetrievalState(),
                      {"route": "commissions", "acteur_ref": "PA722190"})
    assert "Commission des lois — Président — du 2022-07-12 au 2024-06-01" in p.text
    assert "Commission des finances — Membre — du 2024-07-07 au en cours" in p.text   # défaut
    assert p.opposable is False          # acte administratif
    assert p.source_type == "mandat"
    assert p.metadata["nb_commissions"] == 2


def test_commissions_resolves_name_to_uid():
    fc = FakeClient(uid_rows=[{"uid": "PA722190"}],
                    commission_rows=[{"libelle": "Commission des lois", "code_type": "COMPER",
                                      "date_debut": "2022-07-12T00:00:00+02:00", "date_fin": None}])
    prov = MoulineuseRetrievalProvider(client=fc)
    prov.retrieve(_intent(), RetrievalState(), {"route": "commissions", "acteur": "Gabriel Attal"})
    # 1er appel = résolution du nom -> uid, avec ILIKE '%Gabriel Attal%'
    assert fc.calls[0][1]["params"] == ["%Gabriel Attal%"]


def test_commissions_unknown_deputy_raises():
    prov = MoulineuseRetrievalProvider(client=FakeClient(uid_rows=[]))
    try:
        prov.retrieve(_intent(), RetrievalState(), {"route": "commissions", "acteur": "Personne Inconnue"})
        assert False, "attendu RetrievalError"
    except RetrievalError as e:
        assert "introuvable" in str(e)


def test_commissions_no_membership_raises():
    prov = MoulineuseRetrievalProvider(client=FakeClient(commission_rows=[]))
    try:
        prov.retrieve(_intent(), RetrievalState(), {"route": "commissions", "acteur_ref": "PA000000"})
        assert False, "attendu RetrievalError"
    except RetrievalError as e:
        assert "Aucune appartenance" in str(e)


def _rows_hollande():
    return [
        {"libelle": "Commission de la défense nationale et des forces armées", "code_type": "COMPER",
         "date_debut": "2002-06-26T00:00:00+02:00", "date_fin": "2007-06-19T00:00:00+02:00"},
        {"libelle": "Commission des affaires sociales", "code_type": "COMPER",
         "date_debut": "2012-02-11T00:00:00+01:00", "date_fin": "2012-02-20T00:00:00+01:00"},
        {"libelle": "Commission des finances, de l'économie générale", "code_type": "COMPER",
         "date_debut": "2009-07-01T00:00:00+02:00", "date_fin": None},
    ]


def test_commissions_targeted_question_yes():
    prov = MoulineuseRetrievalProvider(client=FakeClient(commission_rows=_rows_hollande()))
    p = prov.retrieve(_intent(), RetrievalState(),
                      {"route": "commissions", "acteur_ref": "PA1", "commission": "affaires sociales"})
    assert p.metadata["reponse"] == "oui"
    assert p.text == "Commission des affaires sociales — Membre — du 2012-02-11 au 2012-02-20"
    assert p.metadata["nb_total"] == 3   # la liste complète reste en traçabilité


def test_commissions_targeted_question_no():
    prov = MoulineuseRetrievalProvider(
        client=FakeClient(uid_rows=[{"uid": "PA1"}], commission_rows=_rows_hollande()))
    p = prov.retrieve(_intent(), RetrievalState(),
                      {"route": "commissions", "acteur": "François Hollande", "commission": "culture"})
    assert p.metadata["reponse"] == "non"
    assert "Aucune appartenance" in p.text
    assert "culture" in p.text


def test_commissions_deduplicates_lines():
    rows = [
        {"libelle": "Commission des lois", "code_type": "COMPER",
         "date_debut": "2022-07-12T00:00:00+02:00", "date_fin": None},
        {"libelle": "Commission des lois", "code_type": "COMPER",
         "date_debut": "2022-07-12T00:00:00+02:00", "date_fin": None},
    ]
    prov = MoulineuseRetrievalProvider(client=FakeClient(commission_rows=rows))
    p = prov.retrieve(_intent(), RetrievalState(), {"route": "commissions", "acteur_ref": "PA1"})
    assert p.metadata["nb_commissions"] == 1
