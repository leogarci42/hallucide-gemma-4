import pytest

from hallucide.retrieval.datagouv import DataGouvRetrievalProvider
from hallucide.core_types.exceptions import RetrievalError
from hallucide.core_types.types import Intent, RetrievalState

_SINGLE_ROW_RESPONSE = """Querying resource: resultats-definitifs-par-regions.csv
Resource ID: f69ffab7-fe37-494e-ad6d-a7cfc80ddc1f
Dataset: Élections législatives (ID: 6682d0c255dcda5df20b1d90)

Filter: Libellé région exact Guadeloupe

Total rows (Tabular API): 1
Total pages: 1 (page size: 5)
Retrieved: 1 row(s) from page 1
Columns: __id, Code région, Libellé région, Inscrits, Votants

Data (1 row):
  Row 1:
    __id: 1
    Code région: 1
    Libellé région: Guadeloupe
    Inscrits: 318961
    Votants: 107028
"""

_NO_ROW_RESPONSE = """Querying resource: resultats-definitifs-par-regions.csv
Resource ID: f69ffab7-fe37-494e-ad6d-a7cfc80ddc1f

Filter: Libellé région exact RegionInexistante

Total rows (Tabular API): 0
Total pages: 0 (page size: 5)
Retrieved: 0 row(s) from page 1
"""

_AMBIGUOUS_RESPONSE = """Querying resource: resultats-definitifs-par-regions.csv
Resource ID: f69ffab7-fe37-494e-ad6d-a7cfc80ddc1f

Filter: Code région exact 1

Total rows (Tabular API): 2
Total pages: 1 (page size: 5)
Retrieved: 2 row(s) from page 1

Data (2 rows):
  Row 1:
    __id: 1
    Inscrits: 318961
  Row 2:
    __id: 2
    Inscrits: 304683
"""


class FakeMcpClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self.calls.append((name, arguments))
        return self.response


def _state() -> RetrievalState:
    return RetrievalState(hop_count=0, visited_documents=set(), remaining_budget=5)


def test_retrieves_exact_cell_value() -> None:
    client = FakeMcpClient(_SINGLE_ROW_RESPONSE)
    provider = DataGouvRetrievalProvider(client=client)

    passage = provider.retrieve(
        Intent(id="1", question="?"),
        _state(),
        {
            "dataset_id": "6682d0c255dcda5df20b1d90",
            "resource_id": "f69ffab7-fe37-494e-ad6d-a7cfc80ddc1f",
            "filter_column": "Libellé région",
            "filter_value": "Guadeloupe",
            "target_column": "Inscrits",
        },
    )

    assert passage.text == "318961"
    assert passage.source_id == "f69ffab7-fe37-494e-ad6d-a7cfc80ddc1f"
    assert passage.source_type == "donnee"
    assert passage.opposable is True
    assert passage.metadata["dataset_id"] == "6682d0c255dcda5df20b1d90"
    assert passage.metadata["target_column"] == "Inscrits"

    name, arguments = client.calls[0]
    assert name == "query_resource_data"
    assert arguments["filter_column"] == "Libellé région"
    assert arguments["filter_value"] == "Guadeloupe"


def test_refuses_when_no_row_matches() -> None:
    client = FakeMcpClient(_NO_ROW_RESPONSE)
    provider = DataGouvRetrievalProvider(client=client)

    with pytest.raises(RetrievalError):
        provider.retrieve(
            Intent(id="1", question="?"),
            _state(),
            {
                "resource_id": "f69ffab7-fe37-494e-ad6d-a7cfc80ddc1f",
                "filter_column": "Libellé région",
                "filter_value": "RegionInexistante",
                "target_column": "Inscrits",
            },
        )


def test_refuses_on_ambiguous_filter_matching_multiple_rows() -> None:
    # Plusieurs lignes correspondent au filtre -- pas de cellule unique adressable.
    client = FakeMcpClient(_AMBIGUOUS_RESPONSE)
    provider = DataGouvRetrievalProvider(client=client)

    with pytest.raises(RetrievalError):
        provider.retrieve(
            Intent(id="1", question="?"),
            _state(),
            {
                "resource_id": "f69ffab7-fe37-494e-ad6d-a7cfc80ddc1f",
                "filter_column": "Code région",
                "filter_value": "1",
                "target_column": "Inscrits",
            },
        )


def test_refuses_when_target_column_missing() -> None:
    client = FakeMcpClient(_SINGLE_ROW_RESPONSE)
    provider = DataGouvRetrievalProvider(client=client)

    with pytest.raises(RetrievalError):
        provider.retrieve(
            Intent(id="1", question="?"),
            _state(),
            {
                "resource_id": "f69ffab7-fe37-494e-ad6d-a7cfc80ddc1f",
                "filter_column": "Libellé région",
                "filter_value": "Guadeloupe",
                "target_column": "ColonneInexistante",
            },
        )


def test_refuses_without_filter_to_avoid_ambiguous_row_choice() -> None:
    provider = DataGouvRetrievalProvider(client=FakeMcpClient(_SINGLE_ROW_RESPONSE))

    with pytest.raises(RetrievalError):
        provider.retrieve(
            Intent(id="1", question="?"),
            _state(),
            {"resource_id": "f69ffab7-fe37-494e-ad6d-a7cfc80ddc1f", "target_column": "Inscrits"},
        )


def test_requires_resource_id_and_target_column() -> None:
    provider = DataGouvRetrievalProvider(client=FakeMcpClient(_SINGLE_ROW_RESPONSE))

    with pytest.raises(RetrievalError):
        provider.retrieve(
            Intent(id="1", question="?"),
            _state(),
            {"filter_column": "Libellé région", "filter_value": "Guadeloupe"},
        )
