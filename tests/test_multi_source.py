import pytest

from hallucide.core_types.exceptions import RetrievalError
from hallucide.retrieval.multi_source import MultiSourceRetrievalProvider
from hallucide.core_types.types import Intent, Passage, RetrievalState


class FakeMoulineuse:
    def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]) -> Passage:
        return Passage(source_id="moulineuse-doc", source_type="normatif", opposable=True, text="texte normatif", metadata={})


class FakeDataGouv:
    def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]) -> Passage:
        return Passage(source_id="datagouv-resource", source_type="donnee", opposable=True, text="318961", metadata={})


def _state() -> RetrievalState:
    return RetrievalState(hop_count=0, visited_documents=set(), remaining_budget=5)


def test_routes_to_moulineuse_when_route_key_present() -> None:
    provider = MultiSourceRetrievalProvider(moulineuse=FakeMoulineuse(), datagouv=FakeDataGouv())
    passage = provider.retrieve(
        Intent(id="1", question="?"), _state(), {"route": "code_article", "article": "1103", "code": "code civil"}
    )

    assert passage.source_id == "moulineuse-doc"


def test_routes_to_datagouv_when_target_column_key_present() -> None:
    provider = MultiSourceRetrievalProvider(moulineuse=FakeMoulineuse(), datagouv=FakeDataGouv())
    passage = provider.retrieve(
        Intent(id="1", question="?"),
        _state(),
        {"resource_id": "abc", "target_column": "Inscrits", "filter_column": "x", "filter_value": "y"},
    )

    assert passage.source_id == "datagouv-resource"


def test_refuses_when_neither_key_present() -> None:
    provider = MultiSourceRetrievalProvider(moulineuse=FakeMoulineuse(), datagouv=FakeDataGouv())

    with pytest.raises(RetrievalError):
        provider.retrieve(Intent(id="1", question="?"), _state(), {"foo": "bar"})
