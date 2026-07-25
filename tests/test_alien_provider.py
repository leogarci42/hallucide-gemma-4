from hallucide import AlienRetrievalProvider
from hallucide.core_types.exceptions import RetrievalError
from hallucide.core_types.types import Intent, RetrievalState


class _DummyClient:
    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.response


def _provider(response):
    client = _DummyClient(response)
    provider = AlienRetrievalProvider(api_token="test-token", client=client)
    return provider, client


def test_alien_provider_requires_dataset_id() -> None:
    provider, _ = _provider({"data": {"results": []}})
    try:
        provider.retrieve(Intent(id="1", question="q"), RetrievalState(), {})
        assert False, "Expected RetrievalError"
    except RetrievalError as exc:
        assert "dataset_id" in str(exc)


def test_alien_provider_aggregates_chunks_without_inline_metadata_header() -> None:
    # Régression : un en-tête "[Source N — score...]" en tête de chunk fait
    # que Gemma le recopie comme affirmation au lieu du contenu réel (bug
    # constaté en direct). Le texte agrégé ne doit contenir QUE le texte des
    # chunks, jamais de métadonnée inline.
    response = {
        "data": {
            "results": [
                {
                    "id": "abc",
                    "score": 0.91,
                    "chunk_text": "Le traitement X réduit les rechutes de 34%.",
                    "metadata": {"entry_id": 42},
                },
                {
                    "id": "def",
                    "score": 0.85,
                    "chunk_text": "Une méta-analyse confirme RR=0.68.",
                    "metadata": {"entry_id": 43},
                },
            ]
        }
    }
    provider, client = _provider(response)
    passage = provider.retrieve(
        Intent(id="1", question="Le traitement X marche-t-il ?"),
        RetrievalState(),
        {"dataset_id": "30"},
    )

    assert "Le traitement X réduit les rechutes de 34%." in passage.text
    assert "Une méta-analyse confirme RR=0.68." in passage.text
    assert "score" not in passage.text
    assert "entry_id" not in passage.text
    assert "[Source" not in passage.text
    assert passage.opposable is True
    assert passage.metadata["nb_passages"] == 2
    assert passage.metadata["chunks"][0]["score"] == 0.91
    assert passage.metadata["chunks"][0]["entry_id"] == "42"

    name, arguments = client.calls[0]
    assert name == "datacluster_vector_search_chunks"
    assert arguments["dataset_ids"] == ["30"]
    assert arguments["query"] == "Le traitement X marche-t-il ?"


def test_alien_provider_raises_when_no_chunks_found() -> None:
    provider, _ = _provider({"data": {"results": []}})
    try:
        provider.retrieve(Intent(id="1", question="q"), RetrievalState(), {"dataset_id": "30"})
        assert False, "Expected RetrievalError"
    except RetrievalError as exc:
        assert "30" in str(exc)


def test_alien_provider_ignores_chunks_without_text() -> None:
    response = {"data": {"results": [{"id": "a", "score": 0.9, "chunk_text": ""}, {"id": "b", "score": 0.5}]}}
    provider, _ = _provider(response)
    try:
        provider.retrieve(Intent(id="1", question="q"), RetrievalState(), {"dataset_id": "X"})
        assert False, "Expected RetrievalError"
    except RetrievalError:
        assert True


def test_alien_provider_raises_on_unexpected_response_shape() -> None:
    provider, _ = _provider({"unexpected": "shape"})
    try:
        provider.retrieve(Intent(id="1", question="q"), RetrievalState(), {"dataset_id": "30"})
        assert False, "Expected RetrievalError"
    except RetrievalError as exc:
        assert "Unexpected response shape" in str(exc)
