from hallucide import AlienRetrievalProvider
from hallucide.core_types.exceptions import RetrievalError
from hallucide.core_types.types import Intent, RetrievalState


def _provider(response):
    class DummyProvider(AlienRetrievalProvider):
        def _send_request(self, payload):
            captured.update(payload)
            return response

    captured = {}
    provider = DummyProvider(api_token="test-token", cluster_id="test-cluster")
    return provider, captured


def test_alien_provider_requires_dataset_id() -> None:
    provider, _ = _provider({"chunks": []})
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
        "chunks": [
            {"chunk_text": "Le traitement X réduit les rechutes de 34%.", "score": 0.91, "entry_id": "e1"},
            {"chunk_text": "Une méta-analyse confirme RR=0.68.", "score": 0.85, "entry_id": "e2"},
        ]
    }
    provider, captured = _provider(response)
    passage = provider.retrieve(
        Intent(id="1", question="Le traitement X marche-t-il ?"),
        RetrievalState(),
        {"dataset_id": "Clinical_Trials"},
    )

    assert "Le traitement X réduit les rechutes de 34%." in passage.text
    assert "Une méta-analyse confirme RR=0.68." in passage.text
    assert "score" not in passage.text
    assert "entry_id" not in passage.text
    assert "[Source" not in passage.text
    assert passage.opposable is True
    assert passage.metadata["nb_passages"] == 2
    assert passage.metadata["chunks"][0]["score"] == 0.91
    assert captured["dataset_ids"] == ["Clinical_Trials"]
    assert captured["query"] == "Le traitement X marche-t-il ?"


def test_alien_provider_raises_when_no_chunks_found() -> None:
    provider, _ = _provider({"chunks": []})
    try:
        provider.retrieve(Intent(id="1", question="q"), RetrievalState(), {"dataset_id": "Clinical_Trials"})
        assert False, "Expected RetrievalError"
    except RetrievalError as exc:
        assert "Clinical_Trials" in str(exc)


def test_alien_provider_ignores_chunks_without_text() -> None:
    response = {"chunks": [{"chunk_text": "", "score": 0.9, "entry_id": "e1"}, {"score": 0.5}]}
    provider, _ = _provider(response)
    try:
        provider.retrieve(Intent(id="1", question="q"), RetrievalState(), {"dataset_id": "X"})
        assert False, "Expected RetrievalError"
    except RetrievalError:
        assert True
