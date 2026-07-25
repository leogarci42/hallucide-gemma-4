import io
import zipfile

import pytest

from hallucide.core_types.exceptions import RetrievalError
from hallucide.retrieval.file_retrieval import FileRetrievalProvider, _extract_csv_bytes, _parse_csv
from hallucide.core_types.types import Intent, RetrievalState

_CSV = (
    "EC_MEASURE;GEO;TIME_PERIOD;OBS_VALUE\n"
    "LVB;06;2025-07;891\n"
    "LVB;06;2024-01;205\n"
    "LVB;18;2024-01;150\n"
)


def _zip_with(names_to_content: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in names_to_content.items():
            zf.writestr(name, content)
    return buf.getvalue()


class FakeClient:
    """Simule get_resource_info ; le téléchargement est monkeypatché à part."""

    def __init__(self, url: str, fmt: str = "csv") -> None:
        self._url = url
        self._fmt = fmt

    def call_tool(self, name, args):
        assert name == "get_resource_info"
        return f"Resource ID: {args['resource_id']}\nFormat: {self._fmt}\nURL: {self._url}\n"


def _provider_with_payload(monkeypatch, payload: bytes) -> FileRetrievalProvider:
    provider = FileRetrievalProvider(client=FakeClient("https://example.test/file.csv"))
    monkeypatch.setattr("hallucide.retrieval.file_retrieval._download", lambda url: payload)
    return provider


def _state():
    return RetrievalState()


def test_extract_csv_from_plain_bytes() -> None:
    assert _extract_csv_bytes(_CSV.encode("utf-8")) == _CSV.encode("utf-8")


def test_extract_csv_from_zip_ignores_metadata() -> None:
    payload = _zip_with({"DS_metadata.csv": "x;y\n1;2\n", "DS_data.csv": _CSV})
    extracted = _extract_csv_bytes(payload).decode("utf-8")
    assert "OBS_VALUE" in extracted


def test_extract_csv_refuses_ambiguous_zip() -> None:
    payload = _zip_with({"a_data.csv": _CSV, "b_data.csv": _CSV})
    with pytest.raises(RetrievalError, match="ambiguous"):
        _extract_csv_bytes(payload)


def test_parse_csv_detects_semicolon() -> None:
    header, data = _parse_csv(_CSV.encode("utf-8"))
    assert header == ["EC_MEASURE", "GEO", "TIME_PERIOD", "OBS_VALUE"]
    assert len(data) == 3


def test_retrieve_addresses_single_cell(monkeypatch) -> None:
    provider = _provider_with_payload(monkeypatch, _CSV.encode("utf-8"))
    passage = provider.retrieve(
        Intent(id="1", question="?"), _state(),
        {"resource_id": "r1", "filters": {"GEO": "06", "TIME_PERIOD": "2025-07"}, "target_column": "OBS_VALUE"},
    )
    assert passage.text == "891"
    assert passage.source_type == "donnee"
    assert passage.metadata["filters"] == {"GEO": "06", "TIME_PERIOD": "2025-07"}


def test_retrieve_refuses_ambiguous_filter(monkeypatch) -> None:
    provider = _provider_with_payload(monkeypatch, _CSV.encode("utf-8"))
    with pytest.raises(RetrievalError, match="matched 2 rows"):
        provider.retrieve(
            Intent(id="1", question="?"), _state(),
            {"resource_id": "r1", "filters": {"GEO": "06"}, "target_column": "OBS_VALUE"},
        )


def test_retrieve_refuses_when_no_match(monkeypatch) -> None:
    provider = _provider_with_payload(monkeypatch, _CSV.encode("utf-8"))
    with pytest.raises(RetrievalError, match="No row matches"):
        provider.retrieve(
            Intent(id="1", question="?"), _state(),
            {"resource_id": "r1", "filters": {"GEO": "99", "TIME_PERIOD": "2025-07"}, "target_column": "OBS_VALUE"},
        )


def test_retrieve_refuses_unknown_target_column(monkeypatch) -> None:
    provider = _provider_with_payload(monkeypatch, _CSV.encode("utf-8"))
    with pytest.raises(RetrievalError, match="Target column"):
        provider.retrieve(
            Intent(id="1", question="?"), _state(),
            {"resource_id": "r1", "filters": {"GEO": "06", "TIME_PERIOD": "2025-07"}, "target_column": "INEXISTANT"},
        )


def test_retrieve_requires_filters(monkeypatch) -> None:
    provider = _provider_with_payload(monkeypatch, _CSV.encode("utf-8"))
    with pytest.raises(RetrievalError, match="filters"):
        provider.retrieve(
            Intent(id="1", question="?"), _state(),
            {"resource_id": "r1", "target_column": "OBS_VALUE"},
        )


def test_xlsx_is_refused(monkeypatch) -> None:
    # Un XLSX (aussi un ZIP) contenant un .xlsx interne -> pas encore supporté.
    # On teste la signature OLE (ancien .xls) qui n'est pas un zip de CSV.
    provider = _provider_with_payload(monkeypatch, b"\xd0\xcf\x11\xe0garbage")
    with pytest.raises(RetrievalError, match="XLSX/XLS not yet supported"):
        provider.retrieve(
            Intent(id="1", question="?"), _state(),
            {"resource_id": "r1", "filters": {"GEO": "06"}, "target_column": "OBS_VALUE"},
        )
