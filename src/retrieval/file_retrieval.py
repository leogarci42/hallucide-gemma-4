from __future__ import annotations

import csv
import io
import urllib.error
import urllib.request
import zipfile
from typing import Any

from hallucide.core_types.exceptions import RetrievalError
from hallucide.retrieval.mcp_client import McpToolClient
from hallucide.analysis.trust import ensure_system_trust_store
from hallucide.core_types.types import Intent, Passage, RetrievalState

DEFAULT_DATAGOUV_URL = "https://mcp.data.gouv.fr/mcp"
_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # garde-fou : ne pas parser un fichier énorme en mémoire
_CSV_DELIMITERS = ";,\t|"


def _resolve_resource_url(client: McpToolClient, resource_id: str) -> tuple[str, str]:
    """Récupère l'URL du fichier et son format déclaré via get_resource_info.
    Retourne (url, format_declare_minuscule)."""
    info = client.call_tool("get_resource_info", {"resource_id": resource_id})
    text = info if isinstance(info, str) else str(info)
    url = None
    fmt = ""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("URL:"):
            url = s[len("URL:"):].strip()
        elif s.startswith("Format:"):
            fmt = s[len("Format:"):].strip().lower()
        elif s.startswith("MIME type:"):
            fmt = fmt or s[len("MIME type:"):].strip().lower()
    if not url:
        raise RetrievalError(f"No downloadable URL for resource '{resource_id}'.")
    return url, fmt


def _download(url: str) -> bytes:
    ensure_system_trust_store()
    req = urllib.request.Request(url, headers={"User-Agent": "hallucide/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read(_MAX_DOWNLOAD_BYTES + 1)
    except urllib.error.URLError as exc:
        raise RetrievalError(f"Download failed for {url}: {exc.reason}") from exc
    if len(raw) > _MAX_DOWNLOAD_BYTES:
        raise RetrievalError(f"Resource too large to parse safely (> {_MAX_DOWNLOAD_BYTES} bytes).")
    return raw


def _extract_csv_bytes(raw: bytes) -> bytes:
    """Détecte le vrai format par magic bytes (le MIME déclaré ment souvent,
    ex. INSEE annonce text/csv pour un ZIP). Retourne les octets du CSV de
    données. Refuse si le contenu est ambigu (plusieurs CSV de données)."""
    if raw[:2] == b"PK":  # signature ZIP
        try:
            zf = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile as exc:
            raise RetrievalError("Downloaded file looks like a ZIP but is corrupt.") from exc
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        data_names = [n for n in csv_names if "metadata" not in n.lower()]
        candidates = data_names or csv_names
        if len(candidates) == 0:
            raise RetrievalError("ZIP archive contains no CSV file.")
        if len(candidates) > 1:
            raise RetrievalError(
                f"ZIP contains {len(candidates)} data CSV files; ambiguous, refusing to guess: {candidates}"
            )
        return zf.read(candidates[0])
    if raw[:4] in (b"PK\x03\x04",) or raw[:2] == b"\xd0\xcf":  # XLSX (zip) ou XLS (OLE)
        raise RetrievalError("XLSX/XLS not yet supported by this route (CSV/ZIP-CSV only).")
    return raw  # supposé CSV texte brut


def _parse_csv(raw_csv: bytes) -> tuple[list[str], list[list[str]]]:
    """Parse un CSV avec détection défensive de l'encodage et du séparateur.
    Refuse si le séparateur est indéterminable."""
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = raw_csv.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RetrievalError("Could not decode CSV (unknown encoding).")

    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=_CSV_DELIMITERS)
    except csv.Error as exc:
        raise RetrievalError("Could not determine CSV delimiter; refusing to guess.") from exc

    reader = csv.reader(io.StringIO(text), dialect)
    rows = list(reader)
    if not rows:
        raise RetrievalError("CSV is empty.")
    header = [h.strip() for h in rows[0]]
    data = [r for r in rows[1:] if any(cell.strip() for cell in r)]
    return header, data


class FileRetrievalProvider:
    """RetrievalProvider pour les ressources data.gouv NON indexées par l'API
    tabulaire mais disponibles comme fichiers CSV (ou ZIP contenant un CSV) --
    fréquent à l'INSEE (§6ter). Télécharge, parse défensivement, et adresse
    UNE cellule de façon déterministe via des filtres multi-colonnes.

    query attendu :
      resource_id     : ressource data.gouv
      filters         : dict {colonne: valeur} -- doivent isoler EXACTEMENT
                        une ligne (0 ou >1 -> refus, jamais de devinette)
      target_column   : colonne dont on extrait la valeur (la cellule)
      dataset_id      : optionnel, pour la traçabilité

    Le résultat est un Passage source_type="donnee", opposable=True, dont le
    text est la valeur exacte de la cellule -> DONNÉE_TRACÉE après vérification
    (INV-013 : égalité numérique stricte).
    """

    def __init__(self, client: McpToolClient | None = None) -> None:
        self.client = client or McpToolClient(DEFAULT_DATAGOUV_URL)

    def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, Any]) -> Passage:
        resource_id = query.get("resource_id")
        target_column = query.get("target_column")
        filters = query.get("filters")

        if not resource_id or not target_column:
            raise RetrievalError("file route requires 'resource_id' and 'target_column'.")
        if not isinstance(filters, dict) or not filters:
            raise RetrievalError("file route requires a non-empty 'filters' dict to address a single cell.")

        url, _fmt = _resolve_resource_url(self.client, str(resource_id))
        header, data = _parse_csv(_extract_csv_bytes(_download(url)))

        if target_column not in header:
            raise RetrievalError(
                f"Target column '{target_column}' not found (available: {', '.join(header)})."
            )
        for col in filters:
            if col not in header:
                raise RetrievalError(f"Filter column '{col}' not found (available: {', '.join(header)}).")

        col_index = {name: i for i, name in enumerate(header)}
        matched = [
            row for row in data
            if all(row[col_index[col]].strip() == str(val).strip() for col, val in filters.items())
        ]

        if len(matched) == 0:
            raise RetrievalError(f"No row matches filters {filters} in resource '{resource_id}'.")
        if len(matched) > 1:
            raise RetrievalError(
                f"Filters {filters} matched {len(matched)} rows; expected exactly one for a traceable cell."
            )

        cell = matched[0][col_index[target_column]].strip()
        if not cell:
            raise RetrievalError(f"Cell '{target_column}' is empty for the matched row.")

        return Passage(
            source_id=str(resource_id),
            source_type="donnee",
            opposable=True,
            text=cell,
            metadata={
                "dataset_id": query.get("dataset_id"),
                "resource_id": resource_id,
                "filters": filters,
                "target_column": target_column,
                "source_url": url,
                "row": dict(zip(header, matched[0])),
            },
        )
