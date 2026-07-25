from __future__ import annotations

import re
from typing import Any

from hallucide.core_types.exceptions import RetrievalError
from hallucide.retrieval.mcp_client import McpToolClient
from hallucide.core_types.types import Intent, Passage, RetrievalState

DEFAULT_DATAGOUV_URL = "https://mcp.data.gouv.fr/mcp"

_ROW_HEADER_PATTERN = re.compile(r"^\s*Row\s+\d+\s*:\s*$")
_FIELD_LINE_PATTERN = re.compile(r"^\s{2,}([^:]+?):\s*(.*)$")
_TOTAL_ROWS_PATTERN = re.compile(r"Total rows \(Tabular API\):\s*(\d+)")


def _parse_first_row(text: str) -> dict[str, str]:
    """Parse la sortie texte semi-structurée de `query_resource_data` ("Row N:\\n  clé: valeur")
    en un dict pour la première ligne retournée. Pas de JSON disponible côté outil.
    """
    lines = text.splitlines()
    row: dict[str, str] = {}
    in_row = False
    for line in lines:
        if _ROW_HEADER_PATTERN.match(line):
            if in_row:
                break  # une seule ligne : on s'arrête à la deuxième "Row N:"
            in_row = True
            continue
        if not in_row:
            continue
        match = _FIELD_LINE_PATTERN.match(line)
        if not match:
            break
        key, value = match.group(1).strip(), match.group(2).strip()
        row[key] = value
    return row


def _extract_total_rows(text: str) -> int | None:
    match = _TOTAL_ROWS_PATTERN.search(text)
    return int(match.group(1)) if match else None


class DataGouvRetrievalProvider:
    """RetrievalProvider backed by the data.gouv.fr MCP server (§6ter).

    La fidélité-donnée se prouve par traçabilité (dataset_id + resource_id +
    cellule), jamais par comparaison de chaîne sur du texte normatif : c'est
    pourquoi cette route exige une cellule explicitement adressée (colonne +
    filtre exact), jamais une recherche heuristique côté code -- même
    politique "structuré par défaut" que MoulineuseRetrievalProvider (§4bis).
    """

    def __init__(self, client: McpToolClient | None = None) -> None:
        self.client = client or McpToolClient(DEFAULT_DATAGOUV_URL)

    def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]) -> Passage:
        dataset_id = query.get("dataset_id")
        resource_id = query.get("resource_id")
        filter_column = query.get("filter_column")
        filter_value = query.get("filter_value")
        target_column = query.get("target_column")

        if not resource_id or not target_column:
            raise RetrievalError(
                "data.gouv route requires 'resource_id' and 'target_column'."
            )
        if not filter_column or not filter_value:
            # Sans filtre, plusieurs lignes peuvent matcher -- pas de cellule
            # unique adressable : refus plutôt que de deviner la bonne ligne.
            raise RetrievalError(
                "data.gouv route requires 'filter_column' and 'filter_value' "
                "to address a single, unambiguous cell."
            )

        arguments: dict[str, Any] = {
            "resource_id": resource_id,
            "filter_column": filter_column,
            "filter_value": filter_value,
            "filter_operator": query.get("filter_operator", "exact"),
            "page_size": 5,
        }

        result = self.client.call_tool("query_resource_data", arguments)
        text = _coerce_text(result)

        total_rows = _extract_total_rows(text)
        if total_rows is None or total_rows == 0:
            raise RetrievalError(
                f"No row found for filter {filter_column}={filter_value!r} on resource '{resource_id}'."
            )
        if total_rows > 1:
            # Filtre ambigu : plusieurs lignes correspondent -- pas de cellule
            # unique, refus plutôt que de prendre la première arbitrairement.
            raise RetrievalError(
                f"Filter {filter_column}={filter_value!r} matched {total_rows} rows; "
                "expected exactly one for a traceable cell."
            )

        row = _parse_first_row(text)
        if target_column not in row:
            raise RetrievalError(
                f"Column '{target_column}' not found in resource '{resource_id}' "
                f"(available: {', '.join(row.keys())})."
            )

        cell_value = row[target_column]
        if not cell_value:
            raise RetrievalError(f"Cell '{target_column}' is empty for this row.")

        return Passage(
            source_id=resource_id,
            source_type="donnee",
            opposable=True,  # §6ter : autorité de mesure, jamais AUTHENTIFIÉ mais publiable
            text=cell_value,
            metadata={
                "dataset_id": dataset_id,
                "resource_id": resource_id,
                "filter_column": filter_column,
                "filter_value": filter_value,
                "target_column": target_column,
                "row": row,
            },
        )


def _coerce_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    raise RetrievalError("Unexpected response shape from query_resource_data.")
