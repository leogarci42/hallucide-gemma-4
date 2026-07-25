from __future__ import annotations

from typing import Any

from hallucide.core_types.exceptions import RetrievalError
from hallucide.retrieval.mcp_client import McpToolClient
from hallucide.retrieval.moulineuse import _strip_html
from hallucide.verification.semantic_similarity import similarity_score
from hallucide.core_types.types import Intent, Passage, RetrievalState

# Serveur MCP « Parlement » (LegiWatch) : expose les interventions verbatim en
# séance (comptes rendus), distinct de la Moulineuse (code4code.eu) qui n'a que
# l'agenda/les métadonnées. Base Canutes Parlement.
DEFAULT_PARLEMENT_MCP_URL = "https://parlement.tricoteuses.fr/mcp"

_AMBIGUITY_MARGIN = 0.08


def _results(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if "error" in payload:
            raise RetrievalError(f"MCP Parlement error: {payload['error']}")
        r = payload.get("results")
        if isinstance(r, list):
            return r
    raise RetrievalError("Unexpected response shape from MCP Parlement.")


class InterventionsRetrievalProvider:
    """RetrievalProvider pour les interventions verbatim en séance publique
    (comptes rendus), via le serveur MCP Parlement.

    Une intervention en débat est un acte parlementaire : le texte existe à
    l'identique dans le compte rendu officiel, mais n'a AUCUNE autorité
    normative (§6ter) -> opposable=False, au mieux CITÉ_NON_OPPOSABLE. On ne
    prouve donc jamais « c'est la loi », seulement « ceci a bien été dit, mot
    pour mot, par tel orateur, à telle séance ».

    Flux : (optionnel) résoudre l'orateur -> acteurRefUid, puis rechercher ses
    interventions sur le sujet, reclasser par proximité lexicale déterministe
    (similarity_score), renvoyer la meilleure comme passage verbatim.
    """

    def __init__(self, client: McpToolClient | None = None) -> None:
        self.client = client or McpToolClient(DEFAULT_PARLEMENT_MCP_URL)

    def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]) -> Passage:
        search = query.get("search") or intent.question
        if not search:
            raise RetrievalError("intervention route requires 'search' (ou une question).")

        args: dict[str, Any] = {"search": search, "perPage": 10}
        orateur = query.get("orateur")
        acteur_uid = None
        if orateur:
            acteur_uid = self._resolve_acteur(orateur)
            if acteur_uid:
                args["acteurRefUid"] = acteur_uid

        result = self.client.call_tool("search_interventions", args)
        rows = _results(result)
        if not rows:
            who = f" de « {orateur} »" if orateur else ""
            raise RetrievalError(f"Aucune intervention trouvée{who} pour « {search} ».")

        best, selection_ambiguous = self._rerank(rows, search)
        texte = _strip_html(best.get("texte"))
        if not texte:
            raise RetrievalError("L'intervention retenue n'a pas de texte exploitable.")

        return Passage(
            source_id=str(best.get("uid")),
            source_type="debat",
            opposable=False,  # intervention en débat : jamais opposable (§6ter)
            text=texte,
            metadata={
                "orateur": best.get("orateur"),
                "date_seance": best.get("dateSeance"),
                "reunion_ref": best.get("reunionRefUid"),
                "dossier_ref": best.get("dossierRefUid"),
                "lien_legiwatch": best.get("lienLegiwatch"),
                "acteur_ref": best.get("acteurRefUid") or acteur_uid,
                # Débat = source par proximité, jamais identité exacte du sujet :
                # la fidélité (« ceci a été dit ») est garantie, pas la pertinence
                # (« ceci répond à la question ») -> même garde-fou que texte_libre.
                "pertinence_non_garantie": True,
                "candidate_count": len(rows),
                "selection_ambiguous": selection_ambiguous,
            },
        )

    def _resolve_acteur(self, name: str) -> str | None:
        """Nom d'orateur -> acteurRefUid (meilleur match), ou None si introuvable."""
        try:
            res = self.client.call_tool("search_acteurs", {"search": name, "perPage": 3})
        except RetrievalError:
            return None
        results = res.get("results") if isinstance(res, dict) else None
        if isinstance(results, list) and results:
            first = results[0]
            return first.get("uid") or first.get("refUid")
        return None

    def _rerank(self, rows: list[dict[str, Any]], search: str) -> tuple[dict[str, Any], bool]:
        """Reclasse les interventions par proximité lexicale du texte au sujet
        (déterministe). Ambigu si les deux meilleures sont à égalité serrée."""
        scored = sorted(rows, key=lambda r: similarity_score(_strip_html(r.get("texte")), search), reverse=True)
        if len(scored) < 2:
            return scored[0], False
        top = similarity_score(_strip_html(scored[0].get("texte")), search)
        second = similarity_score(_strip_html(scored[1].get("texte")), search)
        return scored[0], (top - second) < _AMBIGUITY_MARGIN
