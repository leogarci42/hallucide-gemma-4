from __future__ import annotations

import json
from typing import Any

from hallucide.core_types.exceptions import RetrievalError
from hallucide.retrieval.mcp_client import McpToolClient
from hallucide.core_types.types import Intent, Passage, RetrievalState

# Serveur MCP officiel du hackathon (get.alien.club/gemma4-hackathon) : deux
# clusters pré-indexés, un par endpoint, sans passer par la config manuelle
# de cluster (qui échouait côté plateforme, "Failed to add datasets to your
# library"). Défaut = MedRxiv, dont les catégories (Oncology, Cardiovascular_
# Medicine, Neurology, ...) collent au périmètre "assistant médical" -- BioRxiv
# reste dispo en pointant MCP_URL vers biorxiv.mcp.alien.club/mcp.
DEFAULT_ALIEN_MCP_URL = "https://medrxiv.mcp.alien.club/mcp"
# Observé en direct sur des requêtes pertinentes réelles : les scores tournent
# autour de 0.5-0.7 (embeddings gemini-embedding-001 sur du texte scientifique
# anglais), pas 0.9+. Un seuil à 0.7 par défaut coupait des résultats
# manifestement pertinents -- abaissé pour ne pas sous-remonter le RAG massif.
DEFAULT_ALIEN_SCORE_THRESHOLD = 0.4
DEFAULT_ALIEN_LIMIT = 10

_CHUNK_SEPARATOR = "\n\n---\n\n"


class AlienRetrievalProvider:
    """RetrievalProvider adossé au serveur MCP Alien Intelligence (études
    médicales BioRxiv/MedRxiv), sur le modèle de MoulineuseRetrievalProvider
    (McpToolClient), pas d'appel REST direct -- l'API REST décrite dans le
    brief initial n'est pas ce que le hackathon expose réellement ; le vrai
    accès passe par ces endpoints MCP dédiés (get.alien.club/gemma4-hackathon).

    RAG MASSIF (§4 hackathon) : `retrieve()` agrège TOUS les chunks renvoyés
    par `datacluster_vector_search_chunks` en un seul `Passage` (texte
    concaténé, chunks individuels gardés en `metadata["chunks"]`). Le moteur
    de vérif (verifier.py) n'est pas modifié : un claim AUTHENTIFIÉ doit être
    un sous-segment contigu de CE texte agrégé, ce qui reste vrai qu'il y ait
    un ou N chunks concaténés dedans.

    `query["dataset_id"]` est l'ID NUMÉRIQUE du dataset (ex: "30" pour
    Oncology sur MedRxiv) fourni par l'appelant (étage 1 : Gemma choisit un
    nom de domaine dans une liste fermée, le code le traduit en ID via
    `DOMAINS`/`DATASET_IDS` -- voir scripts/ask_medical.py). Un seul dataset
    par requête (§11 : périmètre de vérité net).
    """

    def __init__(
        self,
        api_token: str,
        mcp_url: str = DEFAULT_ALIEN_MCP_URL,
        client: McpToolClient | None = None,
        score_threshold: float = DEFAULT_ALIEN_SCORE_THRESHOLD,
        limit: int = DEFAULT_ALIEN_LIMIT,
    ) -> None:
        self.api_token = api_token
        self.mcp_url = mcp_url
        self.client = client or McpToolClient(mcp_url, headers={"Authorization": f"Bearer {api_token}"})
        self.score_threshold = score_threshold
        self.limit = limit

    def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]) -> Passage:
        dataset_id = query.get("dataset_id")
        if not dataset_id:
            raise RetrievalError("Alien route requires 'dataset_id' (choisi par le routage étage 1).")

        search_query = query.get("query") or intent.question
        arguments = {
            "query": search_query,
            "dataset_ids": [str(dataset_id)],
            "score_threshold": float(query.get("score_threshold", self.score_threshold)),
            "limit": int(query.get("limit", self.limit)),
        }

        response_data = self.client.call_tool("datacluster_vector_search_chunks", arguments)
        chunks = self._extract_chunks(response_data)
        if not chunks:
            raise RetrievalError(
                f"Aucun passage trouvé dans le dataset '{dataset_id}' pour la requête « {search_query} »."
            )

        # Pas d'en-tête inline ("[Source N — score...]") devant chaque chunk :
        # testé en direct contre Gemma, un en-tête entre crochets en tête de
        # chunk est repris tel quel comme affirmation "AUTHENTIFIÉ" (le petit
        # modèle le traite comme la citation la plus courte/salante plutôt que
        # le contenu réel). Les métadonnées (score, entry_id) restent
        # disponibles pour l'audit/affichage via metadata["chunks"], juste pas
        # dans le texte que Gemma lit pour générer les affirmations.
        aggregated_text = _CHUNK_SEPARATOR.join(c["text"] for c in chunks)

        return Passage(
            source_id=dataset_id,
            source_type="etude_medicale",
            # Décision produit (pas une contrainte technique) : une citation
            # verbatim d'étude, confirmée par le vérificateur déterministe,
            # est traitée comme suffisamment fiable pour un affichage direct
            # (AUTHENTIFIÉ) plutôt que de toujours exiger une validation
            # humaine (CITÉ_NON_OPPOSABLE -> risque élevé). Contrairement aux
            # sources non normatives de MoulineuseRetrievalProvider (§6ter,
            # ex. question parlementaire = jamais d'autorité), une étude
            # correctement citée EST la source de vérité de ce domaine -- il
            # n'y a pas d'équivalent "loi en vigueur" au-dessus d'elle.
            opposable=True,
            text=aggregated_text,
            metadata={
                "dataset_id": dataset_id,
                "query": search_query,
                "nb_passages": len(chunks),
                "nb_chars": len(aggregated_text),
                "chunks": chunks,
                "source": f"Alien Intelligence ({self.mcp_url})",
            },
        )

    def _extract_chunks(self, data: Any) -> list[dict[str, Any]]:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                raise RetrievalError("Alien MCP returned malformed JSON.")

        items = None
        if isinstance(data, dict):
            payload = data.get("data") if isinstance(data.get("data"), dict) else data
            if isinstance(payload, dict) and "results" in payload:
                items = payload["results"]
        if not isinstance(items, list):
            raise RetrievalError(f"Unexpected response shape from Alien datacluster_vector_search_chunks: {data!r}")

        chunks: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            text = item.get("chunk_text")
            if not isinstance(text, str) or not text.strip():
                continue
            metadata = item.get("metadata") or {}
            chunks.append({
                "entry_id": str(metadata.get("entry_id", item.get("id", ""))),
                "score": float(item.get("score", 0.0)),
                "text": text.strip(),
            })
        return chunks
