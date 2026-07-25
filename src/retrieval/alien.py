from __future__ import annotations

import json
import re
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

# Le texte MedRxiv/BioRxiv arrive en markdown de conversion PDF : titres de
# section, marqueurs de citation (`[[1,`, `[15,`), ancres de renvoi
# (`(bibref:c3)`, `(figref:fig6)`), et des phrases dupliquées par le
# pipeline d'extraction. Gemma s'y accroche : testé en direct, il ressortait
# « Introduction », « Discussion » ou « [10](bibref:c10) » comme AFFIRMATIONS,
# parce que ce sont les fragments les plus courts et les plus littéralement
# citables du passage. On nettoie AVANT que Gemma voie le texte, donc le
# vérificateur compare au même texte nettoyé : la garantie verbatim (§7) tient
# toujours, elle porte simplement sur de la prose plutôt que sur du balisage.
_REF_ANCHOR_RE = re.compile(r"\((?:bibref|figref):[^)]*\)")
_CITATION_MARKER_RE = re.compile(r"\[+\s*(?:c)?\d+[\d,\s;:–—-]*\]*")
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s.*$", re.MULTILINE)
# Les marqueurs sont imbriqués et déséquilibrés dans la source (`[[1,`, `.]]`) :
# une passe de regex laisse des crochets orphelins, qu'on retire ensuite.
_STRAY_BRACKET_RE = re.compile(r"[\[\]]+")
# ... et la ponctuation qu'ils laissaient derrière eux (` ,.` -> `.`), sans quoi
# deux phrases identiques ne se dédupliquent pas.
_ORPHAN_PUNCT_RE = re.compile(r"\s+([,;.])")
_PUNCT_RUN_RE = re.compile(r"([,;])\s*\.")
# Exactement deux points, jamais trois : « ... » reste intact, le vérificateur
# s'en sert pour repérer une citation épissée (§7, anti-épissage).
_DOUBLE_PERIOD_RE = re.compile(r"(?<!\.)\.\s*\.(?!\.)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")



def _clean_chunk_text(raw: str) -> str:
    text = _MARKDOWN_HEADING_RE.sub("", raw)
    text = _REF_ANCHOR_RE.sub("", text)
    text = _CITATION_MARKER_RE.sub("", text)
    text = _STRAY_BRACKET_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _ORPHAN_PUNCT_RE.sub(r"\1", text)
    text = _PUNCT_RUN_RE.sub(".", text)
    text = _DOUBLE_PERIOD_RE.sub(".", text)
    text = _BLANKLINES_RE.sub("\n\n", text)

    # Le pipeline d'extraction répète des phrases entières à l'identique dans
    # un même chunk. Les garder gonfle le contexte injecté sans rien apporter,
    # et fait remonter deux fois la même affirmation.
    seen: set[str] = set()
    kept: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        stripped = sentence.strip()
        if not stripped:
            continue
        key = stripped.casefold()
        if key in seen:
            continue
        seen.add(key)
        kept.append(stripped)

    return " ".join(kept).strip()


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
            # Un chunk qui ne contenait qu'un titre de section ne laisse rien
            # après nettoyage : ce n'était pas une source, c'était du sommaire.
            # On ne filtre QUE sur ce critère -- un seuil de longueur jetterait
            # aussi un résultat court mais réel ("RR=0.68, p<0.01").
            cleaned = _clean_chunk_text(text)
            if not cleaned:
                continue
            metadata = item.get("metadata") or {}
            chunks.append({
                "entry_id": str(metadata.get("entry_id", item.get("id", ""))),
                "score": float(item.get("score", 0.0)),
                "text": cleaned,
            })
        return chunks
