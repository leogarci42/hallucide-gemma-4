from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from hallucide.core_types.exceptions import RetrievalError
from hallucide.core_types.types import Intent, Passage, RetrievalState

DEFAULT_ALIEN_BASE_URL = "https://api.alien.club"
DEFAULT_ALIEN_SCORE_THRESHOLD = 0.7
DEFAULT_ALIEN_LIMIT = 10

_CHUNK_SEPARATOR = "\n\n---\n\n"


class AlienRetrievalProvider:
    """RetrievalProvider adossé à la recherche sémantique MCP Alien
    Intelligence (études médicales BioRxiv/MedRxiv), sur le modèle de
    `MoulineuseRetrievalProvider` mais en REST direct (stdlib `urllib`
    uniquement, même convention que les providers LLM).

    RAG MASSIF (§4 hackathon) : contrairement aux autres providers qui
    renvoient un seul extrait ciblé, `retrieve()` agrège TOUS les chunks
    retournés par Alien en un seul `Passage` (texte concaténé, chunks
    individuels conservés en `metadata["chunks"]`). C'est ce gros volume de
    passages, injecté d'un coup dans le prompt Gemma, qui exploite le grand
    contexte (§2). Le moteur de vérif (verifier.py) n'est pas modifié : un
    claim AUTHENTIFIÉ doit être un sous-segment contigu de CE texte agrégé,
    ce qui reste vrai qu'il y ait un ou N chunks concaténés dedans.

    `query["dataset_id"]` est fourni par l'appelant (étage 1 : Gemma choisit
    le dataset dans une liste fermée, ou "AUCUN" -> aucun appel ici). Un seul
    dataset par requête (§11 : périmètre de vérité net).
    """

    def __init__(
        self,
        api_token: str,
        cluster_id: str,
        base_url: str = DEFAULT_ALIEN_BASE_URL,
        score_threshold: float = DEFAULT_ALIEN_SCORE_THRESHOLD,
        limit: int = DEFAULT_ALIEN_LIMIT,
    ) -> None:
        self.api_token = api_token
        self.cluster_id = cluster_id
        self.base_url = base_url.rstrip("/")
        self.score_threshold = score_threshold
        self.limit = limit

    def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]) -> Passage:
        dataset_id = query.get("dataset_id")
        if not dataset_id:
            raise RetrievalError("Alien route requires 'dataset_id' (choisi par le routage étage 1).")

        search_query = query.get("query") or intent.question
        payload = {
            "query": search_query,
            "dataset_ids": [dataset_id],
            "score_threshold": float(query.get("score_threshold", self.score_threshold)),
            "limit": int(query.get("limit", self.limit)),
        }

        try:
            response_data = self._send_request(payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RetrievalError(
                f"Alien API error: {exc.code} {exc.reason} for dataset '{dataset_id}': {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RetrievalError(f"Alien API connection error: {exc.reason}") from exc

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
                "source": "Alien Intelligence (recherche sémantique)",
            },
        )

    def _send_request(self, payload: dict[str, Any]) -> Any:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/clusters/{self.cluster_id}/proxy/api/v1/vector/chunks",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RetrievalError("Alien API returned malformed JSON.") from exc

    def _extract_chunks(self, data: Any) -> list[dict[str, Any]]:
        items = data
        if isinstance(data, dict):
            # `or` en chaîne traiterait un dataset vide ([], réponse valide
            # mais sans résultat) comme une clé absente et retomberait sur la
            # clé suivante -- distinguer explicitement "absent" de "vide".
            for key in ("chunks", "data", "results"):
                if key in data:
                    items = data[key]
                    break
            else:
                items = None
        if not isinstance(items, list):
            raise RetrievalError("Unexpected response shape from Alien vector/chunks.")

        chunks: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            text = item.get("chunk_text")
            if not isinstance(text, str) or not text.strip():
                continue
            chunks.append({
                "entry_id": str(item.get("entry_id", "")),
                "score": float(item.get("score", 0.0)),
                "text": text.strip(),
            })
        return chunks
