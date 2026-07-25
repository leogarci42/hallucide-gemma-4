from __future__ import annotations

from typing import Any

from hallucide.retrieval.datagouv import DataGouvRetrievalProvider
from hallucide.core_types.exceptions import RetrievalError
from hallucide.retrieval.file_retrieval import FileRetrievalProvider
from hallucide.retrieval.interventions import InterventionsRetrievalProvider
from hallucide.retrieval.moulineuse import MoulineuseRetrievalProvider
from hallucide.core_types.types import Intent, Passage, RetrievalState


class MultiSourceRetrievalProvider:
    """Aiguille entre les sources réelles (§6ter) selon les clés présentes
    dans `query`, jamais en devinant depuis le texte de la question (§4bis,
    politique "structuré par défaut") :

    - query["route"] présent ("pastille"/"code_article"/"texte_libre"/
      "parlement_question") -> Moulineuse, texte normatif/parlementaire.
    - query["filters"] (dict) présent -> FileRetrievalProvider : ressource
      data.gouv non-tabulaire (CSV/ZIP-CSV téléchargé et parsé).
    - query["target_column"] présent (sans "filters") -> data.gouv API tabulaire.

    L'appelant choisit la source en construisant la query, exactement comme
    il choisit déjà la route à l'intérieur de Moulineuse -- aucune nouvelle
    heuristique de routage à faire confiance.
    """

    def __init__(
        self,
        moulineuse: MoulineuseRetrievalProvider | None = None,
        datagouv: DataGouvRetrievalProvider | None = None,
        file_provider: FileRetrievalProvider | None = None,
        interventions: InterventionsRetrievalProvider | None = None,
    ) -> None:
        self.moulineuse = moulineuse or MoulineuseRetrievalProvider()
        self.datagouv = datagouv or DataGouvRetrievalProvider()
        self.file_provider = file_provider or FileRetrievalProvider()
        self.interventions = interventions or InterventionsRetrievalProvider()

    def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, Any]) -> Passage:
        # La route "intervention" (comptes rendus verbatim) vit sur un AUTRE
        # serveur MCP (parlement.tricoteuses.fr), d'où l'aiguillage explicite.
        if query.get("route") == "intervention":
            return self.interventions.retrieve(intent, state, query)
        if "route" in query:
            return self.moulineuse.retrieve(intent, state, query)
        if "filters" in query:
            return self.file_provider.retrieve(intent, state, query)
        if "target_column" in query:
            return self.datagouv.retrieve(intent, state, query)
        raise RetrievalError(
            "Unable to determine source for query: expected 'route' (Moulineuse), "
            "'filters' (fichier CSV/ZIP) or 'target_column' (data.gouv tabulaire)."
        )
