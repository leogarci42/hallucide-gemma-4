from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hallucide.core_types.types import Passage, RetrievalState

# §4ter : types de lien Légifrance pertinents pour suivre un renvoi normatif
# réel vers un article de code (la seule cible que MoulineuseRetrievalProvider
# sait récupérer, route code_article). Exclut "MODIFIE" (pointe vers le texte
# modificateur -- souvent une ordonnance/décret JORF, pas un code) et "source"
# (remonte le passé de l'article courant, pas un renvoi applicatif).
_FOLLOWABLE_LIEN_TYPES = {"CITATION", "CONCORDANCE"}
_FOLLOWABLE_NATURE_TEXTE = {"CODE"}


@dataclass(frozen=True)
class NextHop:
    """Un renvoi normatif suivable (§4ter), extrait des métadonnées d'un
    Passage Moulineuse -- jamais deviné, toujours présent tel quel dans les
    LIENS du texte officiel.
    """

    article_num: str
    code_cid: str
    lien_type: str
    description: str


def extract_followable_hops(passage: Passage) -> list[NextHop]:
    """Renvois normatifs suivables présents dans le passage, jamais inférés.

    Double filtre déterministe, jamais une heuristique sur le contenu :
    @typelien (CITATION/CONCORDANCE) ET @naturetexte=CODE, pour ne retenir
    que les renvois réellement adressables par la route code_article (qui
    n'accepte que des articles de code, jamais une ordonnance/décret JORF).
    """
    liens = passage.metadata.get("liens") or []
    hops: list[NextHop] = []
    for lien in liens:
        if not isinstance(lien, dict):
            continue
        lien_type = lien.get("@typelien")
        nature_texte = lien.get("@naturetexte")
        article_num = lien.get("@num")
        code_cid = lien.get("@cidtexte")
        if (
            lien_type not in _FOLLOWABLE_LIEN_TYPES
            or nature_texte not in _FOLLOWABLE_NATURE_TEXTE
            or not article_num
            or not code_cid
        ):
            continue
        hops.append(
            NextHop(
                article_num=str(article_num),
                code_cid=str(code_cid),
                lien_type=str(lien_type),
                description=str(lien.get("#text", "")),
            )
        )
    return hops


def select_next_hop(passage: Passage, state: RetrievalState) -> NextHop | None:
    """Sélectionne le prochain saut (§4ter), ou None si la boucle doit
    s'arrêter -- jamais de "rapprochement opportuniste" : si aucune
    référence NOUVELLE n'est trouvée (déjà visitée, ou aucune suivable),
    le contrôle s'arrête ici et l'appelant doit traiter cela comme un
    refus potentiel (§7bis), pas comme une réponse "à peu près pertinente".

    Choisit le premier renvoi non visité, sans aucun critère de pertinence
    par rapport à la question -- §5 : "le LLM peut proposer le prochain saut
    (c'est de l'intelligence), mais c'est l'orchestrateur qui décide d'arrêter".
    Cette fonction ne fait que la partie bornage (§4ter) ; le choix éclairé
    du renvoi le plus pertinent pour la question de l'utilisateur, s'il y en
    a plusieurs, reste à la charge de l'appelant (LLM ou humain), pas de ce
    module déterministe.
    """
    for hop in extract_followable_hops(passage):
        if hop.code_cid not in state.visited_documents:
            return hop
    return None


def build_hop_query(hop: NextHop, code_title_hint: str) -> dict[str, str]:
    """Construit la query structurée (§4bis) pour suivre ce renvoi via la
    route code_article existante -- le cid identifie le texte cible de façon
    déterministe, mais MoulineuseRetrievalProvider attend un titre de code en
    LIKE ; `code_title_hint` doit donc être fourni par l'appelant (résolu via
    une table de correspondance cid -> titre, ou directement par l'humain qui
    initie la requête), jamais deviné depuis le numéro d'article seul.
    """
    return {"route": "code_article", "article": hop.article_num, "code": code_title_hint}
