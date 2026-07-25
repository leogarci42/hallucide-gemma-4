from __future__ import annotations

import html as _html
import re
from typing import Any

from hallucide.core_types.exceptions import RetrievalError
from hallucide.retrieval.mcp_client import McpToolClient
from hallucide.verification.semantic_similarity import similarity_score
from hallucide.verification.slot_provenance import check_slot_provenance
from hallucide.core_types.types import Intent, Passage, RetrievalState

DEFAULT_MOULINEUSE_URL = "https://mcp.code4code.eu/mcp"

# En dessous de cet écart de score entre les deux meilleurs candidats plein
# texte, la sélection est jugée ambiguë (quasi-égalité) et élève le risque.
# Au-dessus, un candidat gagne nettement : pas d'ambiguïté à signaler.
_RERANK_AMBIGUITY_MARGIN = 0.08


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(raw: Any) -> str:
    """Texte brut depuis un fragment HTML (dispositif/exposé d'amendement).
    Retire les balises, décode les entités (&#160; -> espace), normalise les
    blancs -- indispensable pour que la vérification verbatim (§7) compare du
    texte, pas du markup."""
    if not raw or not isinstance(raw, str):
        return ""
    text = _HTML_TAG_RE.sub(" ", raw)
    text = _html.unescape(text)
    text = text.replace(" ", " ")
    return _WS_RE.sub(" ", text).strip()


def _date_seule(raw: Any) -> str | None:
    """« 2022-07-12T00:00:00+02:00 » -> « 2022-07-12 ». None si vide."""
    if not raw or not isinstance(raw, str):
        return None
    return raw.split("T", 1)[0]


def _hit_title(hit: dict[str, Any]) -> str:
    return (hit.get("document") or {}).get("autocompletion") or ""


def _hit_date(hit: dict[str, Any]) -> str | None:
    """Meilleure date disponible sur un hit (clé contenant 'date'), ou None.
    Défensif : le schéma exact de search_legal_texts n'est pas garanti."""
    document = hit.get("document") or {}
    for key, value in document.items():
        if "date" in key.lower() and isinstance(value, str) and value:
            return value
    return None


def _rerank_hits(
    hits: list[dict[str, Any]], search_query: str, sort: str
) -> tuple[list[dict[str, Any]], bool]:
    """Réordonne les candidats plein texte de façon DÉTERMINISTE.

    - sort="recent" : tri par date décroissante si au moins une date existe
      (sinon repli sur la pertinence). Un tri explicite par date n'est pas
      une sélection ambiguë de pertinence -> ambigu=False.
    - sort="pertinence" (défaut) : reclassement lexical par proximité du titre
      à la requête (similarity_score, aucun modèle). Ambigu seulement si les
      deux meilleurs sont à égalité serrée (< marge).

    Renvoie (hits_ordonnés, selection_ambiguous).
    """
    if not hits:
        return hits, False

    if sort == "recent" and any(_hit_date(h) for h in hits):
        ordered = sorted(hits, key=lambda h: _hit_date(h) or "", reverse=True)
        return ordered, False

    scored = sorted(hits, key=lambda h: similarity_score(_hit_title(h), search_query), reverse=True)
    if len(scored) < 2:
        return scored, False
    top = similarity_score(_hit_title(scored[0]), search_query)
    second = similarity_score(_hit_title(scored[1]), search_query)
    return scored, (top - second) < _RERANK_AMBIGUITY_MARGIN

# §6ter: opposability is derived deterministically from document class / état,
# never from the model. A code article still in force is the clearest case.
_OPPOSABLE_ETATS = {"VIGUEUR"}

_ARTICLE_NORMALIZE_PATTERN = re.compile(r"[^A-Z0-9-]")


def _normalize_article_number(article: str) -> str:
    return _ARTICLE_NORMALIZE_PATTERN.sub("", article.upper())


def _build_code_article_query(article_normalise: str, code_titre_like: str, date: str | None) -> tuple[str, list[Any]]:
    # Mirrors the documented recipe `legifrance_retrouver_article_code_en_vigueur`:
    # normalize the number, join to the parent code via @cid, then filter by date.
    if date:
        return (
            """
            WITH params AS (
                SELECT $1::text AS article_normalise,
                       $2::text AS code_titre,
                       $3::date AS date_application
            )
            SELECT a.id,
                   a.num,
                   a.data->'META'->'META_SPEC'->'META_ARTICLE'->>'DATE_DEBUT' AS date_debut,
                   a.data->'META'->'META_SPEC'->'META_ARTICLE'->>'DATE_FIN' AS date_fin,
                   a.data->'META'->'META_SPEC'->'META_ARTICLE'->>'ETAT' AS etat,
                   tv.data->'META'->'META_SPEC'->'META_TEXTE_VERSION'->>'TITRE' AS titre_texte,
                   a.data->'BLOC_TEXTUEL'->>'CONTENU' AS contenu,
                   a.data->'LIENS'->'LIEN' AS liens
            FROM legifrance.article a
            LEFT JOIN legifrance.texte_version tv
                   ON tv.id = a.data->'CONTEXTE'->'TEXTE'->>'@cid'
            CROSS JOIN params
            WHERE regexp_replace(upper(a.num), '[^A-Z0-9-]', '', 'g') = params.article_normalise
              AND lower(coalesce(tv.data->'META'->'META_SPEC'->'META_TEXTE_VERSION'->>'TITRE', '')) LIKE params.code_titre
              AND (a.data->'META'->'META_SPEC'->'META_ARTICLE'->>'DATE_DEBUT')::date <= params.date_application
              AND (
                a.data->'META'->'META_SPEC'->'META_ARTICLE'->>'DATE_FIN' IS NULL
                OR (a.data->'META'->'META_SPEC'->'META_ARTICLE'->>'DATE_FIN')::date > params.date_application
              )
            ORDER BY date_debut DESC NULLS LAST, a.id
            LIMIT 5;
            """,
            [article_normalise, code_titre_like, date],
        )

    return (
        """
        WITH params AS (
            SELECT $1::text AS article_normalise,
                   $2::text AS code_titre
        )
        SELECT a.id,
               a.num,
               a.data->'META'->'META_SPEC'->'META_ARTICLE'->>'DATE_DEBUT' AS date_debut,
               a.data->'META'->'META_SPEC'->'META_ARTICLE'->>'DATE_FIN' AS date_fin,
               a.data->'META'->'META_SPEC'->'META_ARTICLE'->>'ETAT' AS etat,
               tv.data->'META'->'META_SPEC'->'META_TEXTE_VERSION'->>'TITRE' AS titre_texte,
               a.data->'BLOC_TEXTUEL'->>'CONTENU' AS contenu,
               a.data->'LIENS'->'LIEN' AS liens
        FROM legifrance.article a
        LEFT JOIN legifrance.texte_version tv
               ON tv.id = a.data->'CONTEXTE'->'TEXTE'->>'@cid'
        CROSS JOIN params
        WHERE regexp_replace(upper(a.num), '[^A-Z0-9-]', '', 'g') = params.article_normalise
          AND lower(coalesce(tv.data->'META'->'META_SPEC'->'META_TEXTE_VERSION'->>'TITRE', '')) LIKE params.code_titre
        ORDER BY date_debut DESC NULLS LAST, a.id
        LIMIT 5;
        """,
        [article_normalise, code_titre_like],
    )


class MoulineuseRetrievalProvider:
    """RetrievalProvider backed by the Moulineuse MCP server (§6bis).

    Three routes, selected explicitly via query["route"] (never inferred,
    per the §4bis "structured by default" policy):

    - route="pastille": parliamentary article via get_pastilled_article.
      Requires documentUid or sourceUrl; refuses otherwise rather than
      guessing a document (the tool's own safe-by-construction behavior).
    - route="code_article": consolidated code article via the SQL path
      on legifrance.article, multi-step per the documented recipe
      (normalize number -> resolve parent code -> filter by date).
    - route="texte_libre": full-text fallback via search_legal_texts when
      no safe structured field is identifiable (§4bis). Always flagged
      "pertinence non garantie" -- this route retrieves by semantic
      proximity, which can surface a real but off-topic source (piège C1,
      reproduced live: "congé menstruel" -> 1943-1962 military leave decrees).
    """

    def __init__(self, client: McpToolClient | None = None) -> None:
        self.client = client or McpToolClient(DEFAULT_MOULINEUSE_URL)

    def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]) -> Passage:
        route = query.get("route")
        if route == "pastille":
            return self._retrieve_pastille(query)
        if route == "code_article":
            return self._retrieve_code_article(intent, query)
        if route == "texte_libre":
            return self._retrieve_texte_libre(query)
        if route == "parlement_question":
            return self._retrieve_parlement_question(query)
        if route == "amendement":
            return self._retrieve_amendement(query)
        if route == "commissions":
            return self._retrieve_commissions(query)
        if route == "mandat":
            return self._retrieve_mandat(query)
        raise RetrievalError(
            f"Unknown or missing route '{route}'; expected 'pastille', 'code_article', "
            "'texte_libre', 'parlement_question', 'amendement', 'commissions' or 'mandat'."
        )

    # SQL (recette Tricoteuses) : les mandats sont dans acteurs.data->'mandats',
    # chaque mandat porte organesRefs + dateDebut/dateFin ; on joint organes pour
    # le libellé et on ne garde que les commissions (libellé « Commission … »).
    _COMMISSIONS_SQL = """
        SELECT DISTINCT
               o.data->>'libelle' AS libelle,
               o.data->>'codeType' AS code_type,
               m->'infosQualite'->>'libQualiteSex' AS qualite,
               m->>'dateDebut' AS date_debut,
               m->>'dateFin' AS date_fin
        FROM assemblee.acteurs a
        CROSS JOIN LATERAL jsonb_array_elements(a.data->'mandats') AS m
        CROSS JOIN LATERAL jsonb_array_elements_text(m->'organesRefs') AS oref
        JOIN assemblee.organes o ON o.uid = oref
        WHERE a.uid = $1
          AND lower(o.data->>'libelle') LIKE 'commission%'
        ORDER BY date_debut;
    """

    def _retrieve_commissions(self, query: dict[str, str]) -> Passage:
        # Appartenances aux commissions d'un député, avec dates (open data
        # Assemblée). Acte administratif -> opposable=False (CITÉ_NON_OPPOSABLE) :
        # on prouve « cette appartenance figure exactement dans la donnée
        # officielle », jamais une autorité normative.
        acteur = query.get("acteur")
        acteur_ref = query.get("acteur_ref") or (self._resolve_acteur(acteur) if acteur else None)
        if not acteur_ref:
            raise RetrievalError(f"Député introuvable : « {acteur or ''} ».")

        result = self.client.call_tool(
            "query_sql", {"schema": "assemblee", "query": self._COMMISSIONS_SQL, "params": [acteur_ref]}
        )
        rows = _parse_sql_rows(result)
        lignes: list[str] = []
        for r in rows:
            libelle = (r.get("libelle") or "").strip()
            if not libelle:
                continue
            qualite = (r.get("qualite") or "Membre").strip()
            debut = _date_seule(r.get("date_debut")) or "?"
            fin = _date_seule(r.get("date_fin")) or "en cours"
            lignes.append(f"{libelle} — {qualite} — du {debut} au {fin}")
        lignes = list(dict.fromkeys(lignes))
        if not lignes:
            raise RetrievalError(f"Aucune appartenance à une commission trouvée pour l'acteur {acteur_ref}.")

        # Question ciblée (« est-ce que X a fait partie de la commission Y ? ») :
        # on filtre sur la commission nommée et on répond oui (lignes) / non.
        total = len(lignes)
        cible = (query.get("commission") or "").strip()
        reponse = "liste"
        if cible:
            cible_norm = cible.lower()
            filtrees = [l for l in lignes if cible_norm in l.lower()]
            if filtrees:
                reponse, affichees = "oui", filtrees
            else:
                reponse = "non"
                affichees = [f"Aucune appartenance à une commission « {cible} » dans les "
                             f"{total} appartenances officielles de {acteur or acteur_ref}."]
        else:
            affichees = lignes

        return Passage(
            source_id=str(acteur_ref),
            source_type="mandat",
            opposable=False,  # acte administratif, pas texte normatif (§6ter)
            text="\n".join(affichees),
            metadata={
                "acteur": acteur or "",
                "acteur_ref": acteur_ref,
                "nb_commissions": len(affichees),
                "nb_total": total,
                "commission_cible": cible or None,
                "reponse": reponse,
                "toutes_appartenances": lignes,  # liste complète (traçabilité)
                "source": "Open data Assemblée nationale (mandats)",
            },
        )

    _MANDATS_SQL = """
        SELECT DISTINCT
               o.data->>'libelle' AS libelle,
               m->'infosQualite'->>'libQualiteSex' AS qualite,
               m->>'dateDebut' AS date_debut,
               m->>'dateFin' AS date_fin
        FROM assemblee.acteurs a
        CROSS JOIN LATERAL jsonb_array_elements(a.data->'mandats') AS m
        CROSS JOIN LATERAL jsonb_array_elements_text(m->'organesRefs') AS oref
        JOIN assemblee.organes o ON o.uid = oref
        WHERE a.uid = $1
          AND o.data->>'codeType' = $2
        ORDER BY date_debut;
    """

    # fonction demandée -> type d'organe dans l'open data Assemblée
    _FONCTION_ORGANE = {"depute": "ASSEMBLEE", "ministre": "MINISTERE"}

    def _retrieve_mandat(self, query: dict[str, str]) -> Passage:
        # Mandats de député d'un acteur (open data Assemblée). Même contrat que
        # la route commissions : donnée déterministe, acte administratif ->
        # opposable=False. L'ABSENCE de mandat est aussi une réponse tracée :
        # « X est-il député ? » -> non, aucun mandat dans la donnée officielle.
        acteur = (query.get("acteur") or "").strip()
        fonction = (query.get("fonction") or "depute").strip()
        organe = self._FONCTION_ORGANE.get(fonction, "ASSEMBLEE")
        acteur_ref = query.get("acteur_ref") or (self._resolve_acteur(acteur) if acteur else None)

        lignes: list[str] = []
        if acteur_ref:
            result = self.client.call_tool(
                "query_sql", {"schema": "assemblee", "query": self._MANDATS_SQL, "params": [acteur_ref, organe]}
            )
            for r in _parse_sql_rows(result):
                qualite = (r.get("qualite") or "Député").strip()
                debut = _date_seule(r.get("date_debut")) or "?"
                fin = _date_seule(r.get("date_fin")) or "en cours"
                lignes.append(f"{(r.get('libelle') or 'Assemblée nationale').strip()} — {qualite} — du {debut} au {fin}")
            lignes = list(dict.fromkeys(lignes))

        if not lignes:
            libf = "député" if fonction == "depute" else "ministre"
            lignes = [f"Aucun mandat de {libf} trouvé pour « {acteur or acteur_ref} » "
                      "dans l'open data officiel de l'Assemblée nationale."]

        texte = "\n".join(lignes)
        return Passage(
            source_id=str(acteur_ref or f"acteur:{acteur}"),
            source_type="mandat",
            opposable=False,
            text=texte,
            metadata={"acteur": acteur or None, "acteur_ref": acteur_ref,
                      "nb_mandats": len(lignes) if acteur_ref else 0,
                      "source": "assemblee.acteurs (open data officiel)"},
        )

    def _resolve_acteur(self, name: str) -> str | None:
        """Nom -> uid acteur via nom_complet (recette Tricoteuses)."""
        rows = _parse_sql_rows(
            self.client.call_tool(
                "query_sql",
                {"schema": "assemblee",
                 "query": "SELECT uid FROM assemblee.acteurs WHERE nom_complet ILIKE $1 ORDER BY nom_complet LIMIT 1",
                 "params": [f"%{name}%"]},
            )
        )
        return rows[0].get("uid") if rows else None

    def _retrieve_amendement(self, query: dict[str, str]) -> Passage:
        # Amendement parlementaire (assemblee.amendements) : le DISPOSITIF est le
        # texte verbatim déposé ; l'amendement est un acte parlementaire -> non
        # opposable (§6ter), au mieux CITÉ_NON_OPPOSABLE. On récupère le texte,
        # l'auteur et le SORT (Adopté/Rejeté/Tombé...) -- des faits tracés, pas
        # la « position » orale d'un ministre (absente de cette source).
        numero = query.get("numero")
        if not numero:
            raise RetrievalError("amendement route requires 'numero'.")
        legislature = query.get("legislature")

        params: list[Any] = [str(numero)]
        where = "data->'identification'->>'numeroLong' = $1"
        if legislature:
            where += " AND legislature = $2"
            params.append(int(legislature))
        sql = f"""
            SELECT uid,
                   data->'identification'->>'numeroLong' AS numero,
                   data->'cycleDeVie'->>'sort' AS sort,
                   data->'signataires'->'auteur'->>'typeAuteur' AS type_auteur,
                   data->'corps'->'contenuAuteur'->>'dispositif' AS dispositif,
                   data->'corps'->'contenuAuteur'->>'exposeSommaire' AS expose,
                   data->>'texteLegislatifRef' AS texte_ref,
                   legislature
            FROM assemblee.amendements
            WHERE {where}
            ORDER BY legislature DESC, uid
            LIMIT 10;
        """
        result = self.client.call_tool("query_sql", {"schema": "assemblee", "query": sql, "params": params})
        rows = _parse_sql_rows(result)
        if not rows:
            raise RetrievalError(f"No amendment found for numero '{numero}'.")

        row = rows[0]
        dispositif = _strip_html(row.get("dispositif"))
        if not dispositif:
            raise RetrievalError(f"Amendment '{row.get('uid')}' has no extractable dispositif.")

        # « n° 245 » sans texte de rattachement matche potentiellement un
        # amendement par dossier/législature -- prendre rows[0] serait une
        # sélection silencieuse (piège A3). Signalé pour élever le risque.
        distinct_textes = {r.get("texte_ref") for r in rows}
        selection_ambiguous = len(distinct_textes) > 1

        return Passage(
            source_id=str(row.get("uid")),
            source_type="normatif",
            opposable=False,  # acte parlementaire, jamais opposable (§6ter)
            text=dispositif,
            metadata={
                "numero": row.get("numero"),
                "sort": row.get("sort"),
                "type_auteur": row.get("type_auteur"),
                "expose_sommaire": _strip_html(row.get("expose")),
                "texte_ref": row.get("texte_ref"),
                "legislature": row.get("legislature"),
                "candidate_count": len(rows),
                "selection_ambiguous": selection_ambiguous,
            },
        )

    def _retrieve_parlement_question(self, query: dict[str, str]) -> Passage:
        # Question parlementaire (QE/QOSD/QG) via get_parlement_item : le texte
        # existe à l'identique dans la source, mais une question parlementaire
        # n'a AUCUNE autorité normative (§6ter) -> opposable=False, statut au
        # mieux CITÉ_NON_OPPOSABLE, jamais AUTHENTIFIÉ.
        uid = query.get("uid")
        if not uid:
            raise RetrievalError("parlement_question route requires 'uid'.")

        result = self.client.call_tool("get_parlement_item", {"resource": "questions", "id": uid})
        data = _extract_parlement_question(result)
        text = data.get("texteQuestionNettoye") or data.get("texteQuestion")
        if not text:
            raise RetrievalError(f"Question '{uid}' has no extractable text.")

        return Passage(
            source_id=str(data.get("uid") or uid),
            source_type="normatif",
            opposable=False,  # non opposable par nature (§6ter)
            text=text,
            metadata={
                "titre": data.get("titre"),
                "rubrique": data.get("rubrique"),
                "type": data.get("type"),
                "numero": data.get("numero"),
                "chambre": data.get("chambre"),
                "date_depot": data.get("dateDepot"),
            },
        )

    def _retrieve_pastille(self, query: dict[str, str]) -> Passage:
        chambre = query.get("chambre")
        article = query.get("article")
        if not chambre or not article:
            raise RetrievalError("Pastille route requires 'chambre' and 'article'.")

        document_uid = query.get("documentUid")
        source_url = query.get("sourceUrl")
        if not document_uid and not source_url:
            # Matches the tool's own refusal: never guess a document identity.
            raise RetrievalError(
                "Pastille route requires 'documentUid' or 'sourceUrl'; refusing rather than guessing."
            )

        arguments: dict[str, Any] = {"chambre": chambre, "article": article}
        if document_uid:
            arguments["documentUid"] = document_uid
        if source_url:
            arguments["sourceUrl"] = source_url
        if "alinea" in query:
            arguments["alinea"] = query["alinea"]
        if "date" in query:
            arguments["date"] = query["date"]

        result = self.client.call_tool("get_pastilled_article", arguments)
        text, metadata = _extract_pastille_text(result)

        source_id = document_uid or source_url or f"{chambre}:{article}"
        return Passage(
            source_id=source_id,
            source_type="normatif",
            # §6ter/INV-010 : l'opposabilité dérive du type de document,
            # jamais d'un flag de la requête. Un article parlementaire
            # pastillé (amendement, texte non promulgué) n'est jamais
            # opposable, quel que soit l'appelant -- un override piloté par
            # la query rendrait INV-010 contournable par la formulation de
            # requête (§4 étape 3) le jour où elle est déléguée à un LLM.
            opposable=False,
            text=text,
            metadata=metadata,
        )

    def _retrieve_code_article(self, intent: Intent, query: dict[str, str]) -> Passage:
        article = query.get("article")
        code = query.get("code")
        if not article or not code:
            raise RetrievalError("code_article route requires 'article' and 'code'.")

        date = query.get("date")
        article_normalise = _normalize_article_number(article)
        code_titre_like = f"%{code.lower()}%"

        sql, params = _build_code_article_query(article_normalise, code_titre_like, date)
        result = self.client.call_tool(
            "query_sql",
            {"schema": "legifrance", "query": sql, "params": params},
        )
        rows = _parse_sql_rows(result)
        if not rows:
            raise RetrievalError(
                f"No candidate found for article '{article}' in code matching '{code}'."
            )

        row = rows[0]
        # Piège A3 (variante) : le LIKE sur le titre peut matcher plusieurs
        # textes distincts (ex. '%civil%' -> Code civil ET Code de procédure
        # civile) ; prendre rows[0] serait alors une sélection silencieuse du
        # mauvais code, verbatim exact compris. Signalé pour que
        # l'orchestrateur élève le risque (§2). Plusieurs versions du MÊME
        # texte ne sont pas ambiguës : l'ORDER BY date_debut choisit la plus
        # récente applicable.
        distinct_titres = {r.get("titre_texte") for r in rows}
        selection_ambiguous = len(distinct_titres) > 1
        etat = row.get("etat")
        opposable = etat in _OPPOSABLE_ETATS
        contenu = row.get("contenu")
        if not contenu:
            raise RetrievalError(f"Article '{row.get('id')}' has no extractable content.")

        # §4bis, piège A3 : un numéro existant mais faux, deviné plutôt que
        # copié depuis la question, ne doit jamais être exposé avec la même
        # confiance qu'une référence explicite de l'utilisateur.
        article_provenance = check_slot_provenance(intent.question, "article", article)
        code_provenance = check_slot_provenance(intent.question, "code", code)
        slot_inferred = article_provenance.inferred or code_provenance.inferred

        return Passage(
            source_id=str(row.get("id")),
            source_type="normatif",
            opposable=opposable,
            text=contenu,
            metadata={
                "num": row.get("num"),
                "date_debut": row.get("date_debut"),
                "date_fin": row.get("date_fin"),
                "etat": etat,
                "titre_texte": row.get("titre_texte"),
                "candidate_count": len(rows),
                "selection_ambiguous": selection_ambiguous,
                "slot_inferred": slot_inferred,
                "article_slot_copied": article_provenance.copied,
                "code_slot_copied": code_provenance.copied,
                # §4ter : renvois bruts pour la résolution multi-saut (multi_hop.py),
                # jamais suivis automatiquement ici -- l'orchestrateur appelant décide.
                "liens": _normalize_liens(row.get("liens")),
            },
        )

    def _retrieve_texte_libre(self, query: dict[str, str]) -> Passage:
        search_query = query.get("query")
        if not search_query:
            raise RetrievalError("texte_libre route requires 'query'.")

        result = self.client.call_tool(
            "search_legal_texts", {"query": search_query, "limit": int(query.get("limit", "3"))}
        )
        hits = _parse_search_hits(result)
        if not hits:
            raise RetrievalError(f"No candidate found for free-text query '{search_query}'.")

        # Reranking déterministe : on ne prend plus hits[0] à l'aveugle. Selon
        # `sort`, on classe par pertinence lexicale (défaut) ou par date.
        sort = query.get("sort", "pertinence")
        ordered, selection_ambiguous = _rerank_hits(hits, search_query, sort)
        best = ordered[0]
        document = best.get("document", {})
        title = document.get("autocompletion")
        if not title:
            raise RetrievalError("Top search_legal_texts hit has no extractable title.")

        # §4bis : recherche par proximité sémantique, jamais par identité
        # exacte -- toujours marquée "pertinence non garantie", quel que
        # soit le score de pertinence du moteur de recherche. La fidélité
        # (le titre existe) et la pertinence (c'est la bonne source) sont
        # deux propriétés distinctes ; cette route ne garantit jamais la seconde.
        return Passage(
            source_id=str(document.get("uid") or document.get("id") or search_query),
            source_type="normatif",
            opposable=False,  # statut d'autorité non établi par cette route
            text=title,
            metadata={
                "pertinence_non_garantie": True,
                "page_path": document.get("page_path"),
                "badge": document.get("badge"),
                "candidate_count": len(hits),
                "sort": sort,
                # Après reranking : ambigu seulement si le meilleur candidat ne
                # se détache pas nettement du second (quasi-égalité). Un gagnant
                # net n'est plus signalé comme sélection silencieuse.
                "selection_ambiguous": selection_ambiguous,
            },
        )


def _extract_pastille_text(result: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(result, dict):
        text = result.get("text") or result.get("contenu") or result.get("content")
        if isinstance(text, str) and text:
            return text, result
    if isinstance(result, str):
        return result, {}
    raise RetrievalError("Unable to extract article text from get_pastilled_article response.")


def _parse_sql_rows(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        return result
    if isinstance(result, dict) and "error" in result:
        raise RetrievalError(f"SQL query failed: {result['error']}")
    raise RetrievalError("Unexpected response shape from query_sql.")


def _parse_search_hits(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict) and "error" in result:
        raise RetrievalError(f"search_legal_texts failed: {result['error']}")
    if isinstance(result, dict) and isinstance(result.get("hits"), list):
        return result["hits"]
    raise RetrievalError("Unexpected response shape from search_legal_texts.")


def _extract_parlement_question(result: Any) -> dict[str, Any]:
    """get_parlement_item enveloppe l'objet dans {"data": {"data": {...}}}."""
    if isinstance(result, dict):
        data = result.get("data", {})
        if isinstance(data, dict):
            inner = data.get("data")
            if isinstance(inner, dict):
                return inner
    raise RetrievalError("Unexpected response shape from get_parlement_item.")


def _normalize_liens(raw: Any) -> list[dict[str, Any]]:
    """legifrance.article.data->'LIENS'->'LIEN' est un objet unique quand il
    n'y a qu'un seul lien, une liste sinon (artefact JSONB) -- normalise
    toujours vers une liste pour que les appelants n'aient pas à le savoir.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [raw]
    return []
