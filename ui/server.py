"""Interface graphique simple (stdlib http.server, zéro dépendance web) pour
visualiser le pipeline Hallucide en direct : question -> décomposition
(Mistral) -> récupération réelle (Moulineuse/data.gouv) -> vérification
déterministe -> statut coloré + journal de conformité.

Lancement :  python ui/server.py     puis ouvrir http://localhost:8765
Nécessite une clé LLM dans .env (par défaut ANTHROPIC_API_KEY pour Claude ;
MISTRAL_API_KEY / GEMINI_API_KEY aussi supportées).
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]


# Charger .env (MISTRAL_API_KEY).
env_path = WORKSPACE / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import re  # noqa: E402

import hashlib  # noqa: E402
import uuid  # noqa: E402

from hallucide import ClaudeModelProvider, GeminiModelProvider, GemmaModelProvider, MistralModelProvider, MultiSourceRetrievalProvider, Hallucide  # noqa: E402
from hallucide.llm_providers.gemma import DEFAULT_GEMMA_BASE_URL, DEFAULT_GEMMA_MODEL  # noqa: E402
from hallucide.core_types.exceptions import RetrievalError, HallucideError, VerificationError  # noqa: E402
from hallucide.retrieval.mcp_client import McpToolClient  # noqa: E402
from hallucide.core_types.types import Claim, ClaimStatus, Intent, Passage, RetrievalState  # noqa: E402
from hallucide.verification.verifier import verify_claims  # noqa: E402

HTML_PAGE = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

# --- Routage automatique déterministe (§4bis : transparent, jamais opaque) ---

_QUESTION_NUM_RE = re.compile(r"\b(?:qosd|qe|qg|question(?:\s+orale)?)\D{0,20}?(\d{2,6})\b", re.IGNORECASE)
# « amendement n° 245 », « amendement 245 », « amdt 245 »
_AMENDEMENT_RE = re.compile(r"\b(?:amendements?|amdt)\s*(?:n[°o]\.?\s*)?(\d{1,5})\b", re.IGNORECASE)
# Compte rendu / intervention en séance : « qu'a dit X », « position de X sur … »
_INTERVENTION_HINTS = ("compte rendu", "compte-rendu", "en séance", "en seance", "qu'a dit",
                       "qu a dit", "position de", "position du", "intervention de",
                       "s'est exprimé", "a déclaré", "a defendu", "a défendu", "propos de")
# Nom d'orateur après un déclencheur (M./Mme/de/du/par) : « de Darmanin »,
# « M. Gérald Darmanin ». On garde 1 à 3 mots capitalisés.
_ORATEUR_RE = re.compile(r"(?:M\.|Mme|monsieur|madame|de|du|par)\s+"
                         r"([A-ZÀ-Ÿ][\wÀ-ÿ'’-]+(?:\s+[A-ZÀ-Ÿ][\wÀ-ÿ'’-]+){0,2})")
# Nom complet d'un député (au moins prénom + nom capitalisés) : « Gabriel Attal ».
# Classes latines PROPRES : majuscules A-Z + À-Þ (sauf ×), minuscules a-z + à-ÿ
# (sauf ÷). Sans ça, [À-Ÿ] engloberait les minuscules accentuées (é, è…) et
# capturerait « François Hollande était-il ».
_MAJ = "A-ZÀ-ÖØ-Þ"
_MIN = "a-zà-öø-ÿ"
_MOT_NOM = rf"[{_MAJ}][{_MIN}'’]+(?:-[{_MAJ}][{_MIN}'’]+)*"  # gère Braun-Pivet, Saint-Martin
_NOM_DEPUTE_RE = re.compile(rf"\b({_MOT_NOM}(?:\s+{_MOT_NOM})+)")
# Commission NOMMÉE (question ciblée) : « la commission des affaires sociales ».
# Le « commission » SINGULIER + article distingue du listing (« commissionS où »).
_COMMISSION_CIBLE_RE = re.compile(
    r"\bcommission\s+(?:des?|du|de\s+l['’]|d['’]|de)\s+([^?.,;!]+)", re.IGNORECASE)
_COMMISSION_CIBLE_STOP = re.compile(
    r"\s+(?:depuis|avant|entre|pendant|durant|pour|en|au|à)\b.*$", re.IGNORECASE)


def _extract_commission_cible(q: str) -> str:
    """Descripteur de la commission nommée (« affaires sociales »), ou '' si la
    question est un simple listing (« les commissions où … »)."""
    m = _COMMISSION_CIBLE_RE.search(q)
    if not m:
        return ""
    return _COMMISSION_CIBLE_STOP.sub("", m.group(1).strip()).strip()
# « L. 1232-6 » : préfixe L/R/D avec point et espace optionnels devant le
# numéro -- sans cette alternative, la capture s'arrêtait à « L. ».
# Accepte « article », « articles », « art. » et « art » (abréviations
# courantes) -- sans quoi « art. 1128 du code civil » repartait en recherche
# libre, alors que la référence est parfaitement structurée.
_ARTICLE_RE = re.compile(r"\b(?:articles?|art\.?)\s+((?:[LRD]\.?\s*)?\d[\w.\-]*|[\wÀ-ÿ.\-]+)", re.IGNORECASE)
# La préposition fait partie du titre officiel (« Code DE LA construction et
# de l'habitation ») : la retirer cassait le LIKE sur le titre -- on capture
# tout ce qui suit « code ».
_CODE_RE = re.compile(r"\bcode\s+([\wÀ-ÿ'\s]+?)(?:\s*[?.,;]|$)", re.IGNORECASE)
# Le nom d'un code s'arrête au premier marqueur de nouvelle proposition
# (interrogatif, verbe de question) : « code civil et quelle est la règle… »
# doit donner « civil », pas toute la fin de la phrase. Attention : « et »
# seul ne suffit pas comme frontière, des titres réels en contiennent
# (« code de la construction et de l'habitation ») -- on ne coupe sur « et »
# que s'il introduit un interrogatif ou un verbe de question.
#   ... et aux prépositions qui introduisent un complément de question après le
#   nom du code (« code civil SUR la vie privée », « code civil EN cas de… »).
#   Ces mots n'apparaissent jamais en tête de complément dans un titre de code
#   réel (les titres enchaînent sur « de / du / de la / des / et / général »),
#   donc les couper ne mutile pas « code de la construction et de l'habitation ».
_CODE_BREAK_RE = re.compile(
    r"\s+(?:et\s+)?(?:quelle?s?|que|qui|quoi|comment|combien|pourquoi|dont|est-ce|peut|doit|dit"
    r"|en|sur|dans|pour|avec|afin|concernant|lorsqu\w*|quand)\b.*$",
    re.IGNORECASE,
)
_DATA_HINTS = ("combien", "nombre", "taux", "médiane", "revenu", "population",
               "naissances", "décès", "chômage", "insee", "moyenne", "montant", "effectif")
_PARLEMENT_HINTS = ("qosd", "question orale", "question écrite", "question parlementaire",
                    "question au gouvernement", "députe", "député", "sénateur", "amendement",
                    "assemblée nationale", "au ministre", "interroge le ministre")


def detect_route(question: str) -> dict:
    """Détecte la source la plus probable depuis la question (déterministe).
    Retourne {"route": ..., "reason": ..., "prefill": {...}}. L'UI l'affiche
    et laisse l'utilisateur corriger -- jamais un routage silencieux (§4bis).

    Priorité : référence structurée (article de code) > indice parlementaire >
    question chiffrée > repli recherche libre."""
    # Les guillemets anglais/français englobants (copiés-collés depuis un texte
    # cité) ne font pas partie de la question -- les retirer avant toute
    # détection ou recherche, sinon search_legal_texts échoue en cherchant le
    # caractère guillemet lui-même.
    q = question.strip().strip("\"“”«»").strip()
    low = q.lower()

    # 1. Article de code : le plus spécifique, prioritaire même si "question" apparaît.
    art = _ARTICLE_RE.search(q)
    code = _CODE_RE.search(q)
    if art and code:
        code_name = _CODE_BREAK_RE.sub("", code.group(1)).strip()
        return {"route": "code_article", "reason": "Référence 'article … du code …' détectée",
                "prefill": {"article": art.group(1).rstrip(".,;"), "code": "code " + code_name}}

    # 1bis-a. Mandat de député : « X est-il député ? », « X est-elle sénatrice ? »,
    #    « est-ce que X est député ». Donnée structurée (annuaire acteurs), sans LLM.
    m_mandat = re.search(
        r"^(?:est[- ]ce que\s+)?(.+?)\s+(?:est[- ](?:il|elle)|a[- ]?t[- ]?(?:il|elle)\s+été|était[- ](?:il|elle)|fut[- ](?:il|elle))\s+(?:un[e]?\s+)?(député|députée|sénateur|sénatrice|ministre|membre du gouvernement|secrétaire d.état)",
        q, re.IGNORECASE)
    if not m_mandat:
        m_mandat = re.search(r"(?:député|députée)\s*\?", q, re.IGNORECASE) and _NOM_DEPUTE_RE.search(q) and None or None
    if m_mandat:
        acteur = m_mandat.group(1).strip().rstrip(",")
        # garde le nom propre final si la tournure inclut du contexte
        nom = _NOM_DEPUTE_RE.search(acteur)
        if nom:
            acteur = nom.group(1)
        fonction_mot = m_mandat.group(2).lower()
        fonction = "depute" if fonction_mot.startswith(("député", "sénat")) else "ministre"
        return {"route": "mandat", "reason": "Question sur le mandat d'un acteur (open data)",
                "prefill": {"acteur": acteur, "fonction": fonction}}

    # 1bis. Commissions d'un député (+ dates) : « les commissions où X a siégé /
    #    appartenu », « liste les commissions de X ». Donnée structurée SQL, sûre.
    if "commission" in low:
        nom = _NOM_DEPUTE_RE.search(q)
        if nom:
            prefill = {"acteur": nom.group(1)}
            cible = _extract_commission_cible(q)
            if cible:
                prefill["commission"] = cible
            return {"route": "commissions", "reason": "Appartenances aux commissions d'un député (open data)",
                    "prefill": prefill}

    # 2. Intervention en séance / compte rendu : « qu'a dit X », « position de X
    #    sur … » -- AVANT l'amendement, car ces questions veulent le VERBATIM
    #    de l'orateur, pas le texte de l'amendement (autre serveur MCP).
    if any(h in low for h in _INTERVENTION_HINTS):
        orateur = _ORATEUR_RE.search(q)
        return {"route": "intervention", "reason": "Compte rendu / intervention en séance détecté",
                "prefill": {"search": q, "orateur": orateur.group(1) if orateur else ""}}

    # 3. Amendement (« amendement n° 245 », « amdt 245 ») : AVANT la route
    #    parlementaire, sinon le mot « amendement » y est capté par les indices.
    amdt = _AMENDEMENT_RE.search(q)
    if amdt:
        return {"route": "amendement", "reason": "Numéro d'amendement détecté",
                "prefill": {"numero": amdt.group(1)}}

    # 3. Question parlementaire : numéro explicite ou indice lexical parlementaire.
    num = _QUESTION_NUM_RE.search(q)
    if num or any(h in low for h in _PARLEMENT_HINTS):
        return {"route": "parlement_question", "reason": "Question parlementaire détectée",
                "prefill": {"numero": num.group(1) if num else ""}}

    # 3. Question chiffrée (statistique).
    if any(h in low for h in _DATA_HINTS):
        return {"route": "donnee", "reason": "Question chiffrée détectée (donnée statistique)",
                "prefill": {}}

    # 4. Repli : recherche libre, pertinence non garantie.
    return {"route": "texte_libre", "reason": "Aucune référence structurée — recherche libre (pertinence non garantie)",
            "prefill": {"query": q}}


def resolve_parlement_uid(question: str, numero: str) -> dict:
    """Résout l'UID d'une question parlementaire : par numéro exact si présent,
    sinon par recherche mot-clé. Retourne des candidats (uid + titre) que
    l'utilisateur choisit -- jamais un choix silencieux (§4bis / piège C1)."""
    client = McpToolClient("https://mcp.code4code.eu/mcp")

    def _items(res):
        return res.get("data", {}).get("data", []) if isinstance(res, dict) else []

    candidates = []
    seen_uids = set()

    def _add(it):
        uid = it.get("uid")
        if uid and uid not in seen_uids:
            seen_uids.add(uid)
            candidates.append({"uid": uid, "numero": str(it.get("numero", "")),
                               "type": it.get("type"), "titre": it.get("titre") or it.get("rubrique")})

    # (a) Si un numéro est fourni, on le cherche par type (QOSD/QE) et on filtre
    # côté client -- le filtre 'numero' de l'API n'est pas fiable (observé en direct).
    if numero:
        n = numero.lstrip("0")
        for typ in ("QOSD", "QE", "QG"):
            try:
                res = client.call_tool("list_parlement_items",
                                       {"resource": "questions", "query": {"type": typ, "perPage": 100}})
            except RetrievalError:
                continue
            for it in _items(res):
                if str(it.get("numero", "")).lstrip("0") == n:
                    _add(it)

    # (b) Recherche par mots-clés de la question (complète les candidats).
    search_terms = re.sub(r"\b(qosd|qe|qg|question|orale|écrite|n[°o]|numéro)\b", " ", question, flags=re.IGNORECASE).strip()
    if search_terms:
        try:
            res = client.call_tool("list_parlement_items",
                                   {"resource": "questions", "query": {"search": search_terms, "perPage": 8}})
            for it in _items(res):
                _add(it)
        except RetrievalError as exc:
            if not candidates:
                return {"error": str(exc)}

    return {"candidates": candidates[:8]}


# Abréviations dont le point ne termine PAS une phrase ("M. le ministre",
# "art. 1103", "n° 812") -- couper dessus produirait un claim de contrôle
# tronqué en plein mot ("Mme Danièle Obono interroge M", observé en direct).
_NON_SENTENCE_ABBREVS = {"m", "mme", "mlle", "mm", "dr", "me", "art", "al", "cf", "n°", "no", "p"}
_SENTENCE_BREAK_RE = re.compile(r"\. |\n| ; ")
_LAST_WORD_RE = re.compile(r"[\wÀ-ÿ°]+$")


def _control_claim_from_passage(passage_text: str) -> str:
    """Claim de contrôle déterministe : la première phrase du passage officiel,
    reprise MOT POUR MOT. Sert à démontrer que le vérificateur valide bien un
    verbatim réel (AUTHENTIFIÉ / CITÉ_NON_OPPOSABLE) même quand le LLM, par
    excès de prudence, n'a produit aucune affirmation (sur-refus, §12).
    Ce n'est PAS le LLM : c'est une extraction déterministe côté démo."""
    text = passage_text.strip()
    for match in _SENTENCE_BREAK_RE.finditer(text):
        if match.group().startswith("."):
            prev_word = _LAST_WORD_RE.search(text[: match.start()])
            if prev_word and prev_word.group().lower() in _NON_SENTENCE_ABBREVS:
                continue  # "M. le ministre" : pas une fin de phrase
        head = text[: match.start()].strip()
        if 15 <= len(head) <= 300:
            return head
    return text[:200].strip()


def _build_query(route: str, form: dict) -> dict:
    """Construit la query structurée selon la route choisie dans l'UI."""
    if route == "code_article":
        return {"route": "code_article", "article": form.get("article", ""), "code": form.get("code", "")}
    if route == "parlement_question":
        return {"route": "parlement_question", "uid": form.get("uid", "")}
    if route == "amendement":
        q = {"route": "amendement", "numero": form.get("numero", "")}
        if form.get("legislature"):
            q["legislature"] = form["legislature"]
        return q
    if route == "intervention":
        q = {"route": "intervention", "search": form.get("search", "").strip()}
        if form.get("orateur"):
            q["orateur"] = form["orateur"]
        return q
    if route == "mandat":
        return {"route": "mandat", "acteur": form.get("acteur", "").strip(),
                "fonction": form.get("fonction", "depute").strip()}
    if route == "commissions":
        q = {"route": "commissions", "acteur": form.get("acteur", "").strip()}
        if form.get("commission"):
            q["commission"] = form["commission"].strip()
        return q
    if route == "texte_libre":
        raw_query = form.get("query", "").strip().strip("\"“”«»").strip()
        sort = form.get("sort", "pertinence")
        return {"route": "texte_libre", "query": raw_query, "sort": sort}
    if route == "donnee":
        return {
            "dataset_id": form.get("dataset_id", ""),
            "resource_id": form.get("resource_id", ""),
            "filter_column": form.get("filter_column", ""),
            "filter_value": form.get("filter_value", ""),
            "target_column": form.get("target_column", ""),
        }
    if route == "fichier":
        # 'filters' est un JSON {colonne: valeur, ...} saisi dans l'UI.
        raw = form.get("filters", "{}").strip() or "{}"
        try:
            filters = json.loads(raw)
        except json.JSONDecodeError:
            filters = {}
        return {
            "dataset_id": form.get("dataset_id", ""),
            "resource_id": form.get("resource_id", ""),
            "filters": filters,
            "target_column": form.get("target_column", ""),
        }
    raise ValueError(f"Route inconnue : {route}")


# Providers LLM disponibles : (variable d'environnement de la clé, constructeur).
# Défaut = Gemma 4 (hackathon : modèle unique et central, §8 règlement).
# Ajouter un modèle = une ligne ici.
_MODEL_PROVIDERS = {
    "claude": ("ANTHROPIC_API_KEY", ClaudeModelProvider),
    "mistral": ("MISTRAL_API_KEY", MistralModelProvider),
    "gemini": ("GEMINI_API_KEY", GeminiModelProvider),
}
DEFAULT_MODEL = "gemma"


def _build_model_provider(model: str):
    """Instancie le provider LLM choisi. Renvoie (provider, None) ou
    (None, message d'erreur « ...API_KEY absente du .env ») si la clé manque —
    ce message déclenche le verrou « moteur non connecté » côté UI."""
    name = (model or DEFAULT_MODEL).lower()

    # Gemma n'a pas de clé API (backend local vLLM/Ollama) : URL et nom de
    # modèle configurables via .env pour basculer entre backends sans changer
    # le code (§7 règle NVIDIA : jamais d'URL en dur).
    if name == "gemma":
        base_url = os.environ.get("MODEL_BASE_URL", DEFAULT_GEMMA_BASE_URL)
        model_name = os.environ.get("MODEL_NAME", DEFAULT_GEMMA_MODEL)
        return GemmaModelProvider(base_url=base_url, model=model_name), None

    entry = _MODEL_PROVIDERS.get(name)
    if entry is None:
        return None, f"Modèle inconnu : {model}"
    env_var, provider_cls = entry
    key = os.environ.get(env_var)
    if not key:
        return None, f"{env_var} absente du .env"
    return provider_cls(api_key=key), None


def _run_commissions_pipeline(message: str, query: dict) -> dict:
    """Chemin DÉTERMINISTE pour la route commissions (appartenances d'un député).

    La donnée vient d'une requête SQL sur l'open data officiel : AUCUN modèle
    n'est dans la boucle, donc rien à halluciner. Chaque ligne « Commission … —
    du … au … » est un fait `DONNÉE_TRACÉE` vérifié (égalité exacte à la source).
    Résultat : une seule intention, risque faible, PUBLIABLE -- pas de plancher
    E1 (pas de décomposition LLM) ni de statut « faible » (donnée tracée, pas
    citation approximative). La garantie tient : une ligne inventée serait
    rejetée par le vérificateur. Ne nécessite pas de clé LLM."""
    provider = MultiSourceRetrievalProvider()
    intent = Intent(id="1", question=message)
    try:
        passage = provider.retrieve(intent, RetrievalState(), query)
    except HallucideError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    claims_out = []
    all_pass = True
    for ligne in [l for l in passage.text.split("\n") if l.strip()]:
        ligne_passage = Passage(source_id=passage.source_id, source_type=passage.source_type,
                                opposable=passage.opposable, text=ligne, metadata={})
        try:
            cres = verify_claims([Claim(ref=ligne, status=ClaimStatus.DONNÉE_TRACÉE)], ligne_passage)
            statut = cres.claims[0].status.value
        except VerificationError:
            statut = ClaimStatus.NON_AUTHENTIFIÉ.value
            all_pass = False
        claims_out.append({"ref": ligne, "status": statut, "truncation_flagged": False})

    phash = hashlib.sha256(passage.text.encode("utf-8")).hexdigest()[:16]
    compliance = "VALIDATED" if all_pass else "BLOCKED"
    intent_out = {
        "question": message,
        "source_id": passage.source_id,
        "source_type": passage.source_type,
        "opposable": passage.opposable,
        "titre": passage.metadata.get("acteur"),
        "passage_text": passage.text,
        "pertinence_non_garantie": False,
        "claims": claims_out,
        "control_claim": None,
        "verbatim_check": "PASS" if all_pass else "FAIL",
        "risk_tier": "faible",           # donnée déterministe -> pas de plancher
        "published": all_pass,           # publiable : fait tracé, pas citation faible
        "human_validation": None,
        "validation_key": {"intent_id": "1", "passage_hash": phash},
        "compliance_status": compliance,
        "compliance_json": {
            "compliance_status": compliance,
            "source_id": passage.source_id,
            "source_type": passage.source_type,
            "passage_hash": phash,
            "verbatim_check": "PASS" if all_pass else "FAIL",
            "acteur": passage.metadata.get("acteur"),
            "acteur_ref": passage.metadata.get("acteur_ref"),
            "nb_commissions": passage.metadata.get("nb_commissions"),
            "source": passage.metadata.get("source"),
        },
    }
    return {"echo_back": None, "coverage_ratio": 1.0, "coverage_passed": True,
            "session_ref": uuid.uuid4().hex, "intents": [intent_out]}


def _run_pipeline(message: str, route: str, form: dict, model: str = DEFAULT_MODEL) -> dict:
    """Exécute le pipeline réel et renvoie un dict JSON-sérialisable pour l'UI.

    `model` choisit le provider LLM (défaut : Claude) ; les autres (Mistral,
    Gemini) restent disponibles pour le futur sélecteur de modèle."""
    # Route commissions : chemin déterministe (SQL), sans LLM -> pas besoin de clé.
    if route in ("commissions", "mandat"):
        return _run_commissions_pipeline(message, _build_query(route, form))

    provider, err = _build_model_provider(model)
    if err:
        return {"error": err}

    guard = Hallucide(model_provider=provider)
    query = _build_query(route, form)

    try:
        result = guard.ask(message=message, query=query)
    except HallucideError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    intents_out = []
    for idx, r in enumerate(result.orchestration.results):
        entry = result.compliance_entries[idx]
        llm_claims = [{"ref": c.ref, "status": c.status.value, "truncation_flagged": c.truncation_flagged}
                      for c in r.verification.claims]

        # Fallback de démonstration : si le LLM n'a produit aucune affirmation
        # (sur-refus fréquent avec un petit modèle), on soumet au vérificateur
        # un claim de contrôle = verbatim réel du passage, pour montrer que la
        # garantie déterministe (§7) valide bien le texte officiel.
        #
        # Jamais quand pertinence_non_garantie=True : le claim de contrôle ne
        # démontre que "ce texte existe dans SA source", pas "cette source
        # répond à la question" -- l'afficher sur une source hors-sujet (piège
        # C1) crée une illusion de preuve (voir ANALYSE_TEST_JURISPRUDENCE.md).
        control = None
        pertinence_non_garantie = bool(r.passage.metadata.get("pertinence_non_garantie"))
        if not r.verification.claims and not pertinence_non_garantie:
            # Une source "donnee" (data.gouv) se vérifie par égalité de cellule
            # (DONNÉE_TRACÉE), pas par contiguïté de texte (AUTHENTIFIÉ).
            if r.passage.source_type == "donnee":
                ref = r.passage.text
                control_status = ClaimStatus.DONNÉE_TRACÉE
            else:
                ref = _control_claim_from_passage(r.passage.text)
                control_status = ClaimStatus.AUTHENTIFIÉ
            try:
                cres = verify_claims([Claim(ref=ref, status=control_status)], r.passage)
                c = cres.claims[0]
                control = {"ref": c.ref, "status": c.status.value, "verbatim_check": cres.verbatim_check}
            except VerificationError as exc:
                c = exc.result.claims[0]
                control = {"ref": c.ref, "status": c.status.value, "verbatim_check": "FAIL"}

        intents_out.append({
            "question": r.intent.question,
            "source_id": r.passage.source_id,
            "source_type": r.passage.source_type,
            "opposable": r.passage.opposable,
            "titre": r.passage.metadata.get("titre") or r.passage.metadata.get("titre_texte"),
            "passage_text": r.passage.text,
            "pertinence_non_garantie": bool(r.passage.metadata.get("pertinence_non_garantie")),
            "claims": llm_claims,
            "control_claim": control,
            "verbatim_check": r.verification.verbatim_check,
            "risk_tier": r.risk_tier.value,
            "published": result.published[idx],
            "human_validation": entry.human_validation,
            # Clé de validation (§4 étape 9) : la décision humaine se prend
            # HORS de cette UI (circuit de validation de l'institution, via le
            # HumanValidationRegistry du cœur) -- l'UI se contente d'afficher
            # la clé qui identifie CETTE intention sur CE contenu vérifié.
            "validation_key": {
                "intent_id": r.intent.id,
                "passage_hash": entry.passage_hashes[0],
            },
            "compliance_status": entry.compliance_status,
            "compliance_json": entry.to_dict(),
        })

    return {
        "echo_back": result.orchestration.echo_back,
        "coverage_ratio": result.orchestration.coverage_ratio,
        "coverage_passed": result.orchestration.coverage_passed,
        "session_ref": result.session_ref,
        "intents": intents_out,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence le log par défaut

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def _send_json(self, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path not in ("/ask", "/resolve"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))

        try:
            if self.path == "/resolve":
                # Étape 1 : détecter la source + proposer les candidats UID.
                message = payload.get("message", "")
                detection = detect_route(message)
                result = detection
                if detection["route"] == "parlement_question":
                    resolved = resolve_parlement_uid(message, detection["prefill"].get("numero", ""))
                    result = {**detection, **resolved}
            else:
                result = _run_pipeline(payload.get("message", ""), payload.get("route", ""), payload.get("form", {}))
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}

        self._send_json(result)


def main():
    port = 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Hallucide UI -> http://localhost:{port}  (Ctrl+C pour arrêter)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt.")
        server.shutdown()


if __name__ == "__main__":
    main()
