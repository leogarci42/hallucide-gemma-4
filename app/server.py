"""Backend de l'interface de chat futuriste (app/).

Rôle : exposer le moteur Hallucide à un front web moderne, SANS toucher au
moteur ni au démonstrateur historique (ui/server.py reste le fallback intact).

Choix d'architecture (voir app/README.md) :
  - stdlib `http.server` uniquement → ZÉRO dépendance à installer, l'équipe lance
    `python -m app.server` sans rien de plus que le moteur déjà installé.
  - On RÉUTILISE la logique éprouvée de ui/server.py (`_run_pipeline`, `detect_route`,
    `resolve_parlement_uid`) : c'est la seule source de vérité pour appeler le moteur
    et sérialiser un AskResult. On ne la réécrit pas (elle contient des subtilités
    correctes : claim de contrôle de secours, garde `pertinence_non_garantie`).
  - Par-dessus ce JSON réel, on ajoute UNIQUEMENT la couche de présentation
    (app/presentation.py) : score 0-100 + bande de couleur par claim et par intention.

Endpoints :
  GET  /                → sert le front (app/static/index.html)
  GET  /static/<f>      → sert css/js
  POST /resolve         → détection de route + candidats UID (identique à ui/)
  POST /ask             → pipeline réel + enrichissement score/couleur

Lancement :  python -m app.server    puis http://localhost:8770
Modèle par défaut : Gemma 4 (backend compatible OpenAI, vLLM/Ollama) via
MODEL_BASE_URL/MODEL_NAME dans .env. Les autres providers (Claude/Mistral/
Gemini) restent utilisables via leur clé API respective.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Racine du dépôt = le premier parent contenant pyproject.toml. Robuste quelle
# que soit la profondeur d'imbrication (ce fichier vit sous app/).
WORKSPACE = Path(__file__).resolve()
for _parent in WORKSPACE.parents:
    if (_parent / "pyproject.toml").exists():
        WORKSPACE = _parent
        break
# Le moteur vit dans src/ ; ui/ est un package de voisinage qu'on réutilise.
for p in (str(WORKSPACE),):
    if p not in sys.path:
        sys.path.insert(0, p)

# Charger .env (MISTRAL_API_KEY / GEMINI_API_KEY) — même logique que ui/server.py.
_env_path = WORKSPACE / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# Réutilisation directe du démonstrateur historique (source de vérité moteur).
from ui.server import _run_pipeline, detect_route, resolve_parlement_uid, _build_model_provider  # noqa: E402

from app import presentation  # noqa: E402

STATIC_DIR = Path(__file__).parent / "static"

# Types MIME servis par le front (stdlib, pas de dépendance).
_MIME = {".html": "text/html", ".css": "text/css", ".js": "application/javascript",
         ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon", ".json": "application/json"}


# --- Enrichissement : ajoute le score/couleur SANS toucher aux champs moteur ---

def _enrich(result: dict) -> dict:
    """Reçoit le JSON réel du moteur (via ui.server._run_pipeline) et lui greffe
    la couche de présentation. Ne SUPPRIME ni ne MODIFIE aucun champ moteur — on
    ne fait qu'ajouter des clés `score`. Ainsi le front peut toujours afficher la
    donnée brute (traçabilité) à côté de l'habillage."""
    if not isinstance(result, dict) or result.get("error"):
        return result  # erreur moteur : on la laisse passer telle quelle (voir do_POST).

    for intent in result.get("intents", []):
        published = bool(intent.get("published"))
        risk = intent.get("risk_tier", "")
        # Score par claim (badge individuel).
        for claim in intent.get("claims", []):
            claim["score"] = presentation.score_for_claim(
                claim.get("status", ""), risk, published, claim.get("ref", "")
            ).to_dict()
        # Score du claim de contrôle de secours, s'il existe.
        cc = intent.get("control_claim")
        if cc:
            cc["score"] = presentation.score_for_claim(cc.get("status", ""), risk, published, cc.get("ref", "")).to_dict()
        # Score AGRÉGÉ de l'intention (vignette de tête, gère NO_ANSWER).
        intent["score"] = presentation.score_for_intent(
            claims=intent.get("claims", []),
            compliance_status=intent.get("compliance_status", ""),
            risk_tier=risk,
            published=published,
            control_claim=cc,
        ).to_dict()

    result["engine_connected"] = True
    return result


def _format_answer(result: dict, message: str, model: str) -> None:
    """Rédige une réponse en prose via le LLM, UNIQUEMENT à partir des claims
    vérifiés déjà présents dans le résultat. Le LLM ne vérifie rien et ne peut
    rien ajouter : il met en forme des lignes déjà contrôlées par le moteur.
    Champ ajouté : result["answer_text"]. En cas d'échec, champ absent, le
    front garde l'affichage ligne à ligne (jamais de blocage du résultat)."""
    lines = []
    for intent in result.get("intents", []):
        for claim in intent.get("claims", []):
            statut = claim.get("status", "")
            lines.append(f"- [{statut}] {claim.get('ref', '')}")
    if not lines:
        return
    provider, err = _build_model_provider(model)
    if err:
        return
    try:
        out = provider.generate([
            {"role": "system", "content":
                "Tu mets en forme des données déjà vérifiées. Rédige en français une réponse "
                "claire et professionnelle à la question, en utilisant UNIQUEMENT les lignes "
                "fournies. N'ajoute aucun fait, aucune date, aucun nom qui n'y figure pas. "
                "Ordre chronologique décroissant (le plus récent d'abord). Pas de tiret cadratin."},
            {"role": "user", "content": f"Question : {message}\n\nDonnées vérifiées :\n" + "\n".join(lines[:120])},
        ])
        text = (out or {}).get("text", "").strip()
        if text:
            result["answer_text"] = text
    except Exception:
        pass  # mise en forme facultative : jamais bloquante


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence le log par défaut

    # --- GET : front statique ---
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html")
        elif path.startswith("/static/"):
            target = (STATIC_DIR / path[len("/static/"):]).resolve()
            # Anti-traversée de répertoire : on reste sous STATIC_DIR.
            if STATIC_DIR.resolve() in target.parents and target.exists():
                self._send_file(target)
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def _send_file(self, target: Path):
        if not target.exists():
            self.send_error(404)
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _MIME.get(target.suffix, "application/octet-stream") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # --- POST : /ask et /resolve ---
    def do_POST(self):
        if self.path not in ("/ask", "/resolve"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Corps JSON invalide"}, status=400)
            return

        try:
            if self.path == "/resolve":
                message = payload.get("message", "")
                detection = detect_route(message)
                result = detection
                if detection["route"] == "parlement_question":
                    resolved = resolve_parlement_uid(message, detection["prefill"].get("numero", ""))
                    result = {**detection, **resolved}
            else:
                raw = _run_pipeline(payload.get("message", ""), payload.get("route", ""),
                                    payload.get("form", {}), payload.get("model", "gemma"))
                # Cas « moteur non connecté » : clé API absente → on le dit
                # EXPLICITEMENT au front (jamais de faux résultat déguisé en vrai).
                if isinstance(raw, dict) and "_API_KEY absente" in raw.get("error", ""):
                    result = {"engine_connected": False,
                              "error": "moteur non connecté",
                              "detail": f"{raw['error']}. Impossible d'appeler le moteur. "
                                        "Aucun résultat n'est simulé (ce serait fatal pour un projet anti-hallucination)."}
                else:
                    result = _enrich(raw)
                    _format_answer(result, payload.get("message", ""), payload.get("model", "claude"))
        except Exception as exc:  # défensif : jamais de 500 opaque au front
            result = {"engine_connected": True, "error": f"{type(exc).__name__}: {exc}",
                      "trace": traceback.format_exc()}

        self._send_json(result)


def main():
    host = os.environ.get("WEBAPP_HOST", "127.0.0.1")
    port = int(os.environ.get("WEBAPP_PORT", "8770"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Hallucide, front de chat : http://localhost:{port}  (Ctrl+C pour arrêter)")
    base_url = os.environ.get("MODEL_BASE_URL", "http://localhost:8000/v1")
    model_name = os.environ.get("MODEL_NAME", "google/gemma-4-E4B-it")
    print(f"Modèle par défaut : Gemma 4 -> {model_name} @ {base_url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt.")
        server.shutdown()


if __name__ == "__main__":
    main()
