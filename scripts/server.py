#!/usr/bin/env python3
"""Serveur HTTP de l'engine, contrat attendu par front/lib/types.ts (voir
README « Running the interface ») :

  GET  /health  -> { "model": "..." }
  POST /ask     -> { "question": "..." } -> Answer (draft/verified/claims/verdict/sources/...)

Zéro dépendance (stdlib http.server), même convention que l'ancien
app/server.py. Aucune modification du moteur Hallucide : réutilise
Hallucide.ask() avec GemmaModelProvider + AlienRetrievalProvider +
DomainRouter (étages 1+2+3), exactement comme scripts/ask_medical.py.

Lancement : python -m scripts.server        (port 8080 par défaut)
            ENGINE_PORT=8081 python -m scripts.server
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

env_path = WORKSPACE / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from hallucide import GemmaModelProvider, Hallucide  # noqa: E402
from hallucide.core_types.exceptions import HallucideError  # noqa: E402
from hallucide.core_types.types import ClaimStatus  # noqa: E402
from hallucide.verification.semantic_similarity import is_distant_reformulation  # noqa: E402

from scripts.ask_medical import DATASET_IDS, _build_retrieval_provider, _route_dataset  # noqa: E402

MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-4-E4B-it")

_LITERAL_OK = (ClaimStatus.AUTHENTIFIÉ, ClaimStatus.CITÉ_NON_OPPOSABLE, ClaimStatus.DONNÉE_TRACÉE)


def _build_model_provider() -> GemmaModelProvider:
    return GemmaModelProvider(
        base_url=os.environ.get("MODEL_BASE_URL", "http://localhost:8000/v1"),
        model=MODEL_NAME,
    )


def _claim_to_json(claim, passage, index: int) -> dict:
    literal_pass = claim.status in _LITERAL_OK
    semantic_pass: bool | None
    if claim.status == ClaimStatus.INTERPRÉTATION:
        semantic_pass = not is_distant_reformulation(claim.ref, passage.text)
    elif literal_pass:
        semantic_pass = True
    else:
        semantic_pass = False

    if literal_pass or (claim.status == ClaimStatus.INTERPRÉTATION and semantic_pass):
        status = "grounded"
    elif claim.status == ClaimStatus.NON_AUTHENTIFIÉ:
        status = "hallucinated"
    else:
        status = "unverifiable"

    matched_chunk = None
    for chunk in passage.metadata.get("chunks", []):
        if claim.ref and claim.ref in chunk.get("text", ""):
            matched_chunk = chunk
            break

    source = {
        "id": matched_chunk["entry_id"] if matched_chunk else str(passage.source_id),
        "title": passage.metadata.get("dataset_id", str(passage.source_id)),
        "passage": (matched_chunk["text"] if matched_chunk else passage.text)[:2000],
    }

    return {
        "id": f"c{index}",
        "text": claim.ref,
        "status": status,
        "semanticPass": semantic_pass,
        "literalPass": literal_pass,
        "source": source,
    }, status == "grounded"


def run_ask(question: str) -> dict:
    """Étages 1+2+3, réponse au format Answer (front/lib/types.ts)."""
    start = time.monotonic()
    model_provider = _build_model_provider()

    domain_name = _route_dataset(model_provider, question, None)
    if domain_name is None:
        return {
            "draft": "",
            "verified": "",
            "claims": [],
            "verdict": "refused",
            "sources": [],
            "model": MODEL_NAME,
            "latencyMs": int((time.monotonic() - start) * 1000),
        }

    dataset_id = DATASET_IDS.get(domain_name, domain_name)
    retrieval_provider, _is_real = _build_retrieval_provider()
    guard = Hallucide(model_provider=model_provider, retrieval_provider=retrieval_provider)

    try:
        result = guard.ask(message=question, query={"dataset_id": dataset_id})
    except HallucideError as exc:
        raise

    claims_json: list[dict] = []
    grounded_count = 0
    context_passages = 0
    context_chars = 0
    for r in result.orchestration.results:
        context_passages = max(context_passages, r.passage.metadata.get("nb_passages", 0) or 0)
        context_chars = max(context_chars, r.passage.metadata.get("nb_chars", len(r.passage.text)))
        for claim in r.verification.claims:
            claim_json, grounded = _claim_to_json(claim, r.passage, len(claims_json) + 1)
            claims_json.append(claim_json)
            grounded_count += int(grounded)

    if not claims_json:
        verdict = "unsupported"
    elif grounded_count == len(claims_json):
        verdict = "grounded"
    elif grounded_count == 0:
        verdict = "unsupported"
    else:
        verdict = "partial"

    draft = " ".join(c["text"] for c in claims_json)
    verified = " ".join(c["text"] for c in claims_json if c["status"] == "grounded")

    sources_by_id: dict[str, dict] = {}
    for c in claims_json:
        src = c.get("source")
        if src:
            sources_by_id[src["id"]] = src

    return {
        "draft": draft,
        "verified": verified,
        "claims": claims_json,
        "verdict": verdict,
        "sources": list(sources_by_id.values()),
        "model": MODEL_NAME,
        "dataset": domain_name,
        "contextPassages": context_passages or None,
        "contextTokens": context_chars // 4 if context_chars else None,
        "latencyMs": int((time.monotonic() - start) * 1000),
    }


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # noqa: A003 -- réduit le bruit stdout par défaut
        print(f"[server] {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json({"model": MODEL_NAME})
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/ask":
            self._send_json({"error": "not found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send_json({"error": "body must be JSON"}, status=400)
            return

        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            self._send_json({"error": "question is required"}, status=400)
            return

        try:
            answer = run_ask(question.strip())
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=502)
            return

        self._send_json(answer)


def main() -> None:
    host = os.environ.get("ENGINE_HOST", "0.0.0.0")
    port = int(os.environ.get("ENGINE_PORT", "8080"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Engine: http://{host}:{port}  (GET /health, POST /ask)  model={MODEL_NAME}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
