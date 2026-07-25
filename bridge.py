#!/usr/bin/env python3
"""HTTP bridge between the engine in src/ and the interface in front/.

The engine is a library with a command-line entry point (scripts/ask_medical.py);
the interface speaks HTTP. This exposes the same pipeline on two routes, in the
contract the interface reads:

    GET  /health  -> {"model": "..."}
    POST /ask     -> {"draft", "verified", "verdict", "claims", ...}

Nothing in src/ is touched: the closed domain list, the dataset ids and the
retrieval wiring are imported from scripts/ask_medical.py, so a change there
carries over here.

    python bridge.py            # port 8000
    PORT=9000 python bridge.py

It answers 503 rather than serving anything invented: no model backend, no
Alien token, no dataset the router will accept. The interface renders that
state honestly and offers a retry.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

# Importing the script loads .env and gives us the closed domain list, the
# dataset ids and the retrieval builder, all from one place.
from scripts.ask_medical import (  # noqa: E402
    DATASET_IDS,
    DOMAINS,
    FIXED_DATASET_NAME,
    _build_retrieval_provider,
)

from hallucide import GemmaModelProvider, Hallucide  # noqa: E402
from hallucide.core_types.exceptions import HallucideError  # noqa: E402
from hallucide.core_types.types import ClaimStatus  # noqa: E402
from hallucide.decomposition.routing import DomainRouter  # noqa: E402
from hallucide.verification.normalization import normalize_text  # noqa: E402
from hallucide.verification.semantic_similarity import (  # noqa: E402
    DEFAULT_DISTANCE_THRESHOLD,
    similarity_score,
)

PORT = int(os.environ.get("PORT", "8000"))
MODEL_BASE_URL = os.environ.get("MODEL_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-4-E4B-it")
MAX_QUESTION_CHARS = 600

# A claim the deterministic check found word for word in the source passage.
# INTERPRÉTATION is a reformulation: the check ran and could not confirm it
# either way, which is not the same as finding it contradicted.
_VERBATIM_PASS = {
    ClaimStatus.AUTHENTIFIÉ,
    ClaimStatus.DONNÉE_TRACÉE,
    ClaimStatus.CITÉ_NON_OPPOSABLE,
}
_VERBATIM_FAIL = {ClaimStatus.NON_AUTHENTIFIÉ}


def _model_reachable(timeout: float = 3.0) -> bool:
    """The OpenAI-compatible backend serving Gemma answers /models."""
    url = MODEL_BASE_URL.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 500
    except urllib.error.HTTPError as exc:
        # 401/404 still means something is listening and speaking HTTP
        return exc.code < 500
    except Exception:
        return False


def _claim_source(claim_ref: str, passage: Any) -> dict[str, Any] | None:
    """The individual chunk a claim was matched against, when the retrieval
    provider kept them. Falls back to the whole passage, and to nothing at all
    rather than pointing at a chunk that does not contain the claim."""
    chunks = (passage.metadata or {}).get("chunks") or []
    needle = normalize_text(claim_ref)

    for chunk in chunks:
        text = chunk.get("text", "")
        if needle and needle in normalize_text(text):
            entry = chunk.get("entry_id") or passage.source_id
            return {
                "id": str(entry),
                "title": f"{passage.source_type} · {entry}",
                "passage": text,
            }

    if not chunks:
        return {
            "id": passage.source_id,
            "title": f"{passage.source_type} · {passage.source_id}",
            "passage": passage.text,
        }
    return None


def _map_claim(index: int, claim: Any, passage: Any) -> dict[str, Any]:
    """One engine claim in the interface's shape.

    - the literal check (7b) is the engine's deterministic verbatim check
    - the semantic check (7a) is the engine's similarity score against the
      passage, on the same threshold the risk floor uses
    - a claim is valid only when both pass, which is the aggregation the
      interface displays
    """
    if claim.status in _VERBATIM_PASS:
        literal_pass: bool | None = True
    elif claim.status in _VERBATIM_FAIL:
        literal_pass = False
    else:
        literal_pass = None  # reformulation: the check could not settle it

    semantic_pass = similarity_score(claim.ref, passage.text) >= DEFAULT_DISTANCE_THRESHOLD

    if literal_pass is None:
        status = "unverifiable"
    elif literal_pass and semantic_pass:
        status = "grounded"
    else:
        status = "hallucinated"

    out: dict[str, Any] = {
        "id": f"claim-{index}",
        "text": claim.ref,
        "status": status,
        "semanticPass": semantic_pass,
        "literalPass": literal_pass,
    }
    source = _claim_source(claim.ref, passage) if status == "grounded" else None
    if source:
        out["source"] = source
    return out


def _refusal(question: str) -> dict[str, Any]:
    """No domain in the closed list covers the question, or the router's answer
    was not in the list. The pipeline stops before anything is generated, so
    there is nothing to hallucinate."""
    return {
        "draft": "",
        "verified": "",
        "verdict": "refused",
        "claims": [],
        "sources": [],
        "model": MODEL_NAME,
        "question": question,
    }


def _answer(result: Any, domain_name: str, started: float) -> dict[str, Any]:
    claims: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    passages = 0
    chars = 0

    for execution in result.orchestration.results:
        passage = execution.passage
        meta = passage.metadata or {}
        passages += int(meta.get("nb_passages", 0) or 0)
        chars += int(meta.get("nb_chars", len(passage.text)) or 0)

        for claim in execution.verification.claims:
            mapped = _map_claim(len(claims), claim, passage)
            claims.append(mapped)

        for chunk in meta.get("chunks") or []:
            entry = str(chunk.get("entry_id") or "")
            if entry and entry not in seen_sources:
                seen_sources.add(entry)
                sources.append({"id": entry, "title": f"{passage.source_type} · {entry}"})

    kept = [c["text"] for c in claims if c["status"] == "grounded"]

    if not claims:
        verdict = "refused"
    elif all(c["status"] == "grounded" for c in claims):
        verdict = "grounded"
    elif kept:
        verdict = "partial"
    else:
        verdict = "unsupported"

    payload: dict[str, Any] = {
        "draft": " ".join(c["text"] for c in claims),
        "verified": " ".join(kept),
        "verdict": verdict,
        "claims": claims,
        "sources": sources,
        "model": MODEL_NAME,
        "dataset": domain_name,
        "latencyMs": int((time.monotonic() - started) * 1000),
    }
    # Only reported when the retrieval layer actually counted them.
    if passages:
        payload["contextPassages"] = passages
    if chars:
        # characters, not tokens: the engine does not tokenise, and a converted
        # number would be an estimate presented as a measurement
        payload["contextChars"] = chars
    return payload


def run_question(question: str) -> tuple[int, dict[str, Any]]:
    started = time.monotonic()

    provider = GemmaModelProvider(base_url=MODEL_BASE_URL, model=MODEL_NAME)

    # Stage 1: pick a domain from the closed list, or refuse. A technical
    # failure of the router falls back to the fixed dataset, the way the
    # command-line entry point does: routing must not break the chain.
    try:
        domain_name = DomainRouter(provider, DOMAINS).route(question)
    except HallucideError:
        domain_name = FIXED_DATASET_NAME
    if domain_name is None:
        return 200, _refusal(question)

    retrieval_provider, is_real = _build_retrieval_provider()
    if not is_real:
        return 503, {
            "error": "ALIEN_API_TOKEN is not set: the engine would answer from a "
            "placeholder passage, which must never be served as a source."
        }

    guard = Hallucide(model_provider=provider, retrieval_provider=retrieval_provider)
    try:
        result = guard.ask(message=question, query={"dataset_id": DATASET_IDS.get(domain_name, domain_name)})
    except HallucideError as exc:
        return 502, {"error": f"{type(exc).__name__}: {exc}"}

    return 200, _answer(result, domain_name, started)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # quieter than the default
        pass

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/health":
            self._send(404, {"error": "not found"})
            return
        if not _model_reachable():
            self._send(503, {"error": f"no model backend at {MODEL_BASE_URL}"})
            return
        self._send(200, {"model": MODEL_NAME})

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/ask":
            self._send(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            question = json.loads(self.rfile.read(length) or b"{}").get("question")
        except (ValueError, json.JSONDecodeError):
            self._send(400, {"error": "body must be JSON"})
            return

        if not isinstance(question, str) or not question.strip():
            self._send(400, {"error": "question is required"})
            return
        if len(question) > MAX_QUESTION_CHARS:
            self._send(400, {"error": f"question must be at most {MAX_QUESTION_CHARS} characters"})
            return

        try:
            status, payload = run_question(question.strip())
        except Exception as exc:  # a crash here must not take the server down
            traceback.print_exc()
            status, payload = 500, {"error": f"{type(exc).__name__}: {exc}"}

        self._send(status, payload)


def main() -> None:
    print(f"bridge on http://localhost:{PORT}")
    print(f"  model     {MODEL_NAME} via {MODEL_BASE_URL}")
    print(f"  alien     {'token set' if os.environ.get('ALIEN_API_TOKEN') else 'NO TOKEN — /ask will answer 503'}")
    print(f"  domains   {', '.join(DOMAINS)}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
