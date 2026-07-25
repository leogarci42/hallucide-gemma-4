from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from hallucide.core_types.exceptions import HallucideError
from hallucide.decomposition.llm import ModelProvider

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

_ROLE_MAP = {"assistant": "model"}  # Gemini utilise "model", pas "assistant"


class GeminiModelProvider:
    """Client pour l'API Gemini réelle (generativelanguage.googleapis.com).

    Vérifié en direct : endpoint v1beta/models/{model}:generateContent,
    clé en query param (pas Bearer), payload {"contents": [...]}.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        api_url: str | None = None,
        max_output_tokens: int = 2048,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_url = api_url or f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.max_output_tokens = max_output_tokens
        self.supports_forced_tool_calling = False

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, str]] | None = None,
        tool_choice: str | None = None,
    ) -> dict[str, Any]:
        if tool_choice == "required" and not self.supports_forced_tool_calling:
            raise HallucideError("Gemini provider does not support forced tool calling.")

        system_messages = [m["content"] for m in messages if m.get("role") == "system"]
        contents = [
            {"role": _ROLE_MAP.get(m.get("role", "user"), "user"), "parts": [{"text": m.get("content", "")}]}
            for m in messages
            if m.get("role") != "system"
        ]

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": self.max_output_tokens,
                # Gemini 2.5 consomme une part du budget de tokens en
                # "réflexion" interne avant de produire le texte -- inutile
                # et coûteux pour des réponses structurées courtes (§5/§6) ;
                # désactivé pour ne pas tronquer la sortie utile (observé en
                # direct : réponse coupée en plein milieu du JSON attendu).
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        if system_messages:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(system_messages)}]}

        try:
            response_data = self._send_request(payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise HallucideError(
                f"Gemini API error: {exc.code} {exc.reason} for model '{self.model}': {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise HallucideError(f"Gemini API connection error: {exc.reason}") from exc

        return {"text": self._extract_text(response_data)}

    def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.api_url}?{urllib.parse.urlencode({'key': self.api_key})}"
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HallucideError("Gemini API returned malformed JSON.") from exc

    def _extract_text(self, data: dict[str, Any]) -> str:
        candidates = data.get("candidates")
        if isinstance(candidates, list) and candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts and isinstance(parts[0], dict) and "text" in parts[0]:
                return str(parts[0]["text"])

        raise HallucideError("Unable to extract text output from Gemini API response.")
