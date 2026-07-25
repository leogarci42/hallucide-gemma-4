from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from hallucide.core_types.exceptions import HallucideError
from hallucide.decomposition.llm import ModelProvider

# Défauts pensés pour permettre de basculer Ollama/vLLM/autre backend compatible
# OpenAI en changeant uniquement MODEL_BASE_URL / MODEL_NAME (§7 du hackathon :
# jamais d'URL en dur, pour pouvoir migrer vers vLLM sans toucher au code).
DEFAULT_GEMMA_BASE_URL = "http://localhost:8000/v1"
DEFAULT_GEMMA_MODEL = "google/gemma-4-E4B-it"


class GemmaModelProvider:
    """Client pour Gemma 4 servi via une API compatible OpenAI (vLLM ou Ollama
    en mode /v1), même forme que les providers Mistral/Claude/Gemini : stdlib
    `urllib` uniquement, aucune dépendance.

    Endpoint /v1/chat/completions, Bearer token optionnel (vLLM n'exige pas de
    clé par défaut ; Ollama non plus). URL et nom de modèle configurables pour
    permettre de pointer indifféremment sur Ollama (filet de développement) ou
    vLLM (déploiement temps réel, prix NVIDIA) sans changer le code appelant.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_GEMMA_BASE_URL,
        model: str = DEFAULT_GEMMA_MODEL,
        api_key: str | None = None,
        max_output_tokens: int = 2048,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.supports_forced_tool_calling = False

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, str]] | None = None,
        tool_choice: str | None = None,
    ) -> dict[str, Any]:
        if tool_choice == "required" and not self.supports_forced_tool_calling:
            raise HallucideError("Gemma provider does not support forced tool calling.")

        payload = {
            "model": self.model,
            "messages": [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages],
            "max_tokens": self.max_output_tokens,
        }

        try:
            response_data = self._send_request(payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise HallucideError(
                f"Gemma API error: {exc.code} {exc.reason} for model '{self.model}': {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise HallucideError(f"Gemma API connection error: {exc.reason}") from exc

        return {"text": self._extract_text(response_data)}

    def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HallucideError("Gemma API returned malformed JSON.") from exc

    def _extract_text(self, data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content

        raise HallucideError("Unable to extract text output from Gemma API response.")
