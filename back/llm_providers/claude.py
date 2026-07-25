from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from hallucide.core_types.exceptions import HallucideError
from hallucide.decomposition.llm import ModelProvider

# Par défaut : le modèle Claude le plus capable (cf. référence API Anthropic).
# Surchargeable via le paramètre `model` du constructeur.
DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"

# Version d'API Anthropic (en-tête obligatoire, indépendant du modèle).
_ANTHROPIC_VERSION = "2023-06-01"


class ClaudeModelProvider:
    """Client pour l'API Claude réelle (api.anthropic.com), même forme que les
    providers Mistral et Gemini : stdlib `urllib` uniquement, aucune dépendance.

    Spécificités de l'API Messages d'Anthropic (vérifiées sur la référence) :
      - endpoint /v1/messages, en-tête `x-api-key` (pas de Bearer),
      - en-tête `anthropic-version` obligatoire,
      - le prompt système est un champ de TÊTE `system`, séparé des `messages`
        (les rôles `system` ne sont pas acceptés dans le tableau `messages`),
      - la réponse est une liste de blocs `content` ; le texte est dans le
        premier bloc de type `text`.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_CLAUDE_MODEL,
        api_url: str | None = None,
        max_output_tokens: int = 2048,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_url = api_url or "https://api.anthropic.com/v1/messages"
        self.max_output_tokens = max_output_tokens
        self.supports_forced_tool_calling = False

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, str]] | None = None,
        tool_choice: str | None = None,
    ) -> dict[str, Any]:
        if tool_choice == "required" and not self.supports_forced_tool_calling:
            raise HallucideError("Claude provider does not support forced tool calling.")

        # §API Anthropic : le prompt système est extrait vers le champ `system`
        # de tête ; seuls user/assistant restent dans `messages`.
        system_messages = [m["content"] for m in messages if m.get("role") == "system"]
        chat_messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
            if m.get("role") != "system"
        ]

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "messages": chat_messages,
        }
        if system_messages:
            payload["system"] = "\n".join(system_messages)

        try:
            response_data = self._send_request(payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise HallucideError(
                f"Claude API error: {exc.code} {exc.reason} for model '{self.model}': {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise HallucideError(f"Claude API connection error: {exc.reason}") from exc

        return {"text": self._extract_text(response_data)}

    def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.api_url,
            data=data,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HallucideError("Claude API returned malformed JSON.") from exc

    def _extract_text(self, data: dict[str, Any]) -> str:
        # §API Anthropic : content est une liste de blocs ; le texte utile est
        # le premier bloc de type "text". Une réponse refusée (stop_reason
        # "refusal") a un content vide -> pas de bloc texte -> erreur explicite.
        content = data.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        return text

        raise HallucideError("Unable to extract text output from Claude API response.")
