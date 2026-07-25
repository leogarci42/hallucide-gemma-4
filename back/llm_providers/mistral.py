from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from hallucide.core_types.exceptions import HallucideError
from hallucide.decomposition.llm import ModelProvider

DEFAULT_MISTRAL_MODEL = "mistral-small-latest"


class MistralModelProvider:
    """Client pour l'API Mistral réelle (api.mistral.ai).

    Vérifié en direct : endpoint /v1/chat/completions, Bearer token,
    payload {"model": ..., "messages": [{"role": ..., "content": ...}]}.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MISTRAL_MODEL,
        api_url: str | None = None,
        max_output_tokens: int = 1024,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_url = api_url or "https://api.mistral.ai/v1/chat/completions"
        self.max_output_tokens = max_output_tokens
        self.supports_forced_tool_calling = False

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, str]] | None = None,
        tool_choice: str | None = None,
    ) -> dict[str, Any]:
        if tool_choice == "required" and not self.supports_forced_tool_calling:
            raise HallucideError("Mistral provider does not support forced tool calling.")

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
                f"Mistral API error: {exc.code} {exc.reason} for model '{self.model}': {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise HallucideError(f"Mistral API connection error: {exc.reason}") from exc

        return {"text": self._extract_text(response_data)}

    def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.api_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HallucideError("Mistral API returned malformed JSON.") from exc

    def _extract_text(self, data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content

        raise HallucideError("Unable to extract text output from Mistral API response.")
