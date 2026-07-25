from __future__ import annotations

from typing import Any

from hallucide.core_types.exceptions import HallucideError
from hallucide.decomposition.llm import ModelProvider
from hallucide.analysis.trust import ensure_system_trust_store

DEFAULT_LITELLM_MODEL = "mistral/mistral-small-latest"


class LiteLLMModelProvider:
    """Implémentation de ModelProvider via LiteLLM (§17.1, recommandation
    explicite de la spec pour l'interface ModelProvider -- bascule API/local
    sans changer le code appelant).

    Vérifié en direct sur Mistral (`mistral/mistral-small-latest`) ; Gemini
    accessible par le même mécanisme (`gemini/gemini-2.5-flash`) mais non
    revérifié après le fix SSL faute de quota API disponible pendant les tests.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_LITELLM_MODEL,
        max_output_tokens: int = 2048,
    ) -> None:
        ensure_system_trust_store()
        self.api_key = api_key
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.supports_forced_tool_calling = False

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, str]] | None = None,
        tool_choice: str | None = None,
    ) -> dict[str, Any]:
        if tool_choice == "required" and not self.supports_forced_tool_calling:
            raise HallucideError("LiteLLM provider does not support forced tool calling.")

        import litellm

        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages],
                max_tokens=self.max_output_tokens,
                api_key=self.api_key,
            )
        except Exception as exc:
            raise HallucideError(f"LiteLLM error for model '{self.model}': {exc}") from exc

        return {"text": self._extract_text(response)}

    def _extract_text(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if choices:
            content = choices[0].message.content
            if isinstance(content, str):
                return content

        raise HallucideError("Unable to extract text output from LiteLLM response.")
