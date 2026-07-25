from __future__ import annotations

import json
import re
from typing import Any, Protocol

from hallucide.core_types.exceptions import HallucideError, VerificationError
from hallucide.core_types.types import Claim, ClaimStatus, Intent, Passage


class ModelProvider(Protocol):
    supports_forced_tool_calling: bool

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, str]] | None = None,
        tool_choice: str | None = None,
    ) -> dict[str, Any]:
        ...


class MockModelProvider:
    def __init__(
        self,
        responses: dict[str, str] | None = None,
        supports_forced_tool_calling: bool = False,
    ) -> None:
        self.supports_forced_tool_calling = supports_forced_tool_calling
        self._responses = responses or {}

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, str]] | None = None,
        tool_choice: str | None = None,
    ) -> dict[str, Any]:
        if tool_choice == "required" and not self.supports_forced_tool_calling:
            raise HallucideError("Provider does not support forced tool calling.")

        prompt = next((m for m in messages if m.get("role") == "system"), None)
        if not prompt or not isinstance(prompt.get("content"), str):
            raise HallucideError("Unable to infer prompt type from model messages.")

        content = prompt["content"]
        if "<task>Break down the following user message" in content or "Découpe le message" in content:
            key = "decompose"
        elif "<task>Generate claims that answer the user question" in content or "CITATION vs REFORMULATION" in content:
            key = "claims"
        else:
            key = "default"

        return {"text": self._responses.get(key, "")}


class PromptBuilder:
    @staticmethod
    def build_decomposition_prompt(message: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "<task>Break down the following user message into atomic intents.</task>\n"
                    "<format>Respond strictly with a JSON array in the following format:\n"
                    '[{"id": "1", "question": "..."}, ...]</format>'
                ),
            },
            {"role": "user", "content": message},
        ]

    @staticmethod
    def build_claim_generation_prompt(intent: Intent, passage: Passage) -> list[dict[str, str]]:
        # Prompt B (§5): explicit citation vs paraphrase distinction.
        # The model must output exact verbatim extracts for AUTHENTIFIÉ, and mark any paraphrase as INTERPRÉTATION.
        return [
            {
                "role": "system",
                "content": (
                    "<task>Generate claims that answer the user question based solely on the provided official passage.</task>\n"
                    "<rules>\n"
                    '  <rule type="CITATION">An exact VERBATIM extract copied word-for-word from the passage -> status "AUTHENTIFIÉ".</rule>\n'
                    '  <rule type="REFORMULATION">A paraphrase or faithful summary using your own words -> status "INTERPRÉTATION".</rule>\n'
                    '  <rule type="CONSTRAINTS">Do NOT introduce facts absent from the passage. If relevant information exists, produce at least one claim; otherwise return an empty array [].</rule>\n'
                    "</rules>\n"
                    "<format>Respond strictly with a JSON array in the following format:\n"
                    '[{"ref": "...", "status": "AUTHENTIFIÉ|INTERPRÉTATION"}, ...]</format>'
                ),
            },
            {"role": "user", "content": f"Question: {intent.question}\nPassage: {passage.text}"},
        ]


_MARKDOWN_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_markdown_fence(text: str) -> str:
    """Certain LLMs wrap JSON responses in markdown code blocks despite strict formatting rules.
    This safely strips those code fences before parsing.
    """
    match = _MARKDOWN_FENCE_PATTERN.match(text.strip())
    return match.group(1).strip() if match else text


def _parse_json_response(text: str) -> Any:
    try:
        return json.loads(_strip_markdown_fence(text))
    except json.JSONDecodeError as exc:
        raise VerificationError("LLM response is not valid JSON.") from exc


def _extract_text_response(response: dict[str, Any]) -> str:
    if not isinstance(response, dict):
        raise HallucideError("LLM response must be a JSON object.")

    text = response.get("text")
    if not isinstance(text, str):
        raise HallucideError("LLM response missing text output.")

    text = text.strip()
    if not text:
        raise HallucideError("LLM returned empty text response.")

    return text


class PromptBasedDecomposer:
    """§6 : decomposition never retrieves passages itself -- the orchestrator
    calls MCP directly (§4 step 4). No tools needed here.
    """

    def __init__(self, model_provider: ModelProvider) -> None:
        self.model_provider = model_provider

    def decompose(self, message: str) -> list[Intent]:
        response = self.model_provider.generate(
            messages=PromptBuilder.build_decomposition_prompt(message),
            tools=[],
            tool_choice=None,
        )
        text = _extract_text_response(response)
        payload = _parse_json_response(text)
        if not isinstance(payload, list):
            raise HallucideError("Expected a JSON array of intents.")

        intents: list[Intent] = []
        for item in payload:
            if not isinstance(item, dict) or "id" not in item or "question" not in item:
                raise HallucideError("Invalid intent payload from LLM.")
            intents.append(Intent(id=str(item["id"]), question=str(item["question"])))
        return intents


class PromptBasedIntentGenerator:
    """§6 : constrained generation (step 6) receives the passage retrieved by
    the orchestrator -- it never calls tools directly.
    """

    def __init__(self, model_provider: ModelProvider) -> None:
        self.model_provider = model_provider

    def generate_claims(self, intent: Intent, passage: Passage) -> list[Claim]:
        response = self.model_provider.generate(
            messages=PromptBuilder.build_claim_generation_prompt(intent, passage),
            tools=[],
            tool_choice=None,
        )
        text = _extract_text_response(response)
        payload = _parse_json_response(text)
        if not isinstance(payload, list):
            raise HallucideError("Expected a JSON array of claims.")

        claims: list[Claim] = []
        for item in payload:
            if not isinstance(item, dict) or "ref" not in item or "status" not in item:
                raise HallucideError("Invalid claim payload from LLM.")
            status_value = str(item["status"])
            try:
                status = ClaimStatus(status_value)
            except ValueError as exc:
                raise HallucideError(f"Unsupported claim status: {status_value}") from exc
            claims.append(Claim(ref=str(item["ref"]), status=status))
        return claims
