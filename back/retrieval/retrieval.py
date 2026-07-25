from __future__ import annotations

from typing import Protocol

from hallucide.core_types.exceptions import RetrievalError
from hallucide.core_types.types import Intent, Passage, RetrievalState


class RetrievalProvider(Protocol):
    def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]) -> Passage:
        ...


def advance_retrieval(
    intent: Intent,
    provider: RetrievalProvider,
    query: dict[str, str],
    state: RetrievalState,
    max_hops: int,
) -> tuple[Passage, RetrievalState]:
    if state.hop_count >= max_hops:
        raise RetrievalError("Maximum retrieval hops exceeded.")

    if state.remaining_budget <= 0:
        raise RetrievalError("Retrieval budget exhausted.")

    passage = provider.retrieve(intent, state, query)
    if passage.source_id in state.visited_documents:
        raise RetrievalError("Document already visited.")

    state.visited_documents.add(passage.source_id)
    state.hop_count += 1
    state.remaining_budget = max(0, state.remaining_budget - 1)
    return passage, state
