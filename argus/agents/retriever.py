"""
Retriever agent
===============
Fetches relevant context from the ARGUS knowledge base using semantic
search, then applies a lightweight LLM-based reranking step to surface
the most useful passages.

Returns:
- A list of ranked context passages.
- A Boolean indicating whether retrieval yielded useful results.
"""

from __future__ import annotations

import json
import logging
import re

from argus.agents.base import BaseAgent
from argus.core.memory import ArgusMemory

logger = logging.getLogger(__name__)

RERANKER_SYSTEM = """You are a relevance judge.  Given a query and a list of
retrieved passages, rank them by relevance and filter out any that are
not useful for answering the query.

Output ONLY valid JSON:
{
  "ranked_passages": [
    {"rank": 1, "content": "<passage>", "relevance_score": <0.0–1.0>, "reason": "<why useful>"}
  ],
  "has_useful_context": <bool>
}

Keep at most 3 passages.  Set has_useful_context=false if none are useful."""


class RetrieverAgent(BaseAgent):
    NAME = "Retriever"
    STEP_TYPE = "retrieve"

    def __init__(self, memory: ArgusMemory) -> None:
        super().__init__()
        self._memory = memory

    def run(self, state: dict) -> dict:
        query = state["query"]
        plan = state.get("plan", {})

        # Check if retrieval is needed based on the planner's assessment
        needs_retrieval = plan.get("requires_domain_knowledge", False) or any(
            st.get("needs_retrieval", False)
            for st in plan.get("sub_tasks", [])
        )

        if not needs_retrieval:
            logger.info("[Retriever] Skipped — planner did not flag retrieval need.")
            return {
                "context_passages": [],
                "has_useful_context": False,
                "trace": state.get("trace", []) + [{
                    "agent": "Retriever",
                    "type": "retrieve",
                    "input_summary": "Retrieval skipped",
                    "output": "No retrieval required for this query.",
                    "latency_ms": 0,
                    "tokens": 0,
                }],
            }

        # Retrieve top-k passages
        raw_passages = self._memory.retrieve(query)

        if not raw_passages:
            logger.info("[Retriever] Knowledge base returned no results.")
            return {
                "context_passages": [],
                "has_useful_context": False,
                "trace": state.get("trace", []) + [{
                    "agent": "Retriever",
                    "type": "retrieve",
                    "input_summary": query[:80],
                    "output": "No relevant documents found in knowledge base.",
                    "latency_ms": 0,
                    "tokens": 0,
                }],
            }

        # Format for reranker
        passages_text = "\n\n".join(
            f"[Passage {i+1}] (similarity={p['similarity']:.3f})\n{p['content']}"
            for i, p in enumerate(raw_passages)
        )
        rerank_prompt = f"Query: {query}\n\nPassages:\n{passages_text}"

        step = self._call_llm(
            messages=[{"role": "user", "content": rerank_prompt}],
            system=RERANKER_SYSTEM,
            temperature=0.1,
            max_tokens=800,
            step_input_summary=f"Rerank {len(raw_passages)} passages for: {query[:60]}",
        )

        ranked = self._parse_ranked(step.output)

        logger.info(
            f"[Retriever] {len(raw_passages)} retrieved → "
            f"{len(ranked.get('ranked_passages', []))} after reranking.  "
            f"Has useful context: {ranked.get('has_useful_context')}"
        )

        return {
            "context_passages": ranked.get("ranked_passages", []),
            "has_useful_context": ranked.get("has_useful_context", False),
            "trace": state.get("trace", []) + [step.as_dict()],
        }

    @staticmethod
    def _parse_ranked(raw: str) -> dict:
        try:
            clean = re.sub(r"```(?:json)?", "", raw).strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            return {"ranked_passages": [], "has_useful_context": False}
