"""
Planner agent
=============
Takes the user query and decomposes it into a structured plan of
sub-tasks that downstream agents will execute.

Outputs a JSON plan with the following schema:
{
  "complexity": "simple" | "moderate" | "complex",
  "sub_tasks": [
    {"id": 1, "task": "...", "needs_retrieval": bool},
    ...
  ],
  "requires_domain_knowledge": bool,
  "key_entities": ["...", "..."],
  "reasoning_strategy": "..."
}
"""

from __future__ import annotations

import json
import logging
import re

from argus.agents.base import BaseAgent

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = """You are the Planner agent in the ARGUS multi-agent reasoning system.
Your sole job is to analyse the user's query and produce a concise, structured plan
that will guide the reasoning pipeline.

Output ONLY valid JSON matching this schema:
{
  "complexity": "simple" | "moderate" | "complex",
  "sub_tasks": [
    {"id": <int>, "task": "<string>", "needs_retrieval": <bool>}
  ],
  "requires_domain_knowledge": <bool>,
  "key_entities": ["<string>"],
  "reasoning_strategy": "<string — one sentence describing the best approach>"
}

Rules:
- At most 5 sub-tasks.  Simple queries need only 1–2.
- "needs_retrieval" = true only if external context is likely necessary.
- Do NOT include any text outside the JSON object."""


class PlannerAgent(BaseAgent):
    NAME = "Planner"
    STEP_TYPE = "plan"

    def run(self, state: dict) -> dict:
        query = state["query"]
        similar_episodes = state.get("similar_episodes", [])

        # Build context from past episodes if available
        episode_context = ""
        if similar_episodes:
            examples = similar_episodes[:2]
            episode_context = "\n\nSimilar past queries (for reference):\n" + "\n".join(
                f"  - \"{ep['past_query']}\" "
                f"(uncertainty: {ep['metadata'].get('uncertainty_level', 'N/A')})"
                for ep in examples
            )

        user_message = f"Query to plan:\n{query}{episode_context}"

        step = self._call_llm(
            messages=[{"role": "user", "content": user_message}],
            system=PLANNER_SYSTEM,
            temperature=0.2,   # Low temperature for deterministic planning
            max_tokens=512,
            step_input_summary=f"Plan: {query[:80]}",
        )

        plan = self._parse_plan(step.output)
        logger.info(
            f"[Planner] Complexity={plan.get('complexity')}  "
            f"Sub-tasks={len(plan.get('sub_tasks', []))}"
        )

        return {
            "plan": plan,
            "trace": state.get("trace", []) + [step.as_dict()],
        }

    @staticmethod
    def _parse_plan(raw: str) -> dict:
        """Safely parse the JSON plan, with fallback for malformed output."""
        try:
            # Strip markdown fences if present
            clean = re.sub(r"```(?:json)?", "", raw).strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            logger.warning("[Planner] JSON parse failed.  Using default plan.")
            return {
                "complexity": "moderate",
                "sub_tasks": [{"id": 1, "task": "Answer the query directly.", "needs_retrieval": False}],
                "requires_domain_knowledge": False,
                "key_entities": [],
                "reasoning_strategy": "Direct reasoning.",
            }
