"""
Reasoner agent
==============
The core reasoning workhorse.  Generates a chain-of-thought response
grounded in retrieved context and the planner's strategy.

Critically, after generating a response the Reasoner also asks the
SemanticUncertaintyQuantifier to estimate epistemic uncertainty by
sampling N additional responses.  The uncertainty result is then used
by the orchestrator to decide whether to route to the Critic or loop
back for a deeper pass.
"""

from __future__ import annotations

import logging

from argus.agents.base import BaseAgent
from argus.core.uncertainty import SemanticUncertaintyQuantifier

logger = logging.getLogger(__name__)

REASONER_SYSTEM = """You are the Reasoner agent in the ARGUS multi-agent system.
Your task is to answer the user's query using careful chain-of-thought reasoning.

Guidelines:
1. Work through the problem step by step.  Show your thinking explicitly using
   "Step 1:", "Step 2:", etc.
2. If context passages are provided, cite them with [Passage N] notation.
3. After your reasoning, state a clear, direct answer.
4. End with a one-line confidence statement:
   "Confidence: <low|medium|high> — <one reason>"
5. Be precise.  Do not pad with unnecessary caveats."""

REASONER_SYSTEM_LOOP = """You are the Reasoner agent in a second-pass reasoning loop.
A previous attempt produced high semantic uncertainty.  Re-examine the problem more
carefully, consider alternative interpretations, and produce a more definitive answer.
Be explicit about which interpretation you are adopting and why."""


class ReasonerAgent(BaseAgent):
    NAME = "Reasoner"
    STEP_TYPE = "reason"

    def __init__(self, uncertainty_quantifier: SemanticUncertaintyQuantifier) -> None:
        super().__init__()
        self._uq = uncertainty_quantifier

    def run(self, state: dict) -> dict:
        query = state["query"]
        plan = state.get("plan", {})
        context_passages = state.get("context_passages", [])
        reasoning_loop = state.get("reasoning_loop", 0)   # How many times we've looped

        # Build the prompt
        context_block = self._format_context(context_passages)
        strategy = plan.get("reasoning_strategy", "")
        sub_tasks = plan.get("sub_tasks", [])

        sub_task_text = ""
        if sub_tasks:
            sub_task_text = "\n\nSub-tasks to address:\n" + "\n".join(
                f"  {t['id']}. {t['task']}" for t in sub_tasks
            )

        user_message = (
            f"Query: {query}"
            f"{sub_task_text}"
            + (f"\n\nReasoning strategy: {strategy}" if strategy else "")
            + (f"\n\n{context_block}" if context_block else "")
        )

        system = REASONER_SYSTEM_LOOP if reasoning_loop > 0 else REASONER_SYSTEM

        # Primary reasoning pass
        step = self._call_llm(
            messages=[{"role": "user", "content": user_message}],
            system=system,
            temperature=0.7,
            step_input_summary=f"{'Loop ' + str(reasoning_loop+1) + ': ' if reasoning_loop else ''}Reason: {query[:70]}",
        )

        # Semantic uncertainty estimation
        logger.info(f"[Reasoner] Estimating uncertainty (N={self._uq._n_samples} samples)…")
        uq_result = self._uq.estimate(query=user_message, system_prompt=system)

        logger.info(
            f"[Reasoner] H_norm={uq_result.entropy_normalised:.3f}  "
            f"level={uq_result.uncertainty_level}  "
            f"clusters={uq_result.n_clusters}  "
            f"consensus={uq_result.consensus_ratio:.0%}"
        )

        trace = state.get("trace", []) + [
            step.as_dict(),
            {
                "agent": "Reasoner",
                "type": "uncertainty_estimation",
                "input_summary": f"Semantic entropy over {uq_result.n_clusters} cluster(s)",
                "output": uq_result.reasoning,
                "latency_ms": 0,
                "tokens": 0,
                "entropy_normalised": round(uq_result.entropy_normalised, 4),
                "uncertainty_level": uq_result.uncertainty_level,
                "n_clusters": uq_result.n_clusters,
                "consensus_ratio": round(uq_result.consensus_ratio, 4),
            },
        ]

        return {
            "draft_response": step.output,
            "uncertainty_result": uq_result.as_dict,
            "entropy_normalised": uq_result.entropy_normalised,
            "uncertainty_level": uq_result.uncertainty_level,
            "needs_reroute": self._uq.exceeds_threshold(uq_result),
            "reasoning_loop": reasoning_loop + 1,
            "trace": trace,
        }

    @staticmethod
    def _format_context(passages: list[dict]) -> str:
        if not passages:
            return ""
        lines = ["Relevant context:"]
        for i, p in enumerate(passages, 1):
            score = p.get("relevance_score", p.get("similarity", 0))
            lines.append(f"[Passage {i}] (relevance={score:.2f})\n{p.get('content', '')}")
        return "\n\n".join(lines)
