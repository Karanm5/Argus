"""
Synthesis agent
===============
The final agent in the pipeline.  It takes the (possibly revised) draft
response, the critique result, and the uncertainty estimate, and produces
a clean, calibrated final answer.

The Synthesis agent's job is to:
  1. Integrate any Critic improvements into a polished response.
  2. Attach a calibrated confidence statement that accurately reflects
     the pipeline's measured uncertainty.
  3. Surface the key reasoning steps concisely for the UI trace view.
  4. Flag any remaining caveats or limitations.
"""

from __future__ import annotations

import logging

from argus.agents.base import BaseAgent

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM = """You are the Synthesis agent in ARGUS, the final step before
the answer reaches the user.

Your inputs are:
  - The user's original query.
  - A draft response (possibly already improved by the Critic).
  - A critique summary with issues and a mean quality score.
  - A measured epistemic uncertainty score from semantic entropy analysis.

Your task:
  1. Produce a clean, well-structured final answer.
  2. Integrate any Critic feedback that was not already incorporated.
  3. Include a calibrated confidence statement at the end:
       "Confidence: <low|medium|high> (<reason>) — Semantic uncertainty: <entropy_norm>"
  4. Keep it concise.  No repetition.  No padding.

Do NOT include meta-commentary about the pipeline.  Just deliver the answer."""


class SynthesisAgent(BaseAgent):
    NAME = "Synthesis"
    STEP_TYPE = "synthesise"

    def run(self, state: dict) -> dict:
        query = state["query"]
        revised_response = state.get("revised_response") or state.get("draft_response", "")
        critique_result = state.get("critique_result") or {}
        uncertainty_result = state.get("uncertainty_result") or {}

        mean_score = critique_result.get("mean_score", 1.0)
        issues = critique_result.get("issues", [])
        critique_summary = critique_result.get("critique_summary", "No critique.")
        entropy_norm = uncertainty_result.get("entropy_normalised", 0.0)
        uncertainty_level = state.get("uncertainty_level", "medium")

        user_message = (
            f"Query: {query}\n\n"
            f"Draft response:\n{revised_response}\n\n"
            f"Critique summary (quality={mean_score:.2f}):\n{critique_summary}\n"
            + (f"Remaining issues:\n" + "\n".join(f"  - {i}" for i in issues[:3]) if issues else "")
            + f"\n\nMeasured epistemic uncertainty: {entropy_norm:.3f} (level: {uncertainty_level})"
        )

        step = self._call_llm(
            messages=[{"role": "user", "content": user_message}],
            system=SYNTHESIS_SYSTEM,
            temperature=0.4,
            step_input_summary=f"Synthesise final answer for: {query[:70]}",
        )

        step_dict = step.as_dict()
        step_dict.update({
            "entropy_normalised": round(entropy_norm, 4),
            "critique_mean_score": round(mean_score, 3),
            "is_final": True,
        })

        logger.info(f"[Synthesis] Final answer produced.  Length: {len(step.output)} chars.")

        return {
            "final_answer": step.output,
            "trace": state.get("trace", []) + [step_dict],
        }
