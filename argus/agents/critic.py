"""
Critic agent
============
The Critic performs a structured audit of the Reasoner's draft response
against a six-dimension rubric.  This is inspired by Constitutional AI
(Bai et al., 2022) but applied to post-hoc critique rather than
self-revision training.

Rubric dimensions
-----------------
1. Logical coherence      — Are the reasoning steps valid?
2. Factual grounding      — Are claims supported by context or well-known facts?
3. Completeness           — Are all sub-tasks addressed?
4. Calibration            — Does stated confidence match actual uncertainty?
5. Relevance              — Does the response answer the actual question?
6. Conciseness            — Is there unnecessary padding or repetition?

Each dimension scored 0.0–1.0.  The Critic also produces:
  - A list of specific issues (if any).
  - An improved version of the response (only if score < 0.75).
"""

from __future__ import annotations

import json
import logging
import re

from argus.agents.base import BaseAgent

logger = logging.getLogger(__name__)

CRITIC_SYSTEM = """You are the Critic agent in the ARGUS multi-agent reasoning system.
Your role is to rigorously audit the Reasoner's draft response.

Score the response on each dimension (0.0–1.0):
  - logical_coherence: Are the reasoning steps valid and non-circular?
  - factual_grounding: Are claims supported by provided context or established fact?
  - completeness: Are all aspects of the query addressed?
  - calibration: Does the stated confidence match the actual quality of reasoning?
  - relevance: Does the response directly answer what was asked?
  - conciseness: Is the response free of padding and repetition?

Output ONLY valid JSON:
{
  "scores": {
    "logical_coherence": <float>,
    "factual_grounding": <float>,
    "completeness": <float>,
    "calibration": <float>,
    "relevance": <float>,
    "conciseness": <float>
  },
  "mean_score": <float>,
  "issues": ["<specific issue 1>", "..."],
  "improved_response": "<full revised response, or null if mean_score >= 0.75>",
  "critique_summary": "<2–3 sentence overall assessment>"
}

Be specific in your issues.  Do not praise the response — only identify problems."""


class CriticAgent(BaseAgent):
    NAME = "Critic"
    STEP_TYPE = "critique"

    PASS_THRESHOLD = 0.75   # Responses above this score proceed to synthesis unchanged

    def run(self, state: dict) -> dict:
        query = state["query"]
        draft = state.get("draft_response", "")
        uncertainty_level = state.get("uncertainty_level", "medium")
        plan = state.get("plan", {})
        sub_tasks = plan.get("sub_tasks", [])

        if not draft:
            logger.warning("[Critic] No draft response to critique.  Skipping.")
            return {"critique_result": None, "trace": state.get("trace", [])}

        sub_task_context = ""
        if sub_tasks:
            sub_task_context = "Sub-tasks that should be addressed:\n" + "\n".join(
                f"  {t['id']}. {t['task']}" for t in sub_tasks
            )

        user_message = (
            f"Original query: {query}\n\n"
            f"{sub_task_context}\n\n"
            f"Draft response to audit:\n{draft}\n\n"
            f"Note: Reasoner reported uncertainty level = '{uncertainty_level}'."
        )

        step = self._call_llm(
            messages=[{"role": "user", "content": user_message}],
            system=CRITIC_SYSTEM,
            temperature=0.3,   # Semi-deterministic critique
            max_tokens=1024,
            step_input_summary=f"Critique draft for: {query[:70]}",
        )

        result = self._parse_critique(step.output)
        mean_score = result.get("mean_score", 0.0)

        # Use improved response if score is below threshold
        if mean_score < self.PASS_THRESHOLD and result.get("improved_response"):
            revised = result["improved_response"]
            logger.info(f"[Critic] Low score ({mean_score:.2f}) — adopting improved response.")
        else:
            revised = draft
            if mean_score < self.PASS_THRESHOLD:
                logger.warning(f"[Critic] Low score ({mean_score:.2f}) but no improvement provided.")

        step_dict = step.as_dict()
        step_dict.update({
            "mean_score": round(mean_score, 3),
            "issues": result.get("issues", []),
            "critique_summary": result.get("critique_summary", ""),
            "scores": result.get("scores", {}),
        })

        logger.info(
            f"[Critic] Mean score: {mean_score:.2f}  "
            f"Issues: {len(result.get('issues', []))}  "
            f"Pass: {mean_score >= self.PASS_THRESHOLD}"
        )

        return {
            "critique_result": result,
            "revised_response": revised,
            "critique_passed": mean_score >= self.PASS_THRESHOLD,
            "trace": state.get("trace", []) + [step_dict],
        }

    @staticmethod
    def _parse_critique(raw: str) -> dict:
        try:
            clean = re.sub(r"```(?:json)?", "", raw).strip()
            data = json.loads(clean)
            # Recompute mean score from individual scores for safety
            scores = data.get("scores", {})
            if scores:
                data["mean_score"] = sum(scores.values()) / len(scores)
            return data
        except json.JSONDecodeError:
            logger.warning("[Critic] Failed to parse JSON critique.")
            return {
                "scores": {},
                "mean_score": 0.5,
                "issues": ["Parse error — using fallback."],
                "improved_response": None,
                "critique_summary": "Critique parsing failed.",
            }
