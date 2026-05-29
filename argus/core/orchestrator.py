"""
ARGUS Orchestrator
==================
Implements the multi-agent pipeline as a LangGraph StateGraph.

State flow
----------

  START
    │
    ▼
  [Planner]   — decomposes the query
    │
    ▼
  [Retriever] — fetches grounding context (conditional)
    │
    ▼
  [Reasoner]  — chain-of-thought + semantic uncertainty estimation
    │
    ├─── uncertainty HIGH & loops < max ──► back to [Reasoner]  (re-route)
    │
    ▼
  [Critic]    — rubric-based self-audit
    │
    ▼
  [Synthesis] — final calibrated answer
    │
    ▼
  END

Conditional edges
-----------------
After [Reasoner]:
  - needs_reroute=True  AND  reasoning_loop < max_loops → "reason"   (loop)
  - otherwise                                            → "critique"

The LangGraph StateGraph is compiled once and reused across requests.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from argus.agents.planner import PlannerAgent
from argus.agents.retriever import RetrieverAgent
from argus.agents.reasoner import ReasonerAgent
from argus.agents.critic import CriticAgent
from argus.agents.synthesis import SynthesisAgent
from argus.core.memory import ArgusMemory, MemoryEntry
from argus.core.uncertainty import SemanticUncertaintyQuantifier
from argus.config import config

logger = logging.getLogger(__name__)


class ArgusState(TypedDict, total=False):
    """Shared state dict passed between all agents in the graph."""
    # Input
    query: str
    session_id: str

    # Memory lookups
    similar_episodes: list[dict]

    # Planner outputs
    plan: dict

    # Retriever outputs
    context_passages: list[dict]
    has_useful_context: bool

    # Reasoner outputs
    draft_response: str
    uncertainty_result: dict
    entropy_normalised: float
    uncertainty_level: str
    needs_reroute: bool
    reasoning_loop: int

    # Critic outputs
    critique_result: dict
    revised_response: str
    critique_passed: bool

    # Synthesis output
    final_answer: str

    # Accumulated trace (list of AgentStep dicts)
    trace: list[dict]


class ArgusOrchestrator:
    """
    Main entry point for running the ARGUS multi-agent pipeline.

    Usage
    -----
        orchestrator = ArgusOrchestrator()

        result = orchestrator.run("What is the quantum Hall effect?")
        print(result["final_answer"])
        print(result["trace"])      # Full reasoning trace
    """

    def __init__(self) -> None:
        self._memory = ArgusMemory()
        self._uq = SemanticUncertaintyQuantifier(
            embedding_model=config.memory.embedding_model
        )
        self._planner = PlannerAgent()
        self._retriever = RetrieverAgent(memory=self._memory)
        self._reasoner = ReasonerAgent(uncertainty_quantifier=self._uq)
        self._critic = CriticAgent()
        self._synthesis = SynthesisAgent()
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, query: str, session_id: str = "default") -> dict:
        """
        Execute the full ARGUS pipeline for a given query.

        Returns
        -------
        dict
            {
                "final_answer": str,
                "uncertainty_result": dict,
                "critique_result": dict,
                "trace": list[dict],
                "session_id": str,
            }
        """
        logger.info(f"[Orchestrator] Starting pipeline.  Query: {query[:80]}")

        # Pre-populate state with episodic memory lookups
        similar_episodes = self._memory.find_similar_episodes(query)

        initial_state: ArgusState = {
            "query": query,
            "session_id": session_id,
            "similar_episodes": similar_episodes,
            "reasoning_loop": 0,
            "trace": [],
        }

        final_state = self._graph.invoke(initial_state)

        # Persist episode to memory
        try:
            self._persist_episode(final_state)
        except Exception as e:
            logger.warning(f"[Orchestrator] Episode persistence failed: {e}")

        return {
            "final_answer": final_state.get("final_answer", ""),
            "uncertainty_result": final_state.get("uncertainty_result", {}),
            "critique_result": final_state.get("critique_result", {}),
            "plan": final_state.get("plan", {}),
            "trace": final_state.get("trace", []),
            "session_id": session_id,
        }

    @property
    def memory(self) -> ArgusMemory:
        return self._memory

    # ------------------------------------------------------------------
    # Private — graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        builder = StateGraph(ArgusState)

        # Register nodes (each node is a function: state → partial state)
        builder.add_node("plan",      self._planner.run)
        builder.add_node("retrieve",  self._retriever.run)
        builder.add_node("reason",    self._reasoner.run)
        builder.add_node("critique",  self._critic.run)
        builder.add_node("synthesise", self._synthesis.run)

        # Linear edges
        builder.set_entry_point("plan")
        builder.add_edge("plan", "retrieve")
        builder.add_edge("retrieve", "reason")

        # Conditional edge: re-route on high uncertainty
        builder.add_conditional_edges(
            "reason",
            self._route_after_reason,
            {
                "reason": "reason",        # Loop back
                "critique": "critique",    # Proceed
            },
        )

        builder.add_edge("critique", "synthesise")
        builder.add_edge("synthesise", END)

        return builder.compile()

    @staticmethod
    def _route_after_reason(state: ArgusState) -> str:
        """Routing function: return the name of the next node."""
        needs_reroute = state.get("needs_reroute", False)
        loop_count = state.get("reasoning_loop", 0)
        max_loops = config.max_reasoning_loops

        if needs_reroute and loop_count < max_loops:
            logger.info(
                f"[Router] High uncertainty (H={state.get('entropy_normalised', 0):.3f}) — "
                f"re-routing to Reasoner (loop {loop_count}/{max_loops})."
            )
            return "reason"
        else:
            if needs_reroute:
                logger.warning(
                    f"[Router] Still high uncertainty but max loops ({max_loops}) reached.  "
                    "Forcing critique."
                )
            return "critique"

    def _persist_episode(self, state: ArgusState) -> None:
        entry = MemoryEntry(
            query=state.get("query", ""),
            response=state.get("final_answer", ""),
            uncertainty_score=state.get("entropy_normalised", 0.0),
            uncertainty_level=state.get("uncertainty_level", "unknown"),
            reasoning_trace=state.get("trace", []),
        )
        self._memory.add_episode(entry)
        self._memory.add_turn("user", state.get("query", ""))
        self._memory.add_turn("assistant", state.get("final_answer", ""))
