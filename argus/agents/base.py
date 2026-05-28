"""
Base agent
==========
All ARGUS agents inherit from BaseAgent, which provides:
  - A unified interface for making traced LLM calls via Anthropic SDK.
  - Structured step logging so the orchestrator can build a full
    reasoning trace.
  - Exponential backoff on API rate limit errors.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import anthropic

from argus.config import config

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """One logged step in the reasoning trace."""
    agent_name: str
    step_type: str          # "think" | "retrieve" | "critique" | "synthesise"
    input_summary: str      # Short description of what was sent
    output: str             # Full raw output
    latency_ms: float
    token_count: int = 0
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "agent": self.agent_name,
            "type": self.step_type,
            "input_summary": self.input_summary,
            "output": self.output,
            "latency_ms": round(self.latency_ms, 1),
            "tokens": self.token_count,
            **self.metadata,
        }


class BaseAgent(ABC):
    """
    Abstract base for all ARGUS agents.

    Subclasses must implement `run(state) -> dict`, which takes the
    current orchestrator state dict and returns a dict of updates to
    merge back into the state.
    """

    NAME: str = "base"
    STEP_TYPE: str = "think"

    def __init__(self) -> None:
        self._llm = anthropic.Anthropic(api_key=config.llm.api_key)
        self._model = config.llm.model
        self._steps: list[AgentStep] = []

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, state: dict) -> dict:
        """
        Execute this agent's role.

        Parameters
        ----------
        state : dict
            Current shared orchestrator state.  Keys defined in
            argus.core.orchestrator.ArgusState.

        Returns
        -------
        dict
            Partial state updates to merge.
        """

    # ------------------------------------------------------------------
    # Protected helpers — available to all subclasses
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        step_input_summary: str = "",
    ) -> AgentStep:
        """Make a traced, retried LLM call and return an AgentStep."""
        temperature = temperature if temperature is not None else config.llm.temperature
        max_tokens = max_tokens or config.llm.max_tokens

        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        if system:
            kwargs["system"] = system

        t0 = time.perf_counter()
        response = self._llm_with_retry(kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        text = response.content[0].text.strip()
        token_count = response.usage.input_tokens + response.usage.output_tokens

        step = AgentStep(
            agent_name=self.NAME,
            step_type=self.STEP_TYPE,
            input_summary=step_input_summary or messages[-1]["content"][:120],
            output=text,
            latency_ms=latency_ms,
            token_count=token_count,
        )
        self._steps.append(step)
        logger.debug(f"[{self.NAME}] {latency_ms:.0f}ms  {token_count} tokens")
        return step

    def _llm_with_retry(self, kwargs: dict, max_retries: int = 3) -> Any:
        """Retry with exponential backoff on rate-limit or overload errors."""
        for attempt in range(max_retries):
            try:
                return self._llm.messages.create(**kwargs)
            except anthropic.RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning(f"[{self.NAME}] Rate limited.  Retrying in {wait}s…")
                time.sleep(wait)
            except anthropic.APIStatusError as e:
                if e.status_code == 529 and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"[{self.NAME}] Overloaded.  Retrying in {wait}s…")
                    time.sleep(wait)
                else:
                    raise

    def get_trace(self) -> list[dict]:
        return [s.as_dict() for s in self._steps]

    def clear_trace(self) -> None:
        self._steps.clear()
