"""
ARGUS FastAPI Application
=========================
Exposes the ARGUS pipeline via:
  POST /query            — Synchronous full pipeline run
  POST /query/stream     — Server-Sent Events streaming of reasoning steps
  POST /ingest           — Add documents to the knowledge base
  GET  /health           — Health check
  GET  /episodes         — Recent episodic memory entries
  DELETE /session/{id}   — Clear short-term memory for a session
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from argus.core.orchestrator import ArgusOrchestrator
from argus.config import config

logging.basicConfig(level=config.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ARGUS API",
    description=(
        "Adaptive Reasoning with Guided Uncertainty Sampling — "
        "a self-auditing multi-agent LLM reasoning system."
    ),
    version="0.1.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate once — shared across requests
_orchestrator: ArgusOrchestrator | None = None


def get_orchestrator() -> ArgusOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        logger.info("Initialising ArgusOrchestrator…")
        _orchestrator = ArgusOrchestrator()
    return _orchestrator


# ------------------------------------------------------------------
# Pydantic schemas
# ------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class QueryResponse(BaseModel):
    session_id: str
    final_answer: str
    uncertainty_result: dict
    critique_result: dict
    plan: dict
    trace: list[dict]

class IngestRequest(BaseModel):
    text: str = Field(..., min_length=10)
    source: str = Field(default="user_upload")
    doc_id: str | None = None

class HealthResponse(BaseModel):
    status: str
    version: str
    model: str


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse(
        status="ok",
        version="0.1.0",
        model=config.llm.model,
    )


@app.post("/query", response_model=QueryResponse, tags=["Reasoning"])
def run_query(req: QueryRequest):
    """
    Run the full ARGUS multi-agent pipeline synchronously.

    Returns the final answer, reasoning trace, uncertainty estimate,
    and critique result.
    """
    try:
        orch = get_orchestrator()
        result = orch.run(query=req.query, session_id=req.session_id)
        return QueryResponse(
            session_id=result["session_id"],
            final_answer=result["final_answer"],
            uncertainty_result=result.get("uncertainty_result", {}),
            critique_result=result.get("critique_result", {}),
            plan=result.get("plan", {}),
            trace=result.get("trace", []),
        )
    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream", tags=["Reasoning"])
async def run_query_stream(req: QueryRequest):
    """
    Stream reasoning steps as Server-Sent Events.

    Each event is a JSON object with:
      { "type": "step" | "final" | "error", "data": {...} }
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            orch = get_orchestrator()

            # Run pipeline in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: orch.run(query=req.query, session_id=req.session_id),
            )

            # Stream each trace step
            for step in result.get("trace", []):
                yield f"data: {json.dumps({'type': 'step', 'data': step})}\n\n"
                await asyncio.sleep(0)   # Yield to event loop

            # Final event
            final_payload = {
                "type": "final",
                "data": {
                    "final_answer": result["final_answer"],
                    "uncertainty_result": result.get("uncertainty_result", {}),
                    "session_id": result["session_id"],
                },
            }
            yield f"data: {json.dumps(final_payload)}\n\n"

        except Exception as e:
            logger.exception(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': str(e)}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/ingest", tags=["Knowledge Base"])
def ingest_document(req: IngestRequest):
    """Add a document to ARGUS's persistent knowledge base."""
    try:
        orch = get_orchestrator()
        doc_id = orch.memory.add_document(
            text=req.text,
            metadata={"source": req.source},
            doc_id=req.doc_id,
        )
        return {"status": "ingested", "doc_id": doc_id}
    except Exception as e:
        logger.exception(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/episodes", tags=["Memory"])
def get_episodes(query: str = "", k: int = 5):
    """Retrieve similar past reasoning episodes."""
    orch = get_orchestrator()
    if query:
        episodes = orch.memory.find_similar_episodes(query, k=k)
    else:
        # Return recent episodes (empty query returns the N most recent via similarity to "")
        episodes = orch.memory.find_similar_episodes("the", k=k)
    return {"episodes": episodes}


@app.delete("/session/{session_id}", tags=["Memory"])
def clear_session(session_id: str):
    """Clear short-term memory for a session."""
    orch = get_orchestrator()
    orch.memory.clear_short_term()
    return {"status": "cleared", "session_id": session_id}
