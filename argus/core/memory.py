"""
Memory module
=============
ARGUS uses a three-tier memory architecture:

Tier 1 — Short-term (in-process Python dict)
    Conversation turns from the current session.  Cleared on restart.

Tier 2 — Episodic (ChromaDB collection: argus_episodes)
    Persisted Q&A pairs with metadata: timestamp, uncertainty score,
    reasoning trace hash.  Allows ARGUS to recall similar past queries
    and their confidence levels.

Tier 3 — Knowledge base (ChromaDB collection: argus_knowledge)
    User-ingested documents, chunked and embedded.  Drives the Retriever
    agent's RAG pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from argus.config import config

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    query: str
    response: str
    uncertainty_score: float
    uncertainty_level: str
    reasoning_trace: list[dict]
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        key = f"{self.query}:{self.timestamp}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


class ArgusMemory:
    """
    Manages all three memory tiers for ARGUS.

    Usage
    -----
        memory = ArgusMemory()

        # Add a document to the knowledge base
        memory.add_document("The Eiffel Tower is in Paris.", {"source": "wiki"})

        # Retrieve relevant context
        docs = memory.retrieve("Where is the Eiffel Tower?")

        # Store an episode
        memory.add_episode(entry)

        # Find similar past episodes
        similar = memory.find_similar_episodes("Eiffel Tower height", k=3)
    """

    def __init__(self) -> None:
        self._embedder = SentenceTransformer(config.memory.embedding_model, device="cpu")
        self._client = chromadb.PersistentClient(
            path=config.memory.chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )

        # Knowledge base collection
        self._kb = self._client.get_or_create_collection(
            name=config.memory.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Episodic memory collection
        self._episodes = self._client.get_or_create_collection(
            name="argus_episodes",
            metadata={"hnsw:space": "cosine"},
        )

        # In-process short-term memory (list of dicts)
        self._short_term: list[dict] = []

        logger.info(
            f"ArgusMemory initialised.  "
            f"KB docs: {self._kb.count()}, Episodes: {self._episodes.count()}"
        )

    # ------------------------------------------------------------------
    # Knowledge base (Tier 3)
    # ------------------------------------------------------------------

    def add_document(
        self,
        text: str,
        metadata: Optional[dict] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Chunk and embed a document, adding it to the knowledge base."""
        chunks = self._chunk(text)
        ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = doc_id or hashlib.sha256(chunk.encode()).hexdigest()[:16]
            chunk_id = f"{chunk_id}_{i}"
            embedding = self._embed([chunk])[0]
            self._kb.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[metadata or {}],
            )
            ids.append(chunk_id)
        logger.debug(f"Added {len(chunks)} chunks to knowledge base.")
        return ids[0] if len(ids) == 1 else ids

    def retrieve(self, query: str, k: Optional[int] = None) -> list[dict]:
        """Retrieve the top-k most relevant knowledge base chunks."""
        k = k or config.memory.top_k_retrieve
        if self._kb.count() == 0:
            return []

        embedding = self._embed([query])[0]
        results = self._kb.query(
            query_embeddings=[embedding],
            n_results=min(k, self._kb.count()),
            include=["documents", "distances", "metadatas"],
        )

        docs = []
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            similarity = 1.0 - dist  # Cosine distance → similarity
            if similarity >= config.memory.similarity_threshold:
                docs.append({
                    "content": doc,
                    "similarity": round(similarity, 4),
                    "metadata": meta,
                })

        return docs

    # ------------------------------------------------------------------
    # Episodic memory (Tier 2)
    # ------------------------------------------------------------------

    def add_episode(self, entry: MemoryEntry) -> None:
        """Persist a completed reasoning episode."""
        embedding = self._embed([entry.query])[0]
        self._episodes.upsert(
            ids=[entry.id],
            embeddings=[embedding],
            documents=[entry.query],
            metadatas=[{
                "response": entry.response[:500],       # Truncate for storage
                "uncertainty_score": entry.uncertainty_score,
                "uncertainty_level": entry.uncertainty_level,
                "trace_len": len(entry.reasoning_trace),
                "timestamp": entry.timestamp,
                **entry.metadata,
            }],
        )

    def find_similar_episodes(self, query: str, k: int = 3) -> list[dict]:
        """Find past episodes with semantically similar queries."""
        if self._episodes.count() == 0:
            return []
        embedding = self._embed([query])[0]
        results = self._episodes.query(
            query_embeddings=[embedding],
            n_results=min(k, self._episodes.count()),
            include=["documents", "distances", "metadatas"],
        )
        return [
            {
                "past_query": doc,
                "similarity": round(1.0 - dist, 4),
                "metadata": meta,
            }
            for doc, dist, meta in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            )
        ]

    # ------------------------------------------------------------------
    # Short-term memory (Tier 1)
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str) -> None:
        self._short_term.append({"role": role, "content": content})
        if len(self._short_term) > 20:
            self._short_term = self._short_term[-20:]

    def get_short_term(self) -> list[dict]:
        return list(self._short_term)

    def clear_short_term(self) -> None:
        self._short_term.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    @staticmethod
    def _chunk(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
        """Simple character-level chunking with overlap."""
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks
