"""
Configuration for ARGUS.
All runtime settings are loaded from environment variables or .env file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    api_key: str = field(default_factory=lambda: os.environ["ANTHROPIC_API_KEY"])
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2048
    temperature: float = 0.7          # Base temperature for reasoning
    sampling_temperature: float = 0.9  # Higher temp for uncertainty sampling
    n_uncertainty_samples: int = 5     # Number of samples for entropy estimation
    uncertainty_threshold: float = 0.65  # Shannon entropy threshold (nats) to trigger re-route


@dataclass
class MemoryConfig:
    chroma_path: str = field(default_factory=lambda: os.getenv("CHROMA_PATH", "./data/chroma"))
    embedding_model: str = "all-MiniLM-L6-v2"   # SentenceTransformer model
    collection_name: str = "argus_knowledge"
    top_k_retrieve: int = 5
    similarity_threshold: float = 0.35


@dataclass
class APIConfig:
    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    reload: bool = field(default_factory=lambda: os.getenv("ENV", "dev") == "dev")
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class ArgusConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    api: APIConfig = field(default_factory=APIConfig)
    max_reasoning_loops: int = 3    # Hard cap on re-routing before forced synthesis
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


# Singleton config — import this everywhere
config = ArgusConfig()
