"""
Semantic Entropy-Based Uncertainty Quantification
==================================================
Implements the semantic uncertainty estimator described in:

    Kuhn et al. (2023) "Semantic Uncertainty: Linguistic Invariances
    for Uncertainty Estimation in Natural Language Generation."
    ICLR 2023.  https://arxiv.org/abs/2302.09664

Key insight: purely token-level uncertainty (e.g. perplexity) conflates
linguistically different but semantically equivalent responses as high-
uncertainty, when they are in fact the same answer.  Semantic entropy
clusters responses by meaning first, then computes entropy over the
cluster probabilities — giving a true measure of *epistemic* uncertainty.

Algorithm
---------
1. Sample N responses from the LLM at elevated temperature (0.9).
2. Embed each response with a sentence transformer.
3. Build a cosine-similarity graph; apply agglomerative clustering.
4. Assign each response to a semantic cluster c_i.
5. Compute entropy:  H = -sum_i p(c_i) * log(p(c_i))
6. Normalise to [0, 1] with H_max = log(N).
7. Return raw entropy, normalised entropy, cluster assignment, and
   a three-level label (low / medium / high).
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from argus.config import config

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyResult:
    """Full output of a single uncertainty estimation run."""
    query: str
    responses: list[str]
    embeddings: np.ndarray          # shape (N, D)
    similarity_matrix: np.ndarray  # shape (N, N)
    cluster_labels: list[int]
    n_clusters: int
    cluster_probabilities: dict[int, float]

    # Entropy values
    entropy_raw: float              # Shannon entropy in nats
    entropy_normalised: float       # Scaled to [0, 1]; 1.0 = max disagreement
    entropy_max: float              # log(N) — theoretical max

    # Human-readable summary
    uncertainty_level: str          # "low" | "medium" | "high"
    reasoning: str                  # One-line explanation of the score

    # Derived stats
    majority_response: str          # Response from the largest cluster
    consensus_ratio: float          # Size of majority cluster / N

    @property
    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "responses": self.responses,
            "n_clusters": self.n_clusters,
            "cluster_probabilities": self.cluster_probabilities,
            "entropy_raw": round(self.entropy_raw, 4),
            "entropy_normalised": round(self.entropy_normalised, 4),
            "entropy_max": round(self.entropy_max, 4),
            "uncertainty_level": self.uncertainty_level,
            "reasoning": self.reasoning,
            "majority_response": self.majority_response,
            "consensus_ratio": round(self.consensus_ratio, 4),
            "cluster_labels": self.cluster_labels,
        }


class SemanticUncertaintyQuantifier:
    """
    Estimates epistemic uncertainty for a given query by sampling the LLM
    N times and computing semantic entropy over the response distribution.

    Parameters
    ----------
    embedding_model : str
        SentenceTransformer model name for embedding responses.
    clustering_distance_threshold : float
        Distance threshold for agglomerative clustering.  Lower values
        create more clusters (higher granularity).  Default 0.3 works
        well for most instruction-following tasks.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        clustering_distance_threshold: float = 0.30,
    ) -> None:
        logger.info(f"Loading embedding model: {embedding_model}")
        self._embedder = SentenceTransformer(embedding_model)
        self._distance_threshold = clustering_distance_threshold
        self._llm = Groq(api_key=config.llm.api_key)
        self._model = config.llm.model
        self._n_samples = config.llm.n_uncertainty_samples
        self._threshold = config.llm.uncertainty_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        n: Optional[int] = None,
    ) -> UncertaintyResult:
        """
        Full uncertainty estimation pipeline.

        Parameters
        ----------
        query : str
            The user's question or task.
        system_prompt : str, optional
            System context for the LLM.
        n : int, optional
            Override the default number of samples.

        Returns
        -------
        UncertaintyResult
            Complete result including entropy, clusters, and per-response
            embeddings.
        """
        n = n or self._n_samples
        logger.debug(f"Sampling {n} responses for uncertainty estimation")

        responses = self._sample_responses(query, system_prompt, n)
        embeddings = self._embed(responses)
        similarity_matrix = cosine_similarity(embeddings)
        cluster_labels = self._cluster(embeddings)
        n_clusters = len(set(cluster_labels))

        # Cluster probabilities p(c_i) = count(c_i) / N
        counts: dict[int, int] = {}
        for label in cluster_labels:
            counts[label] = counts.get(label, 0) + 1
        cluster_probs = {c: cnt / n for c, cnt in counts.items()}

        # Shannon entropy H = -sum p(c) * ln(p(c))
        entropy_raw = -sum(p * math.log(p) for p in cluster_probs.values() if p > 0)
        entropy_max = math.log(n) if n > 1 else 1.0
        entropy_normalised = min(entropy_raw / entropy_max, 1.0)

        # Majority response: response from the most common cluster
        majority_cluster = max(counts, key=counts.__getitem__)
        majority_idx = next(i for i, c in enumerate(cluster_labels) if c == majority_cluster)
        majority_response = responses[majority_idx]
        consensus_ratio = counts[majority_cluster] / n

        # Label
        uncertainty_level, reasoning = self._label(entropy_normalised, n_clusters, consensus_ratio)

        return UncertaintyResult(
            query=query,
            responses=responses,
            embeddings=embeddings,
            similarity_matrix=similarity_matrix,
            cluster_labels=list(cluster_labels),
            n_clusters=n_clusters,
            cluster_probabilities=cluster_probs,
            entropy_raw=entropy_raw,
            entropy_normalised=entropy_normalised,
            entropy_max=entropy_max,
            uncertainty_level=uncertainty_level,
            reasoning=reasoning,
            majority_response=majority_response,
            consensus_ratio=consensus_ratio,
        )

    def exceeds_threshold(self, result: UncertaintyResult) -> bool:
        """Returns True if the result's normalised entropy exceeds the
        configured routing threshold — i.e., re-routing is warranted."""
        return result.entropy_normalised > self._threshold

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_responses(
        self,
        query: str,
        system_prompt: Optional[str],
        n: int,
    ) -> list[str]:
        """Sample N independent responses from the LLM."""
        responses = []
        msgs = [{"role": "user", "content": query}]
        kwargs = dict(
            model=self._model,
            max_tokens=512,
            temperature=config.llm.sampling_temperature,
            messages=msgs,
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        for i in range(n):
            try:
                response = self._llm.messages.create(**kwargs)
                text = response.content[0].text.strip()
                responses.append(text)
                logger.debug(f"  Sample {i+1}/{n}: {text[:80]}…")
            except anthropic.APIError as e:
                logger.warning(f"  Sample {i+1} failed: {e}.  Using placeholder.")
                responses.append("")

        return responses

    def _embed(self, responses: list[str]) -> np.ndarray:
        """Return L2-normalised sentence embeddings."""
        vecs = self._embedder.encode(responses, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs)

    def _cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Agglomerative clustering on cosine *distance* matrix.
        distance = 1 - cosine_similarity.
        """
        n = len(embeddings)
        if n == 1:
            return np.array([0])

        distance_matrix = 1.0 - cosine_similarity(embeddings)
        distance_matrix = np.clip(distance_matrix, 0, None)  # Numerical safety

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=self._distance_threshold,
        )
        labels = clustering.fit_predict(distance_matrix)
        return labels

    @staticmethod
    def _label(
        entropy_norm: float,
        n_clusters: int,
        consensus_ratio: float,
    ) -> tuple[str, str]:
        """Assign a human-readable uncertainty level and one-line reason."""
        if entropy_norm < 0.30:
            level = "low"
            reason = (
                f"All {n_clusters} cluster(s) show strong consensus "
                f"(majority ratio {consensus_ratio:.0%}).  "
                "Proceeding directly to synthesis."
            )
        elif entropy_norm < 0.65:
            level = "medium"
            reason = (
                f"{n_clusters} semantic clusters detected.  Moderate disagreement "
                f"(H={entropy_norm:.2f}).  Critic review recommended."
            )
        else:
            level = "high"
            reason = (
                f"High semantic entropy (H={entropy_norm:.2f}) across {n_clusters} clusters.  "
                "Re-routing to extended reasoning loop."
            )
        return level, reason
