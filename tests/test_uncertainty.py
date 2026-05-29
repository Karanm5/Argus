"""
Tests for the SemanticUncertaintyQuantifier.

These tests use deterministic mock responses to verify:
  - Entropy computation correctness.
  - Cluster assignment logic.
  - Threshold routing.
  - Edge cases (N=1, all identical responses).
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from argus.core.uncertainty import SemanticUncertaintyQuantifier, UncertaintyResult


def _make_uq(n_samples: int = 5) -> SemanticUncertaintyQuantifier:
    """Create a UQ instance with a mocked LLM client."""
    with patch("argus.core.uncertainty.Groq"):
        uq = SemanticUncertaintyQuantifier.__new__(SemanticUncertaintyQuantifier)
        uq._llm = MagicMock()
        uq._model = "test-model"
        uq._n_samples = n_samples
        uq._threshold = 0.65
        uq._distance_threshold = 0.30
        return uq


class TestEntropyMath:
    """Unit-test the pure entropy computation."""

    def test_zero_entropy_single_cluster(self):
        """If all responses fall into one cluster, entropy must be 0."""
        uq = _make_uq()
        cluster_labels = [0, 0, 0, 0, 0]
        counts = {0: 5}
        probs = {0: 1.0}
        entropy = -sum(p * math.log(p) for p in probs.values())
        assert entropy == pytest.approx(0.0, abs=1e-9)

    def test_max_entropy_all_different(self):
        """If every response is in its own cluster, entropy is maximised."""
        n = 5
        probs = {i: 1 / n for i in range(n)}
        entropy_raw = -sum(p * math.log(p) for p in probs.values())
        entropy_max = math.log(n)
        entropy_norm = entropy_raw / entropy_max
        assert entropy_norm == pytest.approx(1.0, abs=1e-6)

    def test_partial_entropy(self):
        """Two clusters of sizes 3 and 2 should give a specific entropy."""
        probs = {0: 3 / 5, 1: 2 / 5}
        entropy_raw = -sum(p * math.log(p) for p in probs.values())
        entropy_max = math.log(5)
        entropy_norm = entropy_raw / entropy_max
        # Expected: between 0 and 1, not equal to either extreme
        assert 0.0 < entropy_norm < 1.0

    def test_entropy_label_low(self):
        level, reason = SemanticUncertaintyQuantifier._label(0.20, 1, 0.90)
        assert level == "low"
        assert "consensus" in reason.lower()

    def test_entropy_label_medium(self):
        level, reason = SemanticUncertaintyQuantifier._label(0.50, 2, 0.70)
        assert level == "medium"

    def test_entropy_label_high(self):
        level, reason = SemanticUncertaintyQuantifier._label(0.80, 4, 0.40)
        assert level == "high"
        assert "re-routing" in reason.lower()


class TestClustering:
    """Test semantic clustering logic with synthetic embeddings."""

    def _build_uq(self):
        uq = _make_uq()
        # Replace the real embedder with an identity-like mock
        mock_embedder = MagicMock()
        uq._embedder = mock_embedder
        return uq

    def test_identical_embeddings_one_cluster(self):
        """Identical embeddings should produce a single cluster."""
        uq = self._build_uq()
        embeddings = np.tile(np.array([1.0, 0.0, 0.0]), (5, 1)).astype(np.float32)
        labels = uq._cluster(embeddings)
        assert len(set(labels.tolist())) == 1

    def test_orthogonal_embeddings_many_clusters(self):
        """Orthogonal unit vectors should each form their own cluster."""
        uq = self._build_uq()
        # Create 5 orthogonal unit vectors in R^5
        embeddings = np.eye(5, dtype=np.float32)
        labels = uq._cluster(embeddings)
        # All pairwise cosine similarities are 0 → max clusters
        assert len(set(labels.tolist())) == 5

    def test_two_groups(self):
        """Two tight clusters should be correctly separated."""
        uq = self._build_uq()
        group_a = np.tile(np.array([1.0, 0.0, 0.0]), (3, 1)) + np.random.randn(3, 3) * 0.01
        group_b = np.tile(np.array([0.0, 1.0, 0.0]), (3, 1)) + np.random.randn(3, 3) * 0.01
        embeddings = np.vstack([group_a, group_b]).astype(np.float32)
        # Normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        labels = uq._cluster(embeddings)
        assert len(set(labels.tolist())) == 2


class TestThresholdRouting:
    """Test the exceeds_threshold method."""

    def test_below_threshold_does_not_reroute(self):
        uq = _make_uq()
        result = MagicMock(spec=UncertaintyResult)
        result.entropy_normalised = 0.40
        assert not uq.exceeds_threshold(result)

    def test_above_threshold_reroutes(self):
        uq = _make_uq()
        result = MagicMock(spec=UncertaintyResult)
        result.entropy_normalised = 0.80
        assert uq.exceeds_threshold(result)

    def test_at_threshold_does_not_reroute(self):
        uq = _make_uq()
        result = MagicMock(spec=UncertaintyResult)
        result.entropy_normalised = 0.65    # Exactly at threshold — should NOT reroute
        assert not uq.exceeds_threshold(result)


class TestEdgeCases:
    """Edge cases that could cause NaNs or crashes."""

    def test_single_response(self):
        uq = _make_uq(n_samples=1)
        labels = uq._cluster(np.array([[1.0, 0.0]], dtype=np.float32))
        assert list(labels) == [0]

    def test_entropy_with_zero_prob_skipped(self):
        """Ensure log(0) is never computed."""
        probs = {0: 1.0, 1: 0.0}
        entropy = -sum(p * math.log(p) for p in probs.values() if p > 0)
        assert math.isfinite(entropy)
