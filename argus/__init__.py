"""
ARGUS — Adaptive Reasoning with Guided Uncertainty Sampling
===========================================================
A self-auditing multi-agent LLM reasoning system with semantic
entropy-based uncertainty quantification.

Key innovation: ARGUS estimates epistemic uncertainty by sampling N
responses, clustering them in embedding space, and computing Shannon
entropy over the cluster distribution. If entropy exceeds a threshold
theta, the orchestrator re-routes back to the reasoning loop.

Reference:
    Kuhn et al. (2023) "Semantic Uncertainty: Linguistic Invariances
    for Uncertainty Estimation in Natural Language Generation"
    https://arxiv.org/abs/2302.09664
"""

__version__ = "0.1.0"
__author__ = "Karan Meena"
