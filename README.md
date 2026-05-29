# 👁 ARGUS — Adaptive Reasoning with Guided Uncertainty Sampling

[![CI](https://github.com/Karanm5/argus/actions/workflows/ci.yml/badge.svg)](https://github.com/Karanm5/argus/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Model](https://img.shields.io/badge/LLM-llama--3.3--70b-orange)

**ARGUS** is a self-auditing multi-agent LLM reasoning system that estimates its own epistemic uncertainty using *semantic entropy* — and re-routes to a deeper reasoning loop when that uncertainty is too high.

> **Key insight:** Most LLM applications call the API once and return the first response. ARGUS samples its own outputs N times, clusters them by semantic meaning, and computes Shannon entropy over the cluster distribution. High entropy means the model is genuinely uncertain — not just phrasing things differently — and triggers a multi-step remediation loop.

---

## 🧠 Why this is interesting

Standard approaches to LLM uncertainty either:
- Use token-level log-probabilities (conflates paraphrase with genuine uncertainty), or
- Do nothing at all.

ARGUS implements **semantic entropy** (Kuhn et al., ICLR 2023), adapted for production multi-agent systems:

```
Sample N responses → Embed → Cluster semantically → Compute H = -Σ p(c) log p(c)

H < 0.30  →  Low uncertainty   →  Proceed to Critic
H = 0.30–0.65  →  Medium       →  Critic review recommended
H > 0.65  →  High uncertainty  →  Re-route to extended reasoning loop
```

This gives a calibrated, semantically-grounded confidence score on every query — surfaced to the user alongside the reasoning trace.

---

## 🏗 Architecture

```
User Query
    │
    ▼
[Planner]          Sub-task decomposition, complexity estimation
    │
    ▼
[Retriever]        ChromaDB RAG + LLM reranking (conditional)
    │
    ▼
[Reasoner] ──────── Semantic Uncertainty Engine ──────────────┐
    │                N samples → embeddings → clusters → H     │
    │                                                           │
    ├─── H > θ AND loops < max ─────────────────────────────────┘
    │                                 (re-route)
    ▼
[Critic]           Six-dimension rubric audit
    │              (coherence, grounding, completeness,
    │               calibration, relevance, conciseness)
    ▼
[Synthesis]        Calibrated final answer
    │
    ▼
FastAPI + Streamlit UI
```

Each agent is a separate class inheriting from `BaseAgent`, orchestrated by a **LangGraph** `StateGraph` with conditional edges. The full reasoning trace (every agent step, latency, token count, uncertainty score) is logged and surfaced in the UI.

---

## 🚀 Quick start

### Prerequisites
- Python 3.10+
- - Groq API key (free at https://console.groq.com)

### 1. Clone and install

```bash
git clone https://github.com/Karanm5/argus.git
cd argus
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_groq_key
```

### 3. Run the API

```bash
uvicorn argus.api.main:app --reload
# API docs at http://localhost:8000/docs
```

### 4. Run the UI

```bash
streamlit run frontend/app.py
# Opens at http://localhost:8501
```

### 5. Or use Docker Compose

```bash
GROQ_API_KEY=your_key docker-compose up --build
# API: http://localhost:8000
# UI:  http://localhost:8501
```

---

## 📡 API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Full synchronous pipeline run |
| `POST` | `/query/stream` | SSE streaming of reasoning steps |
| `POST` | `/ingest` | Add document to knowledge base |
| `GET` | `/episodes` | Retrieve past reasoning episodes |
| `GET` | `/health` | Health check |

**Example request:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the limitations of transformer attention?"}'
```

**Example response (abbreviated):**

```json
{
  "final_answer": "Transformer attention has three core limitations: ...",
  "uncertainty_result": {
    "entropy_normalised": 0.312,
    "uncertainty_level": "medium",
    "n_clusters": 2,
    "consensus_ratio": 0.6,
    "reasoning": "2 semantic clusters detected. Moderate disagreement..."
  },
  "critique_result": {
    "mean_score": 0.82,
    "scores": {
      "logical_coherence": 0.90,
      "factual_grounding": 0.85,
      "completeness": 0.78,
      "calibration": 0.80,
      "relevance": 0.88,
      "conciseness": 0.72
    },
    "issues": ["Could expand on quadratic complexity trade-offs."]
  },
  "trace": [
    {"agent": "Planner", "type": "plan", "latency_ms": 823, "tokens": 312, ...},
    {"agent": "Reasoner", "type": "reason", "latency_ms": 1240, ...},
    {"agent": "Reasoner", "type": "uncertainty_estimation", "entropy_normalised": 0.312, ...},
    {"agent": "Critic", "type": "critique", "mean_score": 0.82, ...},
    {"agent": "Synthesis", "type": "synthesise", "is_final": true, ...}
  ]
}
```

---

## 🔬 Uncertainty quantification in depth

The `SemanticUncertaintyQuantifier` implements the following pipeline:

```python
from argus.core.uncertainty import SemanticUncertaintyQuantifier

uq = SemanticUncertaintyQuantifier(
    embedding_model="all-MiniLM-L6-v2",
    clustering_distance_threshold=0.30,
)

result = uq.estimate(
    query="Is Schrödinger's cat actually in a superposition?",
    n=5,
)

print(result.entropy_normalised)    # 0.0 to 1.0
print(result.uncertainty_level)     # "low" | "medium" | "high"
print(result.n_clusters)            # Number of distinct semantic clusters
print(result.consensus_ratio)       # Fraction of responses in the majority cluster
```

**Why agglomerative clustering over k-means?**
The number of clusters is not known in advance. Agglomerative clustering with a distance threshold discovers the cluster count automatically — which is essential when we don't know if responses form 1, 2, or N distinct semantic groups.

---

## 🗂 Project structure

```
argus/
├── argus/
│   ├── config.py                   # All settings via env vars
│   ├── agents/
│   │   ├── base.py                 # BaseAgent with tracing + retry logic
│   │   ├── planner.py              # Sub-task decomposition
│   │   ├── retriever.py            # ChromaDB RAG + LLM reranking
│   │   ├── reasoner.py             # Chain-of-thought + uncertainty estimation
│   │   ├── critic.py               # Six-dimension rubric audit
│   │   └── synthesis.py            # Calibrated final answer
│   ├── core/
│   │   ├── uncertainty.py          # ← Core novel module: semantic entropy
│   │   ├── memory.py               # Three-tier memory (short-term/episodic/KB)
│   │   └── orchestrator.py         # LangGraph StateGraph with conditional routing
│   └── api/
│       └── main.py                 # FastAPI: /query, /query/stream, /ingest
├── frontend/
│   └── app.py                      # Streamlit dashboard
├── tests/
│   ├── test_uncertainty.py         # Unit tests for entropy math + clustering
│   └── test_agents.py              # Agent integration tests
├── .github/workflows/ci.yml        # GitHub Actions CI (lint + test + docker)
├── Dockerfile                      # Multi-stage Docker build
├── docker-compose.yml              # API + frontend together
└── README.md
```

---

## 🧪 Running tests

```bash
pytest tests/ -v --cov=argus --cov-report=term-missing
```

Tests use mocked LLM calls — no API key required to run the test suite.

---

## 📦 Deployment

### Streamlit Community Cloud
1. Go to https://share.streamlit.io
2. Connect your GitHub repo `Karanm5/Argus`
3. Set main file path to `app.py`
4. Add secret: `GROQ_API_KEY = "your_key"`
5. Deploy — live in ~3 minutes


### Railway / Render

```bash
# Set env vars: ANTHROPIC_API_KEY, PORT
# Start command: uvicorn argus.api.main:app --host 0.0.0.0 --port $PORT
```

---

## 📖 References

1. **Kuhn, L., Gal, Y., & Farquhar, S.** (2023). *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* ICLR 2023. [arXiv:2302.09664](https://arxiv.org/abs/2302.09664)

2. **Bai, Y., et al.** (2022). *Constitutional AI: Harmlessness from AI Feedback.* Anthropic. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)

3. **LangGraph Documentation.** LangChain, Inc. https://langchain-ai.github.io/langgraph/

---

## 👤 Author

**Karan** — MSc Data Analytics (Distinction), Aston University  
[LinkedIn](https://linkedin.com/in/karan-th) · [GitHub](https://github.com/Karanm5)  
meena.karan9k@gmail.com

---

## License

MIT — see [LICENSE](LICENSE).
