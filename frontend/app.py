"""
ARGUS — Streamlit Frontend
==========================
Interactive demo UI showing:
  - Query input
  - Real-time reasoning trace (Planner → Retriever → Reasoner → Critic → Synthesis)
  - Semantic entropy gauge
  - Critique rubric radar chart
  - Knowledge base document upload
"""

from __future__ import annotations

import json
import os
import sys

import streamlit as st
import plotly.graph_objects as go
import requests

# Fallback: if the API isn't running, we can import and run locally
API_BASE = os.getenv("ARGUS_API_URL", "http://localhost:8000")

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="ARGUS — Reasoning System",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
.argus-header { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0; }
.argus-sub    { font-size: 0.95rem; color: #6b7280; margin-top: 0.2rem; margin-bottom: 2rem; }
.trace-card   { background: #f9fafb; border-left: 3px solid #6366f1;
                padding: 0.75rem 1rem; margin: 0.4rem 0; border-radius: 6px; font-size: 0.87rem; }
.agent-badge  { display: inline-block; padding: 2px 8px; border-radius: 12px;
                font-size: 0.75rem; font-weight: 600; margin-right: 6px; }
.badge-plan   { background: #dbeafe; color: #1d4ed8; }
.badge-retrieve { background: #dcfce7; color: #15803d; }
.badge-reason { background: #ede9fe; color: #6d28d9; }
.badge-critique { background: #fef3c7; color: #b45309; }
.badge-synthesise { background: #fce7f3; color: #9d174d; }
.badge-uncertainty_estimation { background: #fee2e2; color: #991b1b; }
.entropy-low    { color: #15803d; font-weight: 600; }
.entropy-medium { color: #b45309; font-weight: 600; }
.entropy-high   { color: #991b1b; font-weight: 600; }
.metric-box { text-align: center; padding: 1rem; background: #f1f5f9;
              border-radius: 10px; margin: 0.2rem; }
.metric-val { font-size: 2rem; font-weight: 700; line-height: 1; }
.metric-lbl { font-size: 0.78rem; color: #6b7280; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────
st.markdown('<p class="argus-header">👁 ARGUS</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="argus-sub">Adaptive Reasoning with Guided Uncertainty Sampling — '
    'a self-auditing multi-agent LLM system</p>',
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("API URL", value=API_BASE)
    session_id = st.text_input("Session ID", value="demo-session")

    st.divider()
    st.header("📚 Knowledge Base")
    uploaded = st.file_uploader("Upload a text document", type=["txt", "md"])
    if uploaded and st.button("Ingest document"):
        text = uploaded.read().decode("utf-8")
        try:
            resp = requests.post(
                f"{api_url}/ingest",
                json={"text": text, "source": uploaded.name},
                timeout=30,
            )
            if resp.ok:
                st.success(f"Ingested {len(text)} characters.")
            else:
                st.error(f"Ingest failed: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API.  Is it running?")

    st.divider()
    st.header("🔬 About")
    st.markdown("""
**Key innovation:** Semantic entropy-based uncertainty quantification.
ARGUS samples its own responses N times, clusters them semantically,
and computes Shannon entropy — routing to a deeper reasoning loop
when epistemic uncertainty is high.

**Ref:** Kuhn et al., *Semantic Uncertainty*, ICLR 2023.
    """)


# ── Main panel ───────────────────────────────────────────────────────
query = st.text_area(
    "Enter your query",
    placeholder="e.g. What are the limitations of transformer attention mechanisms?",
    height=120,
)

col_run, col_clear = st.columns([1, 5])
with col_run:
    run_clicked = st.button("▶ Run ARGUS", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=False):
        for k in ["result", "error"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

if run_clicked and query.strip():
    with st.spinner("Running multi-agent pipeline…"):
        try:
            resp = requests.post(
                f"{api_url}/query",
                json={"query": query.strip(), "session_id": session_id},
                timeout=180,
            )
            if resp.ok:
                st.session_state["result"] = resp.json()
                st.session_state.pop("error", None)
            else:
                st.session_state["error"] = f"API error {resp.status_code}: {resp.text[:500]}"
        except requests.exceptions.ConnectionError:
            st.session_state["error"] = (
                "Cannot connect to ARGUS API at "
                f"{api_url}.  Run `uvicorn argus.api.main:app` first."
            )

# ── Results ──────────────────────────────────────────────────────────
if "error" in st.session_state:
    st.error(st.session_state["error"])

if "result" in st.session_state:
    result = st.session_state["result"]
    unc = result.get("uncertainty_result", {})
    crit = result.get("critique_result", {})
    trace = result.get("trace", [])
    plan = result.get("plan", {})

    # ── Answer box ──
    st.subheader("📝 Final answer")
    st.info(result.get("final_answer", ""))

    # ── Key metrics ──
    st.subheader("📊 Pipeline metrics")
    m1, m2, m3, m4 = st.columns(4)

    entropy_norm = unc.get("entropy_normalised", 0.0)
    unc_level = unc.get("uncertainty_level", "—")
    entropy_class = f"entropy-{unc_level}"

    with m1:
        st.markdown(
            f'<div class="metric-box"><div class="metric-val">'
            f'{entropy_norm:.3f}</div>'
            f'<div class="metric-lbl">Semantic entropy (norm)</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-box"><div class="metric-val {entropy_class}">'
            f'{unc_level.upper()}</div>'
            f'<div class="metric-lbl">Uncertainty level</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        mean_score = crit.get("mean_score", 0.0)
        colour = "#15803d" if mean_score >= 0.75 else "#b45309"
        st.markdown(
            f'<div class="metric-box"><div class="metric-val" style="color:{colour}">'
            f'{mean_score:.2f}</div>'
            f'<div class="metric-lbl">Critic mean score</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        n_steps = len(trace)
        st.markdown(
            f'<div class="metric-box"><div class="metric-val">{n_steps}</div>'
            f'<div class="metric-lbl">Reasoning steps</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Charts side-by-side ──
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("🔭 Uncertainty distribution")
        cluster_probs = unc.get("cluster_probabilities", {})
        if cluster_probs:
            fig = go.Figure(go.Bar(
                x=[f"Cluster {k}" for k in cluster_probs],
                y=list(cluster_probs.values()),
                marker_color=["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd"][:len(cluster_probs)],
            ))
            fig.update_layout(
                title=f"Semantic clusters (H={entropy_norm:.3f})",
                xaxis_title="Semantic cluster",
                yaxis_title="Probability p(c)",
                yaxis_range=[0, 1],
                height=300,
                margin=dict(l=40, r=10, t=40, b=40),
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cluster data available.")

    with chart_col2:
        st.subheader("🎯 Critic rubric scores")
        scores = crit.get("scores", {})
        if scores:
            categories = list(scores.keys())
            values = [scores[k] for k in categories]
            categories_display = [c.replace("_", " ").title() for c in categories]

            fig = go.Figure(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories_display + [categories_display[0]],
                fill="toself",
                fillcolor="rgba(99,102,241,0.15)",
                line=dict(color="#6366f1", width=2),
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Quality rubric (0–1 per dimension)",
                height=300,
                margin=dict(l=40, r=40, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No critique scores available.")

    st.divider()

    # ── Reasoning trace ──
    st.subheader("🧠 Reasoning trace")

    agent_colors = {
        "Planner": "plan",
        "Retriever": "retrieve",
        "Reasoner": "reason",
        "Critic": "critique",
        "Synthesis": "synthesise",
    }

    for i, step in enumerate(trace):
        agent = step.get("agent", "?")
        step_type = step.get("type", "think")
        badge_class = agent_colors.get(agent, step_type)
        latency = step.get("latency_ms", 0)
        tokens = step.get("tokens", 0)
        output = step.get("output", "")
        input_summary = step.get("input_summary", "")

        with st.expander(
            f"{'🔴' if step_type == 'uncertainty_estimation' else '⚡'} "
            f"Step {i+1} — {agent} ({step_type})  "
            f"[{latency:.0f}ms · {tokens} tokens]",
            expanded=(i == len(trace) - 1),
        ):
            st.markdown(
                f'<span class="agent-badge badge-{badge_class.replace("_","-")}">{agent}</span>'
                f'<small style="color:#6b7280">{input_summary}</small>',
                unsafe_allow_html=True,
            )

            # Special rendering for uncertainty estimation steps
            if step_type == "uncertainty_estimation":
                ecols = st.columns(3)
                ecols[0].metric("Entropy (norm)", f"{step.get('entropy_normalised', 0):.3f}")
                ecols[1].metric("Clusters", step.get("n_clusters", "?"))
                ecols[2].metric("Consensus", f"{step.get('consensus_ratio', 0):.0%}")
                st.markdown(f"**Reasoning:** {output}")
            else:
                st.markdown(output)

            # Extra metadata for critique step
            if step_type == "critique":
                issues = step.get("issues", [])
                if issues:
                    st.warning("Issues identified:\n" + "\n".join(f"- {iss}" for iss in issues))

    # ── Plan ──
    if plan:
        with st.expander("📋 Execution plan"):
            st.json(plan)

    # ── Raw result ──
    with st.expander("🔩 Raw JSON result"):
        st.json(result)
