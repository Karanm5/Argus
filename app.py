import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="ARGUS — Reasoning System",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.argus-header { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.02em; }
.argus-sub    { font-size: 0.95rem; color: #6b7280; margin-bottom: 2rem; }
.metric-box   { text-align: center; padding: 1rem; background: #f1f5f9; border-radius: 10px; }
.metric-val   { font-size: 2rem; font-weight: 700; line-height: 1; }
.metric-lbl   { font-size: 0.78rem; color: #6b7280; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="argus-header">👁 ARGUS</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="argus-sub">Adaptive Reasoning with Guided Uncertainty Sampling — '
    'self-auditing multi-agent LLM reasoning</p>',
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading ARGUS pipeline...")
def load_orchestrator():
    from argus.core.orchestrator import ArgusOrchestrator
    return ArgusOrchestrator()


with st.sidebar:
    st.header("📚 Knowledge Base")
    uploaded = st.file_uploader("Upload a .txt document", type=["txt", "md"])
    if uploaded and st.button("Ingest document"):
        orch = load_orchestrator()
        text = uploaded.read().decode("utf-8")
        orch.memory.add_document(text, {"source": uploaded.name})
        st.success(f"Ingested {len(text)} characters.")

    st.divider()
    st.markdown("""
**Key innovation:** Semantic entropy-based uncertainty quantification.
Samples responses N times, clusters by meaning, computes Shannon entropy.
Re-routes if H > threshold.

**Ref:** Kuhn et al., *Semantic Uncertainty*, ICLR 2023.

[GitHub](https://github.com/Karanm5/Argus)
    """)

query = st.text_area(
    "Enter your query",
    placeholder="e.g. What are the limitations of transformer attention?",
    height=120,
)

if st.button("Run ARGUS", type="primary") and query.strip():
    orch = load_orchestrator()
    with st.spinner("Running multi-agent pipeline..."):
        result = orch.run(query=query.strip())
    st.session_state["result"] = result

if "result" in st.session_state:
    result = st.session_state["result"]
    unc = result.get("uncertainty_result", {})
    crit = result.get("critique_result", {})
    trace = result.get("trace", [])

    st.subheader("📝 Final answer")
    st.info(result.get("final_answer", ""))

    st.subheader("📊 Pipeline metrics")
    m1, m2, m3, m4 = st.columns(4)
    entropy_norm = unc.get("entropy_normalised", 0.0)
    unc_level = unc.get("uncertainty_level", "—")
    mean_score = crit.get("mean_score", 0.0)

    with m1:
        st.markdown(
            f'<div class="metric-box"><div class="metric-val">{entropy_norm:.3f}</div>'
            f'<div class="metric-lbl">Semantic entropy</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-box"><div class="metric-val">{unc_level.upper()}</div>'
            f'<div class="metric-lbl">Uncertainty level</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        colour = "#15803d" if mean_score >= 0.75 else "#b45309"
        st.markdown(
            f'<div class="metric-box"><div class="metric-val" style="color:{colour}">{mean_score:.2f}</div>'
            f'<div class="metric-lbl">Critic score</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-box"><div class="metric-val">{len(trace)}</div>'
            f'<div class="metric-lbl">Reasoning steps</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🔭 Uncertainty clusters")
        cluster_probs = unc.get("cluster_probabilities", {})
        if cluster_probs:
            fig = go.Figure(go.Bar(
                x=[f"Cluster {k}" for k in cluster_probs],
                y=list(cluster_probs.values()),
                marker_color=["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd"][:len(cluster_probs)],
            ))
            fig.update_layout(
                yaxis_range=[0, 1],
                height=280,
                margin=dict(l=40, r=10, t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cluster data available.")

    with c2:
        st.subheader("🎯 Critic rubric")
        scores = crit.get("scores", {})
        if scores:
            cats = [c.replace("_", " ").title() for c in scores]
            vals = list(scores.values())
            fig = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself",
                fillcolor="rgba(99,102,241,0.15)",
                line=dict(color="#6366f1", width=2),
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 1])),
                height=280,
                margin=dict(l=40, r=40, t=30, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No critique scores available.")

    st.subheader("🧠 Reasoning trace")
    for i, step in enumerate(trace):
        agent = step.get("agent", "?")
        step_type = step.get("type", "")
        latency = step.get("latency_ms", 0)
        tokens = step.get("tokens", 0)
        with st.expander(
            f"Step {i+1} — {agent} ({step_type})  [{latency:.0f}ms · {tokens} tokens]",
            expanded=(i == len(trace) - 1),
        ):
            if step_type == "uncertainty_estimation":
                ec = st.columns(3)
                ec[0].metric("Entropy", f"{step.get('entropy_normalised', 0):.3f}")
                ec[1].metric("Clusters", step.get("n_clusters", "?"))
                ec[2].metric("Consensus", f"{step.get('consensus_ratio', 0):.0%}")
            st.markdown(step.get("output", ""))
