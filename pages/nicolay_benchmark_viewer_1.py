"""
nicolay_benchmark_viewer.py
Nicolay RAG System — Benchmark Results Visualization & Human Annotation Tool
For DHQ article: "Nicolay: A Transparent RAG System for Historical Corpus Exploration"

Usage:
    streamlit run nicolay_benchmark_viewer.py

Data loading — two modes:
  1. EMBEDDED (default, no configuration needed):
     All canonical five-run results are hardcoded. The app runs immediately
     with no external files. Use this for sharing or presenting.

  2. LIVE CSV via GitHub (optional, for granular per-observation inspection):
     Click "Load from GitHub" in the sidebar. The app fetches all five
     merged_run_N.csv files and bootstrap_summary_final.csv directly from:
       https://github.com/Dr-Hutchinson/nicolay/tree/main/benchmark_data/
     Files are cached for the session. Use the refresh button (↻) to reload.
     No local file paths or manual configuration needed.

     The app normalizes column names automatically:
       Query → QueryText
       RubricFactualAccuracy → FA  (etc.)
       RerankerScoreMaxTop5 - RerankerScoreMinTop5 → Rerank spread

  The Human Annotation tab accepts a single CSV uploaded directly via the browser
  file uploader widget (any merged_run_N.csv works as-is).

Architecture: single-file script, no API calls, pure visualization + annotation.
Charts: Plotly.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import io
from datetime import datetime

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Nicolay Benchmark Viewer",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CANONICAL DATA (embedded — used when CSVs not loaded)
# Source: five-run-benchmark-results.md, 2026-03-28
# Configuration: H4N4 ada-002, 886-chunk corpus, rerank-v4.0-pro, k=5
# -----------------------------------------------------------------------------

CANONICAL_ARTICLE_NUMBER = 2.883
CI_LOWER = 2.802
CI_UPPER = 2.961

GRAND_MEANS = {
    "FA": {"mean": 0.810, "ci_lo": 0.790, "ci_hi": 0.831},
    "CA": {"mean": 0.842, "ci_lo": 0.817, "ci_hi": 0.865},
    "HD": {"mean": 0.608, "ci_lo": 0.573, "ci_hi": 0.642},
    "EC": {"mean": 0.623, "ci_lo": 0.592, "ci_hi": 0.654},
    "Total": {"mean": 2.883, "ci_lo": 2.802, "ci_hi": 2.961},
}

PER_RUN_MEANS = {
    "Run 0": {"FA": 0.823, "CA": 0.854, "HD": 0.646, "EC": 0.656, "Total": 2.979},
    "Run 1": {"FA": 0.813, "CA": 0.865, "HD": 0.594, "EC": 0.615, "Total": 2.885},
    "Run 2": {"FA": 0.792, "CA": 0.813, "HD": 0.604, "EC": 0.604, "Total": 2.813},
    "Run 3": {"FA": 0.823, "CA": 0.833, "HD": 0.594, "EC": 0.625, "Total": 2.875},
    "Run 4": {"FA": 0.802, "CA": 0.844, "HD": 0.604, "EC": 0.615, "Total": 2.865},
}

CATEGORY_RESULTS = [
    {"category": "comparative_analysis", "n": 6, "Total": 3.133, "FA": 0.833, "CA": 0.908, "HD": 0.700, "EC": 0.692, "R@5": 0.683},
    {"category": "factual_retrieval",    "n": 5, "Total": 2.960, "FA": 0.840, "CA": 0.890, "HD": 0.570, "EC": 0.660, "R@5": 0.805},
    {"category": "race_citizenship",     "n": 4, "Total": 2.875, "FA": 0.788, "CA": 0.788, "HD": 0.675, "EC": 0.625, "R@5": 0.321},
    {"category": "analysis",             "n": 4, "Total": 2.788, "FA": 0.813, "CA": 0.788, "HD": 0.563, "EC": 0.625, "R@5": 0.800},
    {"category": "synthesis",            "n": 5, "Total": 2.590, "FA": 0.770, "CA": 0.800, "HD": 0.520, "EC": 0.500, "R@5": 0.322},
]

# Per-query table: QID, Category, Total, SD, FA, CA, HD, EC, R@5, KW, Sem, ReRank
PER_QUERY_DATA = [
    {"QID": "Q7",   "Category": "comparative_analysis", "Total": 3.65, "SD": 0.137, "FA": 0.90, "CA": 1.00, "HD": 1.00, "EC": 0.75, "R@5": 0.625, "KW": 1.8, "Sem": 3.2, "Rerank": 0.026},
    {"QID": "Q8",   "Category": "comparative_analysis", "Total": 3.50, "SD": 0.354, "FA": 1.00, "CA": 1.00, "HD": 0.85, "EC": 0.65, "R@5": 0.775, "KW": 2.8, "Sem": 2.2, "Rerank": 0.074},
    {"QID": "S-4",  "Category": "synthesis",            "Total": 3.30, "SD": 0.326, "FA": 0.80, "CA": 0.80, "HD": 0.90, "EC": 0.80, "R@5": 0.200, "KW": 1.2, "Sem": 3.8, "Rerank": 0.043},
    {"QID": "Q1",   "Category": "factual_retrieval",    "Total": 3.30, "SD": 0.326, "FA": 0.95, "CA": 0.95, "HD": 0.65, "EC": 0.75, "R@5": 1.000, "KW": 4.6, "Sem": 0.4, "Rerank": 0.488},
    {"QID": "AN-5", "Category": "analysis",             "Total": 3.20, "SD": 0.209, "FA": 1.00, "CA": 0.85, "HD": 0.60, "EC": 0.75, "R@5": 1.000, "KW": 3.8, "Sem": 1.2, "Rerank": 0.179},
    {"QID": "CA-5", "Category": "comparative_analysis", "Total": 3.20, "SD": 0.209, "FA": 0.80, "CA": 0.85, "HD": 0.80, "EC": 0.75, "R@5": 0.800, "KW": 1.8, "Sem": 3.2, "Rerank": 0.135},
    {"QID": "Q3",   "Category": "factual_retrieval",    "Total": 3.20, "SD": 0.209, "FA": 0.75, "CA": 0.85, "HD": 0.70, "EC": 0.90, "R@5": 0.600, "KW": 3.0, "Sem": 2.0, "Rerank": 0.238},
    {"QID": "RC-3", "Category": "race_citizenship",     "Total": 3.20, "SD": 0.371, "FA": 0.95, "CA": 0.80, "HD": 0.80, "EC": 0.65, "R@5": 0.200, "KW": 1.4, "Sem": 3.6, "Rerank": 0.050},
    {"QID": "Q9",   "Category": "comparative_analysis", "Total": 3.00, "SD": 0.306, "FA": 0.75, "CA": 0.90, "HD": 0.55, "EC": 0.80, "R@5": 0.500, "KW": 1.2, "Sem": 3.8, "Rerank": 0.085},
    {"QID": "R3",   "Category": "factual_retrieval",    "Total": 3.00, "SD": 0.354, "FA": 0.95, "CA": 0.95, "HD": 0.55, "EC": 0.55, "R@5": 0.960, "KW": 3.4, "Sem": 1.6, "Rerank": 0.064},
    {"QID": "RC-5", "Category": "race_citizenship",     "Total": 3.00, "SD": 0.354, "FA": 0.75, "CA": 0.80, "HD": 0.70, "EC": 0.75, "R@5": 0.150, "KW": 1.8, "Sem": 3.2, "Rerank": 0.079},
    {"QID": "Q13",  "Category": "race_citizenship",     "Total": 2.80, "SD": 0.512, "FA": 0.70, "CA": 0.80, "HD": 0.70, "EC": 0.60, "R@5": 0.000, "KW": 3.2, "Sem": 1.8, "Rerank": 0.015},
    {"QID": "CA-6", "Category": "comparative_analysis", "Total": 2.75, "SD": 0.250, "FA": 0.80, "CA": 0.80, "HD": 0.50, "EC": 0.65, "R@5": 0.750, "KW": 2.8, "Sem": 2.2, "Rerank": 0.181},
    {"QID": "Q4",   "Category": "analysis",             "Total": 2.70, "SD": 0.112, "FA": 0.75, "CA": 0.75, "HD": 0.50, "EC": 0.70, "R@5": 1.000, "KW": 2.6, "Sem": 2.4, "Rerank": 0.221},
    {"QID": "R2",   "Category": "comparative_analysis", "Total": 2.70, "SD": 0.209, "FA": 0.75, "CA": 0.90, "HD": 0.50, "EC": 0.55, "R@5": 0.650, "KW": 2.0, "Sem": 3.0, "Rerank": 0.091},
    {"QID": "Q2",   "Category": "factual_retrieval",    "Total": 2.65, "SD": 0.454, "FA": 0.75, "CA": 0.85, "HD": 0.45, "EC": 0.60, "R@5": 0.867, "KW": 4.0, "Sem": 1.0, "Rerank": 0.147},
    {"QID": "FR-2", "Category": "factual_retrieval",    "Total": 2.65, "SD": 0.224, "FA": 0.80, "CA": 0.85, "HD": 0.50, "EC": 0.50, "R@5": 0.600, "KW": 0.6, "Sem": 4.4, "Rerank": 0.041},
    {"QID": "Q10",  "Category": "synthesis",            "Total": 2.65, "SD": 0.224, "FA": 0.80, "CA": 0.85, "HD": 0.50, "EC": 0.50, "R@5": 0.600, "KW": 4.4, "Sem": 0.6, "Rerank": 0.222},
    {"QID": "R1",   "Category": "analysis",             "Total": 2.65, "SD": 0.224, "FA": 0.75, "CA": 0.85, "HD": 0.50, "EC": 0.55, "R@5": 0.800, "KW": 1.0, "Sem": 4.0, "Rerank": 0.097},
    {"QID": "Q5",   "Category": "analysis",             "Total": 2.60, "SD": 0.137, "FA": 0.75, "CA": 0.70, "HD": 0.65, "EC": 0.50, "R@5": 0.400, "KW": 1.2, "Sem": 3.8, "Rerank": 0.013},
    {"QID": "S-5",  "Category": "synthesis",            "Total": 2.55, "SD": 0.112, "FA": 0.75, "CA": 0.80, "HD": 0.50, "EC": 0.50, "R@5": 0.200, "KW": 2.8, "Sem": 2.2, "Rerank": 0.201},
    {"QID": "RC-4", "Category": "race_citizenship",     "Total": 2.50, "SD": 0.000, "FA": 0.75, "CA": 0.75, "HD": 0.50, "EC": 0.50, "R@5": 0.933, "KW": 0.6, "Sem": 4.4, "Rerank": 0.034},
    {"QID": "Q12",  "Category": "synthesis",            "Total": 2.25, "SD": 0.306, "FA": 0.75, "CA": 0.80, "HD": 0.40, "EC": 0.30, "R@5": 0.267, "KW": 2.2, "Sem": 2.8, "Rerank": 0.027},
    {"QID": "Q11",  "Category": "synthesis",            "Total": 2.20, "SD": 0.209, "FA": 0.75, "CA": 0.75, "HD": 0.30, "EC": 0.40, "R@5": 0.343, "KW": 1.0, "Sem": 4.0, "Rerank": 0.062},
]

HEATMAP_DATA = {
    "Q7":   [3.75, 3.50, 3.75, 3.75, 3.50],
    "Q8":   [3.75, 3.75, 3.00, 3.50, 3.50],
    "S-4":  [3.00, 3.25, 3.00, 3.50, 3.75],
    "Q1":   [3.25, 3.50, 3.00, 3.50, 3.25],
    "AN-5": [3.25, 3.00, 3.50, 3.25, 3.00],
    "CA-5": [3.25, 3.00, 3.00, 3.50, 3.25],
    "Q3":   [3.50, 3.25, 3.00, 3.25, 3.00],
    "RC-3": [3.00, 3.50, 3.00, 3.00, 3.50],
    "Q9":   [3.25, 3.25, 2.75, 2.75, 3.00],
    "R3":   [3.25, 3.00, 2.50, 3.00, 3.25],
    "RC-5": [3.25, 3.00, 2.50, 3.00, 3.25],
    "Q13":  [3.50, 2.50, 2.50, 2.75, 2.75],
    "CA-6": [2.50, 3.00, 2.75, 3.00, 2.50],
    "Q4":   [2.75, 2.75, 2.75, 2.50, 2.75],
    "R2":   [2.75, 2.50, 2.75, 2.75, 2.75],
    "Q2":   [3.00, 2.50, 2.50, 2.75, 2.50],
    "FR-2": [2.50, 2.75, 3.00, 2.50, 2.50],
    "Q10":  [2.50, 2.75, 2.50, 2.75, 2.75],
    "R1":   [2.75, 2.50, 2.50, 2.75, 2.75],
    "Q5":   [2.75, 2.50, 2.50, 2.75, 2.50],
    "S-5":  [2.50, 2.50, 2.50, 2.75, 2.50],
    "RC-4": [2.50, 2.50, 2.50, 2.50, 2.50],
    "Q12":  [2.00, 2.50, 2.25, 2.25, 2.25],
    "Q11":  [2.00, 2.25, 2.00, 2.50, 2.25],
}

# Named questions for annotation (from question-guide-article.md)
NAMED_QUESTIONS = {
    "Q7":   "Embedding Model Exhibit / HD=1.00 Anchor",
    "Q11":  "HD Ceiling Floor / Synthesis Failure",
    "Q13":  "Parametric Laundering Canonical Case",
    "RC-3": "HD Upgrade Mechanism (T3→T4)",
    "RC-4": "Zero-Variance Reproducibility",
    "S-4":  "Calibration Decoupling Exhibit",
    "Q1":   "Pipeline Resilience / R@5=1.00",
    "Q3":   "CA Fabrication Exhibit (R4)",
    "FR-2": "Flat-Reranker Pattern",
    "CA-5": "Displaced Quote (R0/R1) / Upgrade",
}

# Flat-reranker cases
FLAT_RERANKER_QIDS = {"Q13", "Q5", "Q12", "Q7", "FR-2"}

# Color palette
CAT_COLORS = {
    "comparative_analysis": "#4C78A8",
    "factual_retrieval":    "#72B7B2",
    "race_citizenship":     "#E45756",
    "analysis":             "#F58518",
    "synthesis":            "#54A24B",
}


# DATA LOADING — GitHub raw fetch
# -----------------------------------------------------------------------------

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/Dr-Hutchinson/nicolay/main/benchmark_data/"
)
RUN_FILES   = [f"merged_run_{i}.csv" for i in range(5)]
SUMMARY_FILE = "bootstrap_summary_final.csv"


def normalize_run_df(df, run_idx):
    """
    Normalize a merged_run_N.csv to the column names the viewer expects.
    Handles known schema quirks from the actual benchmark CSVs.
    """
    df = df.copy()
    df["run"] = run_idx

    renames = {
        "Query":                          "QueryText",
        "RubricFactualAccuracy":          "FA",
        "RubricCitationAccuracy":         "CA",
        "RubricHistoriographicalDepth":   "HD",
        "RubricEpistemicCalibration":     "EC",
        "RecallAt5":                      "R@5",
        "RetrievalKeywordCountTop5":      "KW",
        "RetrievalSemanticCountTop5":     "Sem",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    if "Rerank" not in df.columns:
        if "RerankerScoreMaxTop5" in df.columns and "RerankerScoreMinTop5" in df.columns:
            df["Rerank"] = df["RerankerScoreMaxTop5"] - df["RerankerScoreMinTop5"]
        else:
            df["Rerank"] = float("nan")

    if "RubricTotal" not in df.columns and all(d in df.columns for d in ["FA", "CA", "HD", "EC"]):
        df["RubricTotal"] = df[["FA", "CA", "HD", "EC"]].sum(axis=1)

    return df


@st.cache_data(show_spinner=False)
def fetch_github_csvs():
    """
    Fetch all benchmark CSVs from GitHub raw URLs.
    Cached per session. Sidebar reload button clears the cache.
    Returns dict with keys 'runs' (DataFrame), optionally 'summary' (DataFrame),
    and 'errors' (list of error strings).
    """
    import requests

    results = {}
    errors  = []

    run_dfs = []
    for i, fname in enumerate(RUN_FILES):
        url = GITHUB_RAW_BASE + fname
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df = normalize_run_df(df, i)
            run_dfs.append(df)
        except Exception as e:
            errors.append(f"{fname}: {e}")

    if run_dfs:
        results["runs"] = pd.concat(run_dfs, ignore_index=True)

    url = GITHUB_RAW_BASE + SUMMARY_FILE
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        results["summary"] = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        errors.append(f"{SUMMARY_FILE}: {e}")

    results["errors"] = errors
    return results


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------

st.sidebar.title("🏛️ Nicolay Viewer")
st.sidebar.caption("H4N4 ada-002 · 886 chunks · 5 runs · n=120")

st.sidebar.markdown("---")
st.sidebar.subheader("Live Data")

if "live_loaded" not in st.session_state:
    st.session_state.live_loaded = None

col_btn, col_reload = st.sidebar.columns([2, 1])
with col_btn:
    load_btn = st.button("⬇ Load from GitHub", use_container_width=True)
with col_reload:
    if st.button("↻", help="Clear cache and reload from GitHub"):
        fetch_github_csvs.clear()
        st.session_state.live_loaded = None
        st.rerun()

if load_btn:
    with st.spinner("Fetching CSVs from GitHub…"):
        st.session_state.live_loaded = fetch_github_csvs()

loaded = st.session_state.live_loaded or {}
using_live_data = "runs" in loaded

if using_live_data:
    n_runs = loaded["runs"]["run"].nunique()
    n_obs  = len(loaded["runs"])
    st.sidebar.success(f"✓ {n_runs} runs · {n_obs} observations loaded")
    if loaded.get("errors"):
        for err in loaded["errors"]:
            st.sidebar.warning(err)
elif load_btn:
    st.sidebar.error("Load failed — check the errors above.")
    if loaded.get("errors"):
        for err in loaded["errors"]:
            st.sidebar.warning(err)
else:
    st.sidebar.info("Using embedded canonical data (2026-03-28)")

st.sidebar.markdown("---")
st.sidebar.subheader("Display Options")
show_annotations   = st.sidebar.checkbox("Show named question labels", value=True)
show_flat_reranker = st.sidebar.checkbox("Highlight flat-reranker cases", value=True)
color_by = st.sidebar.radio("Color charts by", ["Category", "Score range"], index=0)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Article: *Nicolay: A Transparent RAG System for Historical Corpus Exploration*  \n"
    "Venue: *Digital Humanities Quarterly* (forthcoming)  \n"
    "Bootstrap: n=1000, seed=42"
)

# -----------------------------------------------------------------------------
# BUILD WORKING DATAFRAME
# -----------------------------------------------------------------------------

df_query = pd.DataFrame(PER_QUERY_DATA)
df_query = df_query.sort_values("Total", ascending=False).reset_index(drop=True)
df_cat = pd.DataFrame(CATEGORY_RESULTS)

# -----------------------------------------------------------------------------
# TAB LAYOUT
# -----------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Grand Summary",
    "📂 Category Performance",
    "🔥 Per-Query Heatmap",
    "🔍 Retrieval Architecture",
    "📐 HD vs EC / Scatter",
    "✏️ Human Annotation",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: GRAND SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Grand Summary — H4N4 ada-002 (5-run Formal Benchmark)")

    # Article number callout
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.markdown(
            f"""
            <div style="background:#1a1a2e;border-radius:12px;padding:24px;text-align:center;border:2px solid #b87333;">
            <div style="color:#b87333;font-size:13px;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">
            CANONICAL ARTICLE NUMBER</div>
            <div style="color:#f5f0e8;font-size:52px;font-weight:700;line-height:1;">{CANONICAL_ARTICLE_NUMBER:.3f}</div>
            <div style="color:#aaa;font-size:14px;margin-top:6px;">
            95% CI [{CI_LOWER:.3f}, {CI_UPPER:.3f}] · bootstrap n=1,000 · k=5 runs · n=120 obs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Dimension means with CI bars
    dims = ["FA", "CA", "HD", "EC"]
    dim_labels = {
        "FA": "Factual Accuracy",
        "CA": "Citation Accuracy",
        "HD": "Historiographical Depth",
        "EC": "Epistemic Calibration",
    }
    dim_colors = {"FA": "#4C78A8", "CA": "#72B7B2", "HD": "#E45756", "EC": "#F58518"}
    dim_notes = {
        "FA": "System strength — stable across runs",
        "CA": "System strength — highest dimension mean",
        "HD": "⚠ Generation ceiling — 60.8% downgrade rate drives HD to 0.50",
        "EC": "Retrieval-mediated — degrades at zero-retrieval cases",
    }

    fig_dims = go.Figure()
    for dim in dims:
        d = GRAND_MEANS[dim]
        fig_dims.add_trace(go.Bar(
            x=[d["mean"]],
            y=[dim_labels[dim]],
            orientation="h",
            error_x=dict(
                type="data",
                array=[d["ci_hi"] - d["mean"]],
                arrayminus=[d["mean"] - d["ci_lo"]],
                color="rgba(255,255,255,0.5)",
                thickness=2,
                width=6,
            ),
            marker_color=dim_colors[dim],
            text=[f"{d['mean']:.3f}"],
            textposition="outside",
            textfont=dict(size=13),
            name=dim_labels[dim],
            hovertemplate=(
                f"<b>{dim_labels[dim]}</b><br>"
                f"Mean: {d['mean']:.3f}<br>"
                f"95% CI: [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]<br>"
                f"{dim_notes[dim]}<extra></extra>"
            ),
        ))

    fig_dims.update_layout(
        title="Dimension Means with 95% Bootstrap CI",
        xaxis=dict(title="Mean Score (0–1 per dimension)", range=[0, 1.15]),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        height=300,
        margin=dict(l=20, r=60, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
    )
    st.plotly_chart(fig_dims, use_container_width=True)

    # Dimension notes
    col1, col2, col3, col4 = st.columns(4)
    for col, dim in zip([col1, col2, col3, col4], dims):
        d = GRAND_MEANS[dim]
        with col:
            st.metric(
                label=dim_labels[dim],
                value=f"{d['mean']:.3f}",
                delta=f"CI [{d['ci_lo']:.3f}–{d['ci_hi']:.3f}]",
                delta_color="off",
            )
            st.caption(dim_notes[dim])

    st.markdown("---")

    # Per-run stability
    st.subheader("Per-Run Stability")
    runs = list(PER_RUN_MEANS.keys())
    totals = [PER_RUN_MEANS[r]["Total"] for r in runs]

    fig_runs = go.Figure()
    fig_runs.add_trace(go.Scatter(
        x=runs, y=totals,
        mode="lines+markers+text",
        text=[f"{v:.3f}" for v in totals],
        textposition="top center",
        marker=dict(size=10, color="#b87333"),
        line=dict(color="#b87333", width=2),
        name="Run Total",
    ))
    fig_runs.add_hline(
        y=CANONICAL_ARTICLE_NUMBER,
        line_dash="dash", line_color="#aaa",
        annotation_text=f"Grand mean: {CANONICAL_ARTICLE_NUMBER:.3f}",
        annotation_position="right",
    )
    fig_runs.add_hrect(
        y0=CI_LOWER, y1=CI_UPPER,
        fillcolor="rgba(184,115,51,0.1)",
        line_width=0,
        annotation_text="95% CI",
        annotation_position="right",
    )
    fig_runs.update_layout(
        title="Run-Level Total Score (spread: 2.813–2.979)",
        yaxis=dict(title="Mean Total Score (0–4)", range=[2.6, 3.2]),
        height=280,
        margin=dict(l=20, r=80, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        showlegend=False,
    )
    st.plotly_chart(fig_runs, use_container_width=True)
    st.caption(
        "FA/CA are stable across runs. HD/EC show greater per-run variance. "
        "Bootstrap CI captures this spread appropriately. Run 0 is the high watermark (2.979); Run 2 the low (2.813)."
    )

    # Quote verification summary
    st.markdown("---")
    st.subheader("Quote Verification (n=120 observations)")
    qv_cols = st.columns(5)
    qv_data = [
        ("Verified", 597, "✓", "#4C78A8"),
        ("Approximate", 0, "~", "#72B7B2"),
        ("Displaced", 2, "⇌", "#F58518"),
        ("Fabricated", 1, "✗", "#E45756"),
        ("Mislabeled", 0, "?", "#aaa"),
    ]
    for col, (label, n, icon, color) in zip(qv_cols, qv_data):
        with col:
            st.markdown(
                f"<div style='text-align:center;padding:12px;background:rgba(0,0,0,0.05);border-radius:8px;"
                f"border-top:3px solid {color};'>"
                f"<div style='font-size:24px;'>{icon}</div>"
                f"<div style='font-size:28px;font-weight:700;color:{color};'>{n}</div>"
                f"<div style='font-size:12px;color:#888;'>{label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.caption(
        "**Fabrication rate: 0.8%** (1/120). Named: CA-5 R0/R1 displaced (sentence splice); Q3 R4 fabricated (Third Annual Message). "
        "Q5 R0: evaluator override (CA=0.50 despite Verified=5 — phrasing mismatch + retrieval failure)."
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: CATEGORY PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Category Performance")
    st.caption(
        "The **race/citizenship paradox** is the key visual: second-best total score (2.875) "
        "with the joint-worst R@5 (0.321). Rubric performance is driven by parametric knowledge, not RAG retrieval."
    )

    cat_order = [c["category"] for c in sorted(CATEGORY_RESULTS, key=lambda x: -x["Total"])]

    # Total + R@5 grouped bars
    fig_cat = go.Figure()
    totals_cat = [next(c for c in CATEGORY_RESULTS if c["category"] == cat)["Total"] for cat in cat_order]
    r5_cat = [next(c for c in CATEGORY_RESULTS if c["category"] == cat)["R@5"] for cat in cat_order]
    colors_cat = [CAT_COLORS[cat] for cat in cat_order]
    labels_cat = [cat.replace("_", " ").title() for cat in cat_order]

    fig_cat.add_trace(go.Bar(
        name="Rubric Total (÷4)",
        x=labels_cat,
        y=[t / 4 for t in totals_cat],
        marker_color=colors_cat,
        text=[f"{t:.3f}" for t in totals_cat],
        textposition="outside",
        textfont=dict(size=12),
        hovertemplate="<b>%{x}</b><br>Total: %{text}<extra></extra>",
    ))
    fig_cat.add_trace(go.Bar(
        name="R@5 (Recall@5)",
        x=labels_cat,
        y=r5_cat,
        marker_color=[c.replace(")", ",0.5)").replace("rgb", "rgba") if "rgb" in c else c + "88" for c in colors_cat],
        marker_pattern_shape="/",
        text=[f"{v:.3f}" for v in r5_cat],
        textposition="outside",
        textfont=dict(size=12),
        hovertemplate="<b>%{x}</b><br>R@5: %{text}<extra></extra>",
    ))

    # Add annotation for the paradox
    fig_cat.add_annotation(
        x="Race Citizenship",
        y=2.875 / 4 + 0.05,
        text="Paradox: high score,<br>low retrieval",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#E45756",
        font=dict(color="#E45756", size=11),
        ax=60, ay=-40,
    )

    fig_cat.update_layout(
        barmode="group",
        title="Category Rubric Total vs. R@5 (Recall@5)",
        yaxis=dict(title="Score (normalized 0–1)", range=[0, 1.15]),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    # Dimension breakdown per category
    st.subheader("Dimension Breakdown by Category")
    dims_display = ["FA", "CA", "HD", "EC"]
    dim_cols = {
        "FA": "#4C78A8", "CA": "#72B7B2", "HD": "#E45756", "EC": "#F58518",
    }

    fig_dim_cat = go.Figure()
    for dim in dims_display:
        vals = [next(c for c in CATEGORY_RESULTS if c["category"] == cat)[dim] for cat in cat_order]
        fig_dim_cat.add_trace(go.Bar(
            name=dim,
            x=labels_cat,
            y=vals,
            marker_color=dim_cols[dim],
            text=[f"{v:.2f}" for v in vals],
            textposition="inside",
            textfont=dict(size=10, color="white"),
        ))

    fig_dim_cat.update_layout(
        barmode="stack",
        title="Stacked Dimension Scores by Category",
        yaxis=dict(title="Cumulative Score", range=[0, 4.2]),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
    )
    st.plotly_chart(fig_dim_cat, use_container_width=True)

    # Summary table
    st.subheader("Category Summary Table")
    df_cat_display = df_cat.copy()
    df_cat_display["category"] = df_cat_display["category"].str.replace("_", " ").str.title()
    df_cat_display = df_cat_display.sort_values("Total", ascending=False)
    st.dataframe(
        df_cat_display.style.format({
            "Total": "{:.3f}", "FA": "{:.3f}", "CA": "{:.3f}",
            "HD": "{:.3f}", "EC": "{:.3f}", "R@5": "{:.3f}",
        }).background_gradient(subset=["Total"], cmap="RdYlGn", vmin=2.4, vmax=3.4),
        use_container_width=True,
        height=230,
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: PER-QUERY HEATMAP
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Per-Query Heatmap — Rubric Total by Run")
    st.caption(
        "Color = total score (0–4). Rows sorted by mean total (high→low). "
        "Q13 has the highest variance (SD=0.512); RC-4 the lowest (SD=0.000, all runs = 2.50)."
    )

    # Build heatmap matrix
    qids_sorted = [row["QID"] for _, row in df_query.iterrows()]
    z_matrix = [HEATMAP_DATA[qid] for qid in qids_sorted]
    means_sorted = [row["Total"] for _, row in df_query.iterrows()]
    sds_sorted = [row["SD"] for _, row in df_query.iterrows()]
    cats_sorted = [row["Category"] for _, row in df_query.iterrows()]

    # Y-axis labels: optionally annotated
    if show_annotations:
        y_labels = [
            f"{'★ ' if qid in NAMED_QUESTIONS else ''}{qid} ({cat.replace('_',' ')[:3].upper()})"
            for qid, cat in zip(qids_sorted, cats_sorted)
        ]
    else:
        y_labels = [
            f"{qid} ({cat.replace('_',' ')[:3].upper()})"
            for qid, cat in zip(qids_sorted, cats_sorted)
        ]

    customdata = [
        [qid, f"{mean:.2f}", f"{sd:.3f}",
         NAMED_QUESTIONS.get(qid, "—"),
         "⚠ flat-reranker" if qid in FLAT_RERANKER_QIDS else ""]
        for qid, mean, sd in zip(qids_sorted, means_sorted, sds_sorted)
    ]

    fig_hm = go.Figure(go.Heatmap(
        z=z_matrix,
        x=["Run 0", "Run 1", "Run 2", "Run 3", "Run 4"],
        y=y_labels,
        colorscale=[
            [0.0, "#67000d"],
            [0.25, "#E45756"],
            [0.50, "#F58518"],
            [0.75, "#4C78A8"],
            [1.0,  "#1a6b3c"],
        ],
        zmin=2.0, zmax=4.0,
        text=[[f"{v:.2f}" for v in row] for row in z_matrix],
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(
            title="Score (0–4)",
            tickvals=[2.0, 2.5, 3.0, 3.5, 4.0],
        ),
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[0]}</b> — %{x}<br>"
            "Score: %{z:.2f}<br>"
            "Mean: %{customdata[1]} · SD: %{customdata[2]}<br>"
            "Named: %{customdata[3]}<br>"
            "%{customdata[4]}<extra></extra>"
        ),
    ))

    if show_flat_reranker:
        flat_idxs = [i for i, qid in enumerate(qids_sorted) if qid in FLAT_RERANKER_QIDS]
        for idx in flat_idxs:
            fig_hm.add_shape(
                type="rect",
                x0=-0.5, x1=4.5,
                y0=idx - 0.5, y1=idx + 0.5,
                line=dict(color="#FFD700", width=1.5, dash="dot"),
                fillcolor="rgba(0,0,0,0)",
            )

    fig_hm.update_layout(
        title="Rubric Total by Query × Run (sorted high→low by mean)",
        height=820,
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=11)),
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    if show_flat_reranker:
        fig_hm.add_annotation(
            text="⚠ Flat-reranker cases (dashed gold border)",
            xref="paper", yref="paper",
            x=1.0, y=-0.02,
            showarrow=False,
            font=dict(size=10, color="#FFD700"),
            xanchor="right",
        )

    st.plotly_chart(fig_hm, use_container_width=True)

    # Per-query table
    st.subheader("Per-Query Summary Table")
    df_display = df_query.copy()
    df_display["Named"] = df_display["QID"].map(lambda q: NAMED_QUESTIONS.get(q, ""))
    df_display["Flat-Reranker"] = df_display["QID"].map(lambda q: "⚠" if q in FLAT_RERANKER_QIDS else "")
    df_display["Category"] = df_display["Category"].str.replace("_", " ").str.title()
    cols_show = ["QID", "Category", "Total", "SD", "FA", "CA", "HD", "EC", "R@5", "Named"]
    st.dataframe(
        df_display[cols_show].style.format({
            "Total": "{:.2f}", "SD": "{:.3f}", "FA": "{:.2f}", "CA": "{:.2f}",
            "HD": "{:.2f}", "EC": "{:.2f}", "R@5": "{:.3f}",
        }).background_gradient(subset=["Total"], cmap="RdYlGn", vmin=2.0, vmax=4.0)
          .background_gradient(subset=["SD"], cmap="Oranges", vmin=0, vmax=0.55),
        use_container_width=True,
        height=680,
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: RETRIEVAL ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Retrieval Architecture")

    col_left, col_right = st.columns(2)

    # Search method composition: stacked bars sorted by total
    with col_left:
        st.subheader("Keyword vs. Semantic Composition")
        st.caption("Mean slots (of 5) contributed by each search method. Sorted by rubric total.")

        qids_s = df_query["QID"].tolist()
        kw_vals = df_query["KW"].tolist()
        sem_vals = df_query["Sem"].tolist()
        totals_s = df_query["Total"].tolist()

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Keyword",
            x=qids_s,
            y=kw_vals,
            marker_color="#4C78A8",
            hovertemplate="<b>%{x}</b><br>Keyword slots: %{y:.1f}<extra></extra>",
        ))
        fig_comp.add_trace(go.Bar(
            name="Semantic",
            x=qids_s,
            y=sem_vals,
            marker_color="#E45756",
            hovertemplate="<b>%{x}</b><br>Semantic slots: %{y:.1f}<extra></extra>",
        ))
        # Total overlay line
        fig_comp.add_trace(go.Scatter(
            name="Rubric Total",
            x=qids_s,
            y=totals_s,
            mode="markers",
            yaxis="y2",
            marker=dict(size=8, color="#F58518", symbol="diamond"),
            hovertemplate="<b>%{x}</b><br>Total: %{y:.2f}<extra></extra>",
        ))
        fig_comp.update_layout(
            barmode="stack",
            height=460,
            yaxis=dict(title="Mean Slots (of 5)", range=[0, 5.5]),
            yaxis2=dict(title="Rubric Total", overlaying="y", side="right", range=[1.8, 4.2]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.05)",
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.caption(
            "Keyword-dominant queries (Q1, Q10, Q2, AN-5, R3) are factual/analysis — best factual performance. "
            "Semantic-dominant (FR-2, Q11, R1, RC-4, Q5, Q9) show mixed outcomes."
        )

    # R@5 vs Total scatter
    with col_right:
        st.subheader("R@5 vs. Rubric Total")
        st.caption("Bubble size = SD (larger = more variable across runs). Color = category.")

        fig_scatter = px.scatter(
            df_query,
            x="R@5",
            y="Total",
            size="SD",
            size_max=30,
            color="Category",
            color_discrete_map={
                cat: CAT_COLORS[cat] for cat in CAT_COLORS
            },
            hover_name="QID",
            hover_data={"Total": ":.2f", "R@5": ":.3f", "SD": ":.3f"},
            text="QID" if show_annotations else None,
        )

        # Annotate key cases
        if show_annotations:
            for _, row in df_query.iterrows():
                if row["QID"] in NAMED_QUESTIONS:
                    fig_scatter.add_annotation(
                        x=row["R@5"], y=row["Total"],
                        text=row["QID"],
                        showarrow=True,
                        arrowhead=1,
                        arrowsize=0.5,
                        font=dict(size=9),
                        ax=15, ay=-15,
                    )

        fig_scatter.update_traces(textposition="top center", textfont=dict(size=8))
        fig_scatter.update_layout(
            height=460,
            xaxis=dict(title="R@5 (Recall@5)", range=[-0.05, 1.1]),
            yaxis=dict(title="Rubric Total (mean, 5 runs)", range=[1.9, 4.0]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.05)",
            legend=dict(title="Category", font=dict(size=10)),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption(
            "Race/citizenship cluster (red): high total, low R@5 — the parametric knowledge paradox. "
            "Q13 (R@5=0.000, Total=2.80) is the canonical exhibit."
        )

    # Reranker spread
    st.markdown("---")
    st.subheader("Reranker Spread — Flat-Reranker Detection")
    st.caption(
        "Reranker spread < 0.05 combined with R@5 < 0.30 indicates confident retrieval in the wrong semantic neighborhood. "
        "The reranker evaluates internal coherence, not query relevance — it cannot self-diagnose wrong-neighborhood retrieval."
    )

    df_rerank = df_query.sort_values("Rerank")
    flat_flag = df_rerank["QID"].isin(FLAT_RERANKER_QIDS)

    fig_rr = go.Figure()
    fig_rr.add_trace(go.Bar(
        x=df_rerank["QID"].tolist(),
        y=df_rerank["Rerank"].tolist(),
        marker_color=["#FFD700" if f else "#4C78A8" for f in flat_flag],
        text=[f"{v:.3f}" for v in df_rerank["Rerank"]],
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate="<b>%{x}</b><br>Spread: %{y:.3f}<extra></extra>",
    ))
    fig_rr.add_hline(y=0.05, line_dash="dash", line_color="#FFD700",
                     annotation_text="Flat-reranker threshold (0.05)",
                     annotation_position="right")

    fig_rr.update_layout(
        title="Reranker Score Spread by Query (sorted ascending)",
        height=320,
        yaxis=dict(title="Reranker Spread"),
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
    )
    st.plotly_chart(fig_rr, use_container_width=True)
    st.caption("Gold bars = flat-reranker cases (Q13, Q5, Q12, Q7, FR-2). Q1 has highest spread (0.488) = reranker most discriminating.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: HD vs EC / TYPE CLASSIFICATION SCATTER
# ═════════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("HD vs. EC Quadrant Analysis & Type Classification")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("HD vs. EC Quadrant")
        st.caption(
            "The HD ceiling at ~0.60 is the benchmark's structural feature. "
            "Queries above HD=0.75 are correct/upgrade classifications. "
            "EC degrades specifically at zero-retrieval (R@5=0)."
        )

        fig_hdec = go.Figure()
        for cat in CAT_COLORS:
            mask = df_query["Category"] == cat
            sub = df_query[mask]
            fig_hdec.add_trace(go.Scatter(
                x=sub["EC"],
                y=sub["HD"],
                mode="markers+text",
                name=cat.replace("_", " ").title(),
                marker=dict(
                    color=CAT_COLORS[cat],
                    size=[12 + s * 20 for s in sub["SD"]],
                    opacity=0.8,
                    line=dict(width=1, color="white"),
                ),
                text=sub["QID"] if show_annotations else None,
                textposition="top center",
                textfont=dict(size=8),
                hovertemplate="<b>%{text}</b><br>EC: %{x:.2f}<br>HD: %{y:.2f}<extra></extra>",
            ))

        # Quadrant lines
        fig_hdec.add_hline(y=GRAND_MEANS["HD"]["mean"], line_dash="dot", line_color="#888",
                           annotation_text=f"HD mean={GRAND_MEANS['HD']['mean']:.3f}")
        fig_hdec.add_vline(x=GRAND_MEANS["EC"]["mean"], line_dash="dot", line_color="#888",
                           annotation_text=f"EC mean={GRAND_MEANS['EC']['mean']:.3f}")

        fig_hdec.update_layout(
            height=440,
            xaxis=dict(title="Epistemic Calibration (EC)", range=[0.15, 1.05]),
            yaxis=dict(title="Historiographical Depth (HD)", range=[0.15, 1.10]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.05)",
            legend=dict(font=dict(size=9), itemsizing="constant"),
        )
        st.plotly_chart(fig_hdec, use_container_width=True)

    with col_r:
        st.subheader("Type Classification vs. Rubric Outcome")
        st.caption(
            "Direction impact: correct/upgrade → HD≈0.77. Downgrade → HD=0.500 (the ceiling mechanism). "
            "60.8% of observations produce a downgrade."
        )

        direction_data = [
            {"Direction": "Correct",   "n": 33, "Mean Total": 3.242, "Mean HD": 0.780},
            {"Direction": "Upgrade",   "n": 14, "Mean Total": 3.214, "Mean HD": 0.768},
            {"Direction": "Downgrade", "n": 73, "Mean Total": 2.658, "Mean HD": 0.500},
        ]
        dir_colors = {"Correct": "#4C78A8", "Upgrade": "#72B7B2", "Downgrade": "#E45756"}

        fig_dir = go.Figure()
        for d in direction_data:
            fig_dir.add_trace(go.Bar(
                name=d["Direction"],
                x=[d["Direction"]],
                y=[d["Mean Total"]],
                marker_color=dir_colors[d["Direction"]],
                text=[f"Total: {d['Mean Total']:.3f}<br>HD: {d['Mean HD']:.3f}<br>n={d['n']}"],
                textposition="outside",
                textfont=dict(size=11),
                hovertemplate=(
                    f"<b>{d['Direction']}</b><br>"
                    f"n={d['n']}<br>"
                    f"Mean Total: {d['Mean Total']:.3f}<br>"
                    f"Mean HD: {d['Mean HD']:.3f}<extra></extra>"
                ),
            ))

        fig_dir.update_layout(
            title="Classification Direction vs. Rubric Outcome",
            height=360,
            yaxis=dict(title="Mean Total Score", range=[0, 4.0]),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.05)",
        )
        st.plotly_chart(fig_dir, use_container_width=True)

        st.markdown("---")
        st.subheader("Nicolay Type Accuracy Summary")
        type_acc = [
            {"Type": "T1 (Direct)",        "n": 5,  "Accuracy": 1.000, "Note": "Perfect"},
            {"Type": "T2 (Basic Synth)",   "n": 25, "Accuracy": 0.400, "Note": "Downgrade to T1 common"},
            {"Type": "T3 (Comparative)",   "n": 45, "Accuracy": 0.089, "Note": "Black hole — 29/45 → T2"},
            {"Type": "T4 (Historio.)",     "n": 40, "Accuracy": 0.350, "Note": "Downgrade to T2 common"},
            {"Type": "T5 (Multi-source)",  "n": 5,  "Accuracy": 0.000, "Note": "All → T4 (Q13 all runs)"},
        ]
        df_ta = pd.DataFrame(type_acc)
        st.dataframe(
            df_ta.style.format({"Accuracy": "{:.1%}","n": "{:d}"})
                       .background_gradient(subset=["Accuracy"], cmap="RdYlGn", vmin=0, vmax=1),
            use_container_width=True,
            height=220,
        )
        st.caption(
            "T3 is the structural black hole (8.9% accuracy). "
            "Upgrade trigger = retrieval ambiguity, not query recognition. "
            "Nicolay has a functional 3-tier range: T1, T2, T4 — with T3 and T5 effectively absent."
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6: HUMAN ANNOTATION PANEL
# ═════════════════════════════════════════════════════════════════════════════

with tab6:
    st.header("Human Annotation Panel — Blind Rubric Scoring")
    st.info(
        "This panel implements the blind annotation protocol from the article methodology. "
        "Load a raw (unscored) benchmark CSV below. Responses are presented one at a time without "
        "prior knowledge of LLM scores. Score each response, then export the results."
    )

    if "annotation_scores" not in st.session_state:
        st.session_state.annotation_scores = {}
    if "annotation_idx" not in st.session_state:
        st.session_state.annotation_idx = 0
    if "annotation_df" not in st.session_state:
        st.session_state.annotation_df = None

    upload = st.file_uploader(
        "Upload raw benchmark CSV (must contain: QueryID, Category, QueryText, FinalAnswerText)",
        type=["csv"],
    )

    if upload:
        try:
            ann_df = pd.read_csv(upload)
            # Accept either 'Query' (actual CSV) or 'QueryText' (normalized name)
            if "Query" in ann_df.columns and "QueryText" not in ann_df.columns:
                ann_df = ann_df.rename(columns={"Query": "QueryText"})
            required_cols = {"QueryID", "Category", "QueryText", "FinalAnswerText"}
            missing = required_cols - set(ann_df.columns)
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                st.session_state.annotation_df = ann_df
                st.success(f"Loaded {len(ann_df)} responses.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if st.session_state.annotation_df is not None:
        ann_df = st.session_state.annotation_df
        n_total = len(ann_df)
        n_scored = len(st.session_state.annotation_scores)
        idx = st.session_state.annotation_idx

        # Progress
        progress_pct = n_scored / n_total if n_total > 0 else 0
        st.progress(progress_pct, text=f"Scored {n_scored} of {n_total} responses")

        # Navigation
        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 3, 1])
        with nav_col1:
            if st.button("⬅ Previous") and idx > 0:
                st.session_state.annotation_idx -= 1
                st.rerun()
        with nav_col2:
            if st.button("Next ➡") and idx < n_total - 1:
                st.session_state.annotation_idx += 1
                st.rerun()
        with nav_col3:
            jump_to = st.selectbox(
                "Jump to response",
                options=list(range(n_total)),
                index=idx,
                format_func=lambda i: f"{ann_df.iloc[i]['QueryID']} ({i+1}/{n_total})",
            )
            if jump_to != idx:
                st.session_state.annotation_idx = jump_to
                st.rerun()
        with nav_col4:
            scored_flag = "✓ Scored" if ann_df.iloc[idx]["QueryID"] in st.session_state.annotation_scores else "○ Unscored"
            st.markdown(f"**{scored_flag}**")

        st.markdown("---")

        # Response display
        row = ann_df.iloc[idx]
        qid = row["QueryID"]
        prior = st.session_state.annotation_scores.get(qid, {})

        st.markdown(f"**Response {idx+1}/{n_total} — {qid}** · *{row['Category']}*")
        st.markdown(f"**Query:** {row['QueryText']}")

        with st.expander("📄 Response Text", expanded=True):
            st.markdown(row["FinalAnswerText"])

        st.markdown("---")
        st.subheader("Score this response")
        st.caption(
            "Score each dimension 0–1 in 0.25 steps. "
            "**FA** = factual accuracy. **CA** = citation accuracy. "
            "**HD** = historiographical depth. **EC** = epistemic calibration."
        )

        score_cols = st.columns(4)
        dim_descs = {
            "FA": "Factual Accuracy\n(Is the factual content correct?)",
            "CA": "Citation Accuracy\n(Are sources cited and attributable?)",
            "HD": "Historiographical Depth\n(Does it engage historiographically?)",
            "EC": "Epistemic Calibration\n(Does it acknowledge limits honestly?)",
        }
        dim_defaults = {d: prior.get(d, 0.75) for d in ["FA", "CA", "HD", "EC"]}

        score_vals = {}
        for col, dim in zip(score_cols, ["FA", "CA", "HD", "EC"]):
            with col:
                score_vals[dim] = st.select_slider(
                    dim_descs[dim],
                    options=[0.0, 0.25, 0.50, 0.75, 1.0],
                    value=dim_defaults[dim],
                    key=f"slider_{qid}_{dim}",
                )

        notes = st.text_area(
            "Notes (optional)",
            value=prior.get("notes", ""),
            height=80,
            key=f"notes_{qid}",
        )

        save_col, clear_col = st.columns([2, 1])
        with save_col:
            if st.button("💾 Save Score", type="primary"):
                total_score = sum(score_vals[d] for d in ["FA", "CA", "HD", "EC"])
                st.session_state.annotation_scores[qid] = {
                    **score_vals,
                    "Total": total_score,
                    "notes": notes,
                    "timestamp": datetime.now().isoformat(),
                }
                st.success(f"Saved: {qid} → Total={total_score:.2f}")
                # Auto-advance
                if idx < n_total - 1:
                    st.session_state.annotation_idx += 1
                    st.rerun()

        with clear_col:
            if st.button("🗑 Clear Score") and qid in st.session_state.annotation_scores:
                del st.session_state.annotation_scores[qid]
                st.rerun()

        # Export
        st.markdown("---")
        st.subheader("Export Scores")
        if st.session_state.annotation_scores:
            export_rows = []
            for scored_qid, scores in st.session_state.annotation_scores.items():
                row_match = ann_df[ann_df["QueryID"] == scored_qid]
                if not row_match.empty:
                    export_rows.append({
                        "QueryID": scored_qid,
                        "Category": row_match.iloc[0]["Category"],
                        "HumanFA": scores["FA"],
                        "HumanCA": scores["CA"],
                        "HumanHD": scores["HD"],
                        "HumanEC": scores["EC"],
                        "HumanTotal": scores["Total"],
                        "Notes": scores.get("notes", ""),
                        "Timestamp": scores.get("timestamp", ""),
                    })
            export_df = pd.DataFrame(export_rows)
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"⬇ Download Human Scores CSV ({len(export_rows)} responses)",
                data=csv_bytes,
                file_name=f"human_annotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

            # Summary comparison (if LLM scores also loaded)
            if using_live_data and "runs" in loaded:
                st.subheader("Human vs. LLM Score Comparison")
                llm_df = loaded["runs"]
                rubric_col = "RubricTotal" if "RubricTotal" in llm_df.columns else None
                if "QueryID" in llm_df.columns and rubric_col:
                    llm_means = llm_df.groupby("QueryID")[rubric_col].mean().reset_index()
                    llm_means.columns = ["QueryID", "LLM_Mean"]
                    comp_df = export_df.merge(llm_means, on="QueryID", how="inner")
                    if not comp_df.empty:
                        comp_df["Delta"] = comp_df["HumanTotal"] - comp_df["LLM_Mean"]
                        st.dataframe(
                            comp_df[["QueryID", "HumanTotal", "LLM_Mean", "Delta"]].style.format({
                                "HumanTotal": "{:.2f}", "LLM_Mean": "{:.2f}", "Delta": "{:+.2f}",
                            }).background_gradient(subset=["Delta"], cmap="RdBu", vmin=-1, vmax=1),
                            use_container_width=True,
                        )
                        correction_rate = (comp_df["Delta"].abs() > 0.24).mean()
                        mean_delta = comp_df["Delta"].mean()
                        st.metric("Correction Rate (|delta| > 0.25)", f"{correction_rate:.1%}")
                        st.metric("Mean Delta (Human − LLM)", f"{mean_delta:+.3f}")
        else:
            st.caption("No scores saved yet.")

    else:
        st.markdown("""
        **To use this panel:**
        1. Upload a CSV with columns: `QueryID`, `Category`, `QueryText`, `FinalAnswerText`
        2. Score responses one at a time without prior knowledge of LLM scores
        3. Export the scored CSV when complete

        The exported file can be fed directly into the bootstrap pipeline for comparison with automated scores.

        **Blind annotation protocol:** Strip model configuration labels before your session.
        Score before reviewing aggregate benchmark results to prevent anchoring bias.
        """)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "Nicolay Benchmark Viewer · H4N4 ada-002 · 886-chunk corpus · rerank-v4.0-pro · k=5  \n"
    "Canonical result: **2.883 [2.802, 2.961]** (95% CI, bootstrap n=1,000, 5 runs, n=120 obs)  \n"
    "Data: 2026-03-28 · For *Digital Humanities Quarterly* (forthcoming)"
)
