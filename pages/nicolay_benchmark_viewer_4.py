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

# ---------------------------------------------------------------------------
# Pure-python cell coloring — replaces pandas background_gradient()
# which requires matplotlib (not available on Streamlit Cloud by default).
# ---------------------------------------------------------------------------

def _hex_gradient(val, vmin, vmax, low_rgb, high_rgb):
    """Return an inline CSS background color string for a single scalar value."""
    if val is None or (hasattr(val, '__class__') and val.__class__.__name__ == 'float' and val != val):
        return ""
    try:
        t = max(0.0, min(1.0, (float(val) - vmin) / (vmax - vmin)))
    except (TypeError, ZeroDivisionError):
        return ""
    r = int(low_rgb[0] + t * (high_rgb[0] - low_rgb[0]))
    g = int(low_rgb[1] + t * (high_rgb[1] - low_rgb[1]))
    b = int(low_rgb[2] + t * (high_rgb[2] - low_rgb[2]))
    # pick text color for legibility
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    txt = "#000000" if lum > 140 else "#ffffff"
    return f"background-color: rgba({r},{g},{b},0.85); color: {txt}"

def _apply_gradient(series, vmin, vmax, low_rgb, high_rgb):
    return [_hex_gradient(v, vmin, vmax, low_rgb, high_rgb) for v in series]

# Preset palettes  (low_rgb, high_rgb)
_RdYlGn  = ((215, 48,  39),  (26,  152, 80))   # red  → green
_Oranges  = ((255, 245, 235), (127, 39,  4))    # light → dark orange
_RdBu     = ((178, 24,  43),  (33,  102, 172))  # red  → blue
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

# Full query registry — text, category, expected types, ideal docs, named role
QUERY_REGISTRY = [
    {"id":"Q1",  "category":"factual_retrieval",    "hay":"A","nic":"T1",
     "query":"Lincoln noted how many voters from Kansas and Nevada participated in the 1864 election",
     "ideal_docs":[413,414], "missing":None,
     "named":"Pipeline Resilience / R@5=1.00"},
    {"id":"Q2",  "category":"factual_retrieval",    "hay":"D","nic":"T3",
     "query":"How does Russia factor into Lincoln's speeches?",
     "ideal_docs":[305,351,381], "missing":"Eduard de Stoeckl, Alaska purchase negotiations",
     "named":None},
    {"id":"Q3",  "category":"factual_retrieval",    "hay":"D","nic":"T2",
     "query":"In what ways did Lincoln highlight the contributions of immigrants during the Civil War?",
     "ideal_docs":[390,349,350], "missing":None,
     "named":"CA Fabrication Exhibit (R4)"},
    {"id":"Q4",  "category":"analysis",             "hay":"A","nic":"T2",
     "query":"How did Lincoln incorporate allusions in his Second Inaugural Address?",
     "ideal_docs":[419,420,421,422], "missing":None, "named":None},
    {"id":"Q5",  "category":"analysis",             "hay":"D","nic":"T2",
     "query":"How did Lincoln characterize the implications of major Supreme Court decisions before the Civil War?",
     "ideal_docs":[88,95,101], "missing":None, "named":None},
    {"id":"Q7",  "category":"comparative_analysis", "hay":"E","nic":"T4",
     "query":"How did Lincoln's discussion of slavery evolve between his House Divided speech and his Second Inaugural Address?",
     "ideal_docs":[88,95,101,419,420,421,422], "missing":"Lincoln-Douglas Debates — watch whether retrieved",
     "named":"Embedding Model Exhibit / HD=1.00 Anchor"},
    {"id":"Q8",  "category":"comparative_analysis", "hay":"D","nic":"T4",
     "query":"How did Lincoln's justification for the Civil War evolve between his First Inaugural and Second Inaugural?",
     "ideal_docs":[185,191,197,202,419,420,421,422], "missing":None, "named":None},
    {"id":"Q9",  "category":"comparative_analysis", "hay":"D","nic":"T4",
     "query":"How did Lincoln's views of African American soldiers change or remain the same over time?",
     "ideal_docs":[288,295,367,374], "missing":"Executive orders on Black troop pay",
     "named":None},
    {"id":"Q10", "category":"synthesis",            "hay":"D","nic":"T4",
     "query":"How did Lincoln develop the theme of divine providence throughout his wartime speeches?",
     "ideal_docs":[298,418,419,420,421,422], "missing":"Thanksgiving proclamations (not in corpus)",
     "named":None},
    {"id":"Q11", "category":"synthesis",            "hay":"D","nic":"T5",
     "query":"How did Lincoln consistently frame the relationship between liberty and law?",
     "ideal_docs":[153,159,185,191,418,419], "missing":None,
     "named":"HD Ceiling Floor / Synthesis Failure"},
    {"id":"Q12", "category":"synthesis",            "hay":"D","nic":"T5",
     "query":"What themes did Lincoln consistently employ when discussing the Constitution's relationship to slavery?",
     "ideal_docs":[153,159,185,191], "missing":None, "named":None},
    {"id":"Q13", "category":"race_citizenship",     "hay":"E","nic":"T5",
     "query":"How did Lincoln's views on African American citizenship and racial equality evolve across his speeches?",
     "ideal_docs":[288,295,367,374,413,414,419], "missing":"Last Public Address Apr 11 1865 — NOT IN CORPUS",
     "named":"Parametric Laundering Canonical Case"},
    {"id":"R1",  "category":"analysis",             "hay":"D","nic":"T3",
     "query":"How did Lincoln justify the naval blockade of Confederate ports?",
     "ideal_docs":[218,272,300,345,359], "missing":"Trent Affair — NOT IN CORPUS", "named":None},
    {"id":"R2",  "category":"comparative_analysis", "hay":"D","nic":"T4",
     "query":"How did Lincoln describe U.S. relations with Great Britain during the Civil War?",
     "ideal_docs":[242,243,247,300,301,345,346,388], "missing":"Trent Affair — NOT IN CORPUS", "named":None},
    {"id":"R3",  "category":"factual_retrieval",    "hay":"A","nic":"T1",
     "query":"How did Lincoln report on the financial condition of the Post Office Department during the war?",
     "ideal_docs":[311,312,364,365,401], "missing":None, "named":None},
    {"id":"AN-5","category":"analysis",             "hay":"A","nic":"T2",
     "query":"How did Lincoln develop the labor-capital argument in his First Annual Message?",
     "ideal_docs":[279,280,281], "missing":None, "named":None},
    {"id":"CA-5","category":"comparative_analysis", "hay":"E","nic":"T3",
     "query":"How did Lincoln's tone toward the South differ between his First and Second Inaugural Addresses?",
     "ideal_docs":[193,195,420,421,422], "missing":None,
     "named":"Displaced Quote (R0/R1) / Upgrade"},
    {"id":"CA-6","category":"comparative_analysis", "hay":"D","nic":"T3",
     "query":"How did Lincoln justify the suspension of habeas corpus during the Civil War?",
     "ideal_docs":[214,219,221,380], "missing":None, "named":None},
    {"id":"FR-2","category":"factual_retrieval",    "hay":"D","nic":"T3",
     "query":"How did Lincoln address wartime taxation, debt, and civic obligation in his Annual Messages?",
     "ideal_docs":[249,309,310,393,395], "missing":None,
     "named":"Flat-Reranker Pattern"},
    {"id":"S-4", "category":"synthesis",            "hay":"E","nic":"T4",
     "query":"How did Lincoln use the Declaration of Independence as a rhetorical framework across his career?",
     "ideal_docs":[44,45,418,624,626,628], "missing":None,
     "named":"Calibration Decoupling Exhibit"},
    {"id":"S-5", "category":"synthesis",            "hay":"D","nic":"T4",
     "query":"How did Lincoln frame self-government as a test of democratic viability?",
     "ideal_docs":[46,47,48,239,418], "missing":None, "named":None},
    {"id":"RC-3","category":"race_citizenship",     "hay":"E","nic":"T3",
     "query":"How did Lincoln distinguish between natural rights and political equality in 1858?",
     "ideal_docs":[41,481,550,624,679], "missing":None,
     "named":"HD Upgrade Mechanism (T3→T4)"},
    {"id":"RC-4","category":"race_citizenship",     "hay":"D","nic":"T3",
     "query":"How did Lincoln link emancipation, Black military service, and the future status of freed people?",
     "ideal_docs":[293,295,372,374], "missing":None,
     "named":"Zero-Variance Reproducibility"},
    {"id":"RC-5","category":"race_citizenship",     "hay":"E","nic":"T4",
     "query":"What did Lincoln leave unresolved about the future political status of freed people?",
     "ideal_docs":[297,375,376,378,410,416], "missing":None, "named":None},
]
QUERY_REGISTRY_BY_ID = {q["id"]: q for q in QUERY_REGISTRY}


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

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔎 Question Browser",
    "📊 Grand Summary",
    "📂 Category Performance",
    "🔥 Per-Query Heatmap",
    "🔍 Retrieval Architecture",
    "📐 HD vs EC / Scatter",
    "✏️ Human Annotation",
])

# ═════════════════════════════════════════════════════════════════════════════

# =============================================================================
# TAB 0: QUESTION BROWSER
# =============================================================================

with tab0:
    st.header("Question Browser")
    st.caption(
        "Browse all 24 benchmark questions by category. Select a question to inspect "
        "the full pipeline output for any individual run."
    )

    all_cats = ["All categories"] + sorted(set(q["category"] for q in QUERY_REGISTRY))
    cat_labels = {
        "All categories": "All categories",
        "factual_retrieval":    "Factual Retrieval",
        "analysis":             "Analysis",
        "comparative_analysis": "Comparative Analysis",
        "synthesis":            "Synthesis",
        "race_citizenship":     "Race & Citizenship",
    }
    selected_cat = st.selectbox(
        "Filter by category",
        options=all_cats,
        format_func=lambda c: cat_labels.get(c, c),
    )

    filtered_qs = [
        q for q in QUERY_REGISTRY
        if selected_cat == "All categories" or q["category"] == selected_cat
    ]

    def _q_label(q):
        star = "★ " if q.get("named") else "  "
        return f"{star}{q['id']}  —  {q['query'][:80]}{'…' if len(q['query'])>80 else ''}"

    selected_q = st.selectbox(
        "Select question",
        options=filtered_qs,
        format_func=_q_label,
    )

    if selected_q:
        qid = selected_q["id"]
        qmeta = QUERY_REGISTRY_BY_ID[qid]

        named_badge = (
            f"<span style='background:#b87333;color:#fff;padding:2px 8px;"
            f"border-radius:10px;font-size:11px;margin-left:8px;'>"
            f"★ {qmeta['named']}</span>"
            if qmeta.get("named") else ""
        )
        cat_color = CAT_COLORS.get(qmeta["category"], "#888")

        st.markdown(
            f"<div style='border-left:4px solid {cat_color};padding:10px 16px;"
            f"background:rgba(0,0,0,0.04);border-radius:0 8px 8px 0;margin-bottom:12px;'>"
            f"<div style='font-size:18px;font-weight:700;'>{qid}{named_badge}</div>"
            f"<div style='font-size:13px;color:#888;margin:2px 0 8px;'>"
            f"{cat_labels.get(qmeta['category'], qmeta['category'])}</div>"
            f"<div style='font-size:15px;font-style:italic;'>\"{qmeta['query']}\"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
        with meta_col1:
            st.metric("Expected Hay Type", qmeta["hay"])
        with meta_col2:
            st.metric("Expected Nicolay Type", qmeta["nic"])
        with meta_col3:
            st.metric("Ideal Docs", len(qmeta["ideal_docs"]))
        with meta_col4:
            q_row = next((r for r in PER_QUERY_DATA if r["QID"] == qid), None)
            if q_row:
                st.metric("Mean Total (5 runs)", f"{q_row['Total']:.2f}")

        if qmeta.get("missing"):
            st.warning(f"**Corpus gap:** {qmeta['missing']}")

        with st.expander("Ideal document IDs", expanded=False):
            st.code(", ".join(str(d) for d in qmeta["ideal_docs"]))

        if q_row:
            st.markdown("---")
            st.subheader("Aggregate Results (5-run mean)")
            r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns(5)
            for col, dim, label in zip(
                [r_col1, r_col2, r_col3, r_col4, r_col5],
                ["Total", "FA", "CA", "HD", "EC"],
                ["Total", "Fact. Acc.", "Cite. Acc.", "Hist. Depth", "Epist. Cal."],
            ):
                col.metric(label, f"{q_row[dim]:.2f}", delta=f"+-{q_row['SD']:.3f}" if dim == "Total" else None)

            run_scores = HEATMAP_DATA.get(qid, [])
            if run_scores:
                fig_mini = go.Figure(go.Heatmap(
                    z=[run_scores],
                    x=["Run 0","Run 1","Run 2","Run 3","Run 4"],
                    y=[qid],
                    colorscale=[[0,"#d73027"],[0.375,"#fc8d59"],[0.625,"#4575b4"],[1,"#1a6b3c"]],
                    zmin=2.0, zmax=4.0,
                    text=[[f"{v:.2f}" for v in run_scores]],
                    texttemplate="%{text}",
                    showscale=False,
                ))
                fig_mini.update_layout(
                    height=100,
                    margin=dict(l=60, r=20, t=10, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_mini, use_container_width=True)

            ret_col1, ret_col2, ret_col3 = st.columns(3)
            ret_col1.metric("R@5", f"{q_row['R@5']:.3f}")
            ret_col2.metric("Keyword slots", f"{q_row['KW']:.1f} / 5")
            ret_col3.metric("Semantic slots", f"{q_row['Sem']:.1f} / 5")
            if qid in FLAT_RERANKER_QIDS:
                st.warning(
                    f"**Flat-reranker case** — spread: {q_row['Rerank']:.3f} (below 0.05 threshold). "
                    "Confident retrieval in wrong semantic neighborhood."
                )

        st.markdown("---")
        st.subheader("Individual Run Inspection")

        if not using_live_data:
            st.info(
                "Load live data from GitHub (sidebar button) to inspect individual run outputs: "
                "Hay InitialAnswer, QueryAssessment, NicolaySynthesisAssessmentRaw, "
                "FinalAnswerText, retrieval composition, quote verification, and LLM rubric scores."
            )
        else:
            run_df = loaded["runs"]
            q_runs = run_df[run_df["QueryID"] == qid].copy()

            if q_runs.empty:
                st.warning(f"No rows found for {qid} in the loaded data.")
            else:
                run_nums = sorted(q_runs["run"].unique())
                selected_run = st.selectbox(
                    "Select run",
                    options=run_nums,
                    format_func=lambda r: f"Run {r}",
                    key="browser_run_select",
                )
                row = q_runs[q_runs["run"] == selected_run].iloc[0]

                def _get(col, default="—"):
                    v = row.get(col, "")
                    s = str(v).strip()
                    return s if s not in ("", "nan") else default

                pip_col1, pip_col2, pip_col3, pip_col4 = st.columns(4)
                pip_col1.metric("Hay Type", f"{_get('HayTypeGot')} (exp {_get('HayTypeExpected')})")
                pip_col2.metric("Nicolay Type", f"{_get('NicolayTypeGot')} (exp {_get('NicolayTypeExpected')})")
                pip_col3.metric("R@5", _get("R@5"))
                pip_col4.metric("RubricTotal", _get("RubricTotal"))

                dim_cols = st.columns(4)
                for dc, (col, lbl) in zip(dim_cols, [("FA","Fact. Acc."),("CA","Cite. Acc."),("HD","Hist. Depth"),("EC","Epist. Cal.")]):
                    dc.metric(lbl, _get(col))

                with st.expander("Hay Layer — InitialAnswer & QueryAssessment", expanded=True):
                    hay_correct = _get("HayTypeCorrect")
                    hay_color = "#2d6a4f" if hay_correct == "True" else "#c1121f"
                    st.markdown(
                        f"<div style='font-size:12px;color:{hay_color};font-weight:600;margin-bottom:6px;'>"
                        f"Type: {_get('HayTypeGot')} (expected {_get('HayTypeExpected')}) — "
                        f"{'Correct' if hay_correct=='True' else 'Incorrect'}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**InitialAnswer** *(seeds retrieval)*")
                    st.markdown(
                        f"<div style='background:#eef7ee;padding:10px;border-radius:6px;"
                        f"font-size:13px;border-left:3px solid #52b788;'>{_get('InitialAnswer')}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**QueryAssessment** *(Hay self-assessment)*")
                    st.markdown(
                        f"<div style='background:#f8f4ef;padding:10px;border-radius:6px;"
                        f"font-size:13px;border-left:3px solid #cca855;'>{_get('QueryAssessment')}</div>",
                        unsafe_allow_html=True,
                    )
                    st.metric("Keywords generated", _get("HayKeywordCount"))

                with st.expander("Retrieval — Composition & Scoring", expanded=True):
                    ret_a, ret_b, ret_c, ret_d = st.columns(4)
                    ret_a.metric("KW slots", _get("KW"))
                    ret_b.metric("Sem slots", _get("Sem"))
                    rerank_val = _get("Rerank")
                    ret_c.metric("Reranker spread", f"{float(rerank_val):.3f}" if rerank_val != "—" else "—")
                    ret_d.metric("Precision@5", _get("PrecisionAt5"))
                    st.markdown("**Retrieved doc IDs (all candidates)**")
                    st.code(_get("RetrievedDocIDs"))
                    st.markdown("**Top-5 retrieval path**")
                    st.code(_get("RetrievalPathTop5"))
                    hit_col, miss_col = st.columns(2)
                    hit_col.markdown(f"Ideal docs hit: `{_get('IdealDocsHit')}`")
                    miss_col.markdown(f"Ideal docs missed: `{_get('IdealDocsMissed')}`")

                with st.expander("Nicolay Layer — Synthesis & Final Answer", expanded=True):
                    nic_correct = _get("NicolayTypeCorrect")
                    nic_color = "#2d6a4f" if nic_correct == "True" else "#c1121f"
                    st.markdown(
                        f"<div style='font-size:12px;color:{nic_color};font-weight:600;margin-bottom:6px;'>"
                        f"Type: {_get('NicolayTypeGot')} (expected {_get('NicolayTypeExpected')}) — "
                        f"{'Correct' if nic_correct=='True' else 'Incorrect'}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**NicolaySynthesisAssessmentRaw** *(chain-of-thought)*")
                    st.markdown(
                        f"<div style='background:#e8f0f8;padding:10px;border-radius:6px;"
                        f"font-size:13px;border-left:3px solid #4c78a8;white-space:pre-wrap;'>"
                        f"{_get('NicolaySynthesisAssessmentRaw')}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**FinalAnswerText** *(evaluated output)*")
                    st.markdown(
                        f"<div style='background:#f8f4ef;padding:10px;border-radius:6px;"
                        f"font-size:14px;border-left:3px solid #b87333;'>"
                        f"{_get('FinalAnswerText')}</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Word count: {_get('FinalAnswerWordCount')}")

                with st.expander("Quote Verification", expanded=False):
                    qv_cols_b = st.columns(5)
                    for qvc, label, col in zip(
                        qv_cols_b,
                        ["Verified","Approx","Displaced","Fabricated","Mislabeled"],
                        ["QuotesVerified","QuotesApprox","QuotesDisplaced","QuotesFabricated","QuotesMislabeled"],
                    ):
                        qvc.metric(label, _get(col, "0"))
                    conf_c1, conf_c2, conf_c3 = st.columns(3)
                    conf_c1.metric("Confidence Rating", _get("ConfidenceRating"))
                    conf_c2.metric("ROUGE-1 max retrieved", _get("Rouge1MaxRetrieved"))
                    conf_c3.metric("Calib. Warning", _get("ConfidenceCalibWarning"))

                with st.expander("LLM Rubric Scores & Rationales", expanded=True):
                    score_c = st.columns(5)
                    for sc, (dim_col, label) in zip(score_c, [
                        ("FA","Fact. Acc."),("CA","Cite. Acc."),
                        ("HD","Hist. Depth"),("EC","Epist. Cal."),("RubricTotal","Total"),
                    ]):
                        sc.metric(label, _get(dim_col))

                    for rat_col, label, bg in [
                        ("RationaleFactualAccuracy",        "Factual Accuracy Rationale",        "#eef7ee"),
                        ("RationaleCitationAccuracy",       "Citation Accuracy Rationale",       "#e8f0f8"),
                        ("RationaleHistoriographicalDepth", "Historiographical Depth Rationale", "#f8f4ef"),
                        ("RationaleEpistemicCalibration",   "Epistemic Calibration Rationale",   "#fef9ec"),
                        ("RationaleHayDiagnostic",          "Hay Diagnostic",                    "#fff3cd"),
                    ]:
                        val = _get(rat_col)
                        if val != "—":
                            st.markdown(f"**{label}**")
                            st.markdown(
                                f"<div style='background:{bg};padding:8px;border-radius:6px;"
                                f"font-size:13px;margin-bottom:6px;'>{val}</div>",
                                unsafe_allow_html=True,
                            )

                ev_notes = _get("EvaluatorNotes")
                if ev_notes != "—":
                    st.info(f"**Evaluator Notes:** {ev_notes}")

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
        marker_color=[
            "rgba({},{},{},0.5)".format(
                int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
            ) for c in colors_cat
        ],
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
        }).apply(_apply_gradient, vmin=2.4, vmax=3.4, low_rgb=_RdYlGn[0], high_rgb=_RdYlGn[1], subset=["Total"]),
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
        }).apply(_apply_gradient, vmin=2.0, vmax=4.0, low_rgb=_RdYlGn[0], high_rgb=_RdYlGn[1], subset=["Total"])
          .apply(_apply_gradient, vmin=0, vmax=0.55, low_rgb=_Oranges[0], high_rgb=_Oranges[1], subset=["SD"]),
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
                       .apply(_apply_gradient, vmin=0, vmax=1, low_rgb=_RdYlGn[0], high_rgb=_RdYlGn[1], subset=["Accuracy"]),
            use_container_width=True,
            height=220,
        )
        st.caption(
            "T3 is the structural black hole (8.9% accuracy). "
            "Upgrade trigger = retrieval ambiguity, not query recognition. "
            "Nicolay has a functional 3-tier range: T1, T2, T4 — with T3 and T5 effectively absent."
        )

# ═════════════════════════════════════════════════════════════════════════════
# =============================================================================
# TAB 6: HUMAN ANNOTATION PANEL
# =============================================================================

with tab6:
    st.header('Human Annotation Panel -- Blind Rubric Scoring')
    st.caption(
        'Implements the blind annotation protocol from the article methodology. '
        'Upload any merged_run_N.csv. Each response is presented with full pipeline '
        'context so you can score exactly what Claude saw during auto-scoring.'
    )

    if 'annotation_scores' not in st.session_state:
        st.session_state.annotation_scores = {}
    if 'annotation_idx' not in st.session_state:
        st.session_state.annotation_idx = 0
    if 'annotation_df' not in st.session_state:
        st.session_state.annotation_df = None

    upload = st.file_uploader(
        'Upload benchmark CSV (merged_run_N.csv)',
        type=['csv'],
        help='Any merged_run_N.csv works as-is. Columns normalized automatically.',
    )

    if upload:
        try:
            ann_df = pd.read_csv(upload)
            if 'Query' in ann_df.columns and 'QueryText' not in ann_df.columns:
                ann_df = ann_df.rename(columns={'Query': 'QueryText'})
            required_cols = {'QueryID', 'Category', 'QueryText', 'FinalAnswerText'}
            missing_cols = required_cols - set(ann_df.columns)
            if missing_cols:
                st.error(f'Missing required columns: {missing_cols}')
            else:
                st.session_state.annotation_df = ann_df
                st.success(f'Loaded {len(ann_df)} responses.')
        except Exception as e:
            st.error(f'Could not read CSV: {e}')

    if st.session_state.annotation_df is not None:
        ann_df = st.session_state.annotation_df
        n_total = len(ann_df)
        n_scored = len(st.session_state.annotation_scores)
        idx = st.session_state.annotation_idx

        st.progress(n_scored / n_total if n_total > 0 else 0,
                    text=f'Scored {n_scored} of {n_total} responses')

        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 3, 1])
        with nav_col1:
            if st.button('Prev', key='ann_prev') and idx > 0:
                st.session_state.annotation_idx -= 1
                st.rerun()
        with nav_col2:
            if st.button('Next', key='ann_next') and idx < n_total - 1:
                st.session_state.annotation_idx += 1
                st.rerun()
        with nav_col3:
            jump_to = st.selectbox(
                'Jump to', options=list(range(n_total)), index=idx,
                format_func=lambda i: f"{ann_df.iloc[i]['QueryID']} ({i+1}/{n_total})",
                key='ann_jump',
            )
            if jump_to != idx:
                st.session_state.annotation_idx = jump_to
                st.rerun()
        with nav_col4:
            row_qid_nav = ann_df.iloc[idx]['QueryID']
            st.markdown('**Scored**' if row_qid_nav in st.session_state.annotation_scores else '**Unscored**')

        st.markdown('---')
        ann_row = ann_df.iloc[idx]
        qid = ann_row['QueryID']
        prior = st.session_state.annotation_scores.get(qid, {})

        def _ann_get(col, default='--'):
            v = ann_row.get(col, '')
            s = str(v).strip()
            return s if s not in ('', 'nan') else default

        qmeta_ann = QUERY_REGISTRY_BY_ID.get(qid, {})
        cat_color_ann = CAT_COLORS.get(ann_row.get('Category', ''), '#888')
        named_ann = qmeta_ann.get('named', '') if qmeta_ann else ''
        named_badge_ann = (
            f"<span style='background:#b87333;color:#fff;padding:2px 8px;"
            f"border-radius:10px;font-size:11px;margin-left:8px;'>* {named_ann}</span>"
            if named_ann else ''
        )
        st.markdown(
            f"<div style='border-left:4px solid {cat_color_ann};padding:8px 14px;"
            f"background:rgba(0,0,0,0.04);border-radius:0 8px 8px 0;margin-bottom:10px;'>"
            f"<b style='font-size:16px;'>{qid}{named_badge_ann}</b>"
            f"<span style='font-size:12px;color:#888;margin-left:10px;'>"
            f"{ann_row.get('Category','').replace('_',' ').title()}</span><br>"
            f"<i style='font-size:14px;'>{ann_row.get('QueryText','')}</i>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.subheader('Pipeline Context')
        st.caption(
            'Evidentiary hierarchy (same as automated rubric): '
            '(1) FinalAnswerText is primary for FA and CA; '
            '(2) SynthesisAssessmentRaw is primary for HD and EC; '
            '(3) InitialAnswer is diagnostic only.'
        )

        with st.expander('1. Hay Layer -- InitialAnswer (diagnostic only)', expanded=True):
            hay_got = _ann_get('HayTypeGot')
            hay_exp = _ann_get('HayTypeExpected')
            hay_ok  = _ann_get('HayTypeCorrect')
            hay_col_c = '#2d6a4f' if hay_ok == 'True' else '#c1121f'
            st.markdown(
                f"<div style='font-size:12px;color:{hay_col_c};font-weight:600;margin-bottom:4px;'>"
                f"Hay type: {hay_got} (expected {hay_exp}) -- "
                f"{'Correct' if hay_ok=='True' else 'Incorrect'} | "
                f"Keywords: {_ann_get('HayKeywordCount')}</div>"
                f"<div style='font-size:11px;color:#666;margin-bottom:6px;'>"
                f"If FinalAnswerText independently corrects a Hay error, do not penalize FA/HD.</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='background:#eef7ee;padding:9px;border-radius:6px;"
                f"font-size:13px;border-left:3px solid #52b788;'>"
                f"<b>InitialAnswer:</b> {_ann_get('InitialAnswer')}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='background:#f8f4ef;padding:8px;border-radius:6px;"
                f"font-size:12px;border-left:3px solid #cca855;margin-top:5px;'>"
                f"<b>QueryAssessment:</b> {_ann_get('QueryAssessment')}</div>",
                unsafe_allow_html=True,
            )

        with st.expander('2. Retrieval Context', expanded=False):
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric('R@5', _ann_get('RecallAt5'))
            rc2.metric('P@5', _ann_get('PrecisionAt5'))
            rc3.metric('KW slots', _ann_get('RetrievalKeywordCountTop5'))
            rc4.metric('Sem slots', _ann_get('RetrievalSemanticCountTop5'))
            st.code(_ann_get('RetrievalPathTop5'))
            h2c, m2c = st.columns(2)
            h2c.markdown(f"Ideal docs hit: `{_ann_get('IdealDocsHit')}`")
            m2c.markdown(f"Ideal docs missed: `{_ann_get('IdealDocsMissed')}`")
            if qmeta_ann and qmeta_ann.get('missing'):
                st.warning(f"Corpus gap: {qmeta_ann['missing']}")

        with st.expander('3. SynthesisAssessmentRaw -- primary for HD and EC scoring', expanded=True):
            nic_got = _ann_get('NicolayTypeGot')
            nic_exp = _ann_get('NicolayTypeExpected')
            nic_ok  = _ann_get('NicolayTypeCorrect')
            nic_col_c = '#2d6a4f' if nic_ok == 'True' else '#c1121f'
            st.markdown(
                f"<div style='font-size:12px;color:{nic_col_c};font-weight:600;margin-bottom:4px;'>"
                f"Nicolay type: {nic_got} (expected {nic_exp}) -- "
                f"{'Correct' if nic_ok=='True' else 'Incorrect'}</div>"
                f"<div style='font-size:11px;color:#666;margin-bottom:6px;'>"
                f"Score HD on historiographical engagement. Score EC on uncertainty acknowledgment.</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='background:#e8f0f8;padding:10px;border-radius:6px;"
                f"font-size:13px;border-left:3px solid #4c78a8;white-space:pre-wrap;'>"
                f"{_ann_get('NicolaySynthesisAssessmentRaw')}</div>",
                unsafe_allow_html=True,
            )

        with st.expander('4. FinalAnswerText -- primary for FA and CA scoring', expanded=True):
            st.markdown(
                "<div style='font-size:11px;color:#666;margin-bottom:6px;'>"
                "Score FA on factual correctness. Score CA on citation attribution. "
                "Hard rule: QuotesFabricated >= 1 caps CA at 0.50.</div>",
                unsafe_allow_html=True,
            )
            qv_cols_ann = st.columns(5)
            for qvc, lbl, col in zip(
                qv_cols_ann,
                ['Verified','Approx','Displaced','Fabricated','Mislabeled'],
                ['QuotesVerified','QuotesApprox','QuotesDisplaced','QuotesFabricated','QuotesMislabeled'],
            ):
                val = _ann_get(col, '0')
                color = '#c1121f' if lbl in ('Fabricated','Displaced') and val not in ('0','--') else '#333'
                qvc.markdown(
                    f"<div style='text-align:center;'>"
                    f"<div style='font-size:20px;font-weight:700;color:{color};'>{val}</div>"
                    f"<div style='font-size:11px;color:#888;'>{lbl}</div></div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div style='background:#f8f4ef;padding:10px;border-radius:6px;"
                f"font-size:14px;border-left:3px solid #b87333;margin-top:8px;'>"
                f"{_ann_get('FinalAnswerText')}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Word count: {_ann_get('FinalAnswerWordCount')}")

        with st.expander('5. LLM Auto-Scores (reveal after scoring to check agreement)', expanded=False):
            st.warning('Blind protocol: score the response above before expanding this section.')
            llm_sc = st.columns(5)
            for sc, (dim, lbl) in zip(llm_sc, [
                ('RubricFactualAccuracy','FA'),
                ('RubricCitationAccuracy','CA'),
                ('RubricHistoriographicalDepth','HD'),
                ('RubricEpistemicCalibration','EC'),
                ('RubricTotal','Total'),
            ]):
                sc.metric(lbl, _ann_get(dim))
            for rat_col, lbl, bg in [
                ('RationaleFactualAccuracy',        'FA Rationale',   '#eef7ee'),
                ('RationaleCitationAccuracy',       'CA Rationale',   '#e8f0f8'),
                ('RationaleHistoriographicalDepth', 'HD Rationale',   '#f8f4ef'),
                ('RationaleEpistemicCalibration',   'EC Rationale',   '#fef9ec'),
                ('RationaleHayDiagnostic',          'Hay Diagnostic', '#fff3cd'),
            ]:
                val = _ann_get(rat_col)
                if val != '--':
                    st.markdown(
                        f"<div style='background:{bg};padding:7px;border-radius:5px;"
                        f"font-size:12px;margin-bottom:5px;'>"
                        f"<b>{lbl}:</b> {val}</div>",
                        unsafe_allow_html=True,
                    )

        st.markdown('---')
        st.subheader('Your Scores')

        dim_guidance = {
            'FA': ('Factual Accuracy', 'Are all specific claims (names, dates, figures) correct?',
                   '1.0 All correct | 0.75 Minor errors | 0.5 Significant errors | 0.25 Major distortions | 0.0 Fabrication'),
            'CA': ('Citation Accuracy', 'Are sources cited and correctly attributed?',
                   '1.0 All verified | 0.75 Minor issues | 0.5 Displaced/unsupported | 0.0 Fabricated | Hard rule: Fabricated>=1 caps at 0.50'),
            'HD': ('Historiographical Depth', 'Does the synthesis engage historiographical complexity? Score from SynthesisAssessmentRaw.',
                   '1.0 Sophisticated | 0.75 Solid framing | 0.5 Descriptive only | 0.25 Superficial | 0.0 No engagement'),
            'EC': ('Epistemic Calibration', 'Does it acknowledge uncertainty and corpus limits? Score from SynthesisAssessmentRaw.',
                   '1.0 Explicitly calibrated | 0.75 Generally appropriate | 0.5 Overconfident in places | 0.0 Systematically overconfident'),
        }

        score_vals = {}
        for dim in ['FA', 'CA', 'HD', 'EC']:
            label, question, rubric = dim_guidance[dim]
            st.markdown(f'**{label}** -- {question}')
            st.caption(rubric)
            score_vals[dim] = st.select_slider(
                f'Score {label}',
                options=[0.0, 0.25, 0.50, 0.75, 1.0],
                value=prior.get(dim, 0.75),
                key=f'ann_slider_{qid}_{dim}',
                label_visibility='collapsed',
            )

        total_score = sum(score_vals[d] for d in ['FA', 'CA', 'HD', 'EC'])
        st.markdown(f'**Running total: {total_score:.2f} / 4.00**')

        notes = st.text_area(
            'Evaluator notes (flag anomalies, overrides, uncertainty)',
            value=prior.get('notes', ''),
            height=70,
            key=f'ann_notes_{qid}',
        )

        save_col, clear_col = st.columns([2, 1])
        with save_col:
            if st.button('Save Score', type='primary', key='ann_save'):
                st.session_state.annotation_scores[qid] = {
                    **score_vals, 'Total': total_score,
                    'notes': notes, 'timestamp': datetime.now().isoformat(),
                }
                st.success(f'Saved: {qid} -- Total {total_score:.2f}/4.00')
                if idx < n_total - 1:
                    st.session_state.annotation_idx += 1
                    st.rerun()
        with clear_col:
            if st.button('Clear Score', key='ann_clear'):
                if qid in st.session_state.annotation_scores:
                    del st.session_state.annotation_scores[qid]
                    st.rerun()

        st.markdown('---')
        st.subheader('Export and Comparison')

        if st.session_state.annotation_scores:
            export_rows = []
            for scored_qid, scores in st.session_state.annotation_scores.items():
                row_match = ann_df[ann_df['QueryID'] == scored_qid]
                if not row_match.empty:
                    llm_fa  = float(row_match.iloc[0].get('RubricFactualAccuracy', 0) or 0)
                    llm_ca  = float(row_match.iloc[0].get('RubricCitationAccuracy', 0) or 0)
                    llm_hd  = float(row_match.iloc[0].get('RubricHistoriographicalDepth', 0) or 0)
                    llm_ec  = float(row_match.iloc[0].get('RubricEpistemicCalibration', 0) or 0)
                    llm_tot = round(llm_fa + llm_ca + llm_hd + llm_ec, 2)
                    h_tot   = scores['Total']
                    export_rows.append({
                        'QueryID': scored_qid,
                        'Category': row_match.iloc[0]['Category'],
                        'HumanFA': scores['FA'], 'HumanCA': scores['CA'],
                        'HumanHD': scores['HD'], 'HumanEC': scores['EC'],
                        'HumanTotal': h_tot,
                        'LLM_FA': llm_fa, 'LLM_CA': llm_ca,
                        'LLM_HD': llm_hd, 'LLM_EC': llm_ec,
                        'LLM_Total': llm_tot,
                        'Delta': round(h_tot - llm_tot, 2),
                        'Notes': scores.get('notes', ''),
                        'Timestamp': scores.get('timestamp', ''),
                    })

            export_df = pd.DataFrame(export_rows)
            csv_bytes = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f'Download Human Scores CSV ({len(export_rows)} responses)',
                data=csv_bytes,
                file_name=f"human_annotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
            )

            if len(export_rows) > 1:
                st.subheader('Human vs. LLM Comparison')
                correction_rate = (export_df['Delta'].abs() > 0.24).mean()
                mean_delta = export_df['Delta'].mean()
                cr1, cr2 = st.columns(2)
                cr1.metric('Correction Rate (|delta| > 0.25)', f'{correction_rate:.1%}')
                cr2.metric('Mean Delta (Human - LLM)', f'{mean_delta:+.3f}')
                dim_delta_cols = st.columns(4)
                for dc, dim in zip(dim_delta_cols, ['FA','CA','HD','EC']):
                    if f'Human{dim}' in export_df.columns and f'LLM_{dim}' in export_df.columns:
                        d = (export_df[f'Human{dim}'] - export_df[f'LLM_{dim}']).mean()
                        dc.metric(dim, f'{d:+.3f}')
                st.dataframe(
                    export_df[['QueryID','HumanTotal','LLM_Total','Delta','Notes']].style.format({
                        'HumanTotal': '{:.2f}', 'LLM_Total': '{:.2f}', 'Delta': '{:+.2f}',
                    }).apply(
                        _apply_gradient, vmin=-1, vmax=1,
                        low_rgb=_RdBu[0], high_rgb=_RdBu[1], subset=['Delta']
                    ),
                    use_container_width=True,
                )
        else:
            st.caption('No scores saved yet.')

    else:
        st.markdown(
            '**To use this panel:**\n'
            '1. Upload any merged_run_N.csv from the benchmark directory.\n'
            '2. Each response shows full pipeline context: InitialAnswer, '
            'SynthesisAssessmentRaw, FinalAnswerText, retrieval metrics, quote counts.\n'
            '3. Score each dimension using the inline rubric guidance.\n'
            '4. Expand LLM Auto-Scores only after scoring to check agreement.\n'
            '5. Export the comparison CSV when complete.\n\n'
            '**Blind annotation protocol:** The LLM scores section is collapsed by default.'
        )

# FOOTER
# -----------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "Nicolay Benchmark Viewer · H4N4 ada-002 · 886-chunk corpus · rerank-v4.0-pro · k=5  \n"
    "Canonical result: **2.883 [2.802, 2.961]** (95% CI, bootstrap n=1,000, 5 runs, n=120 obs)  \n"
    "Data: 2026-03-28 · For *Digital Humanities Quarterly* (forthcoming)"
)
