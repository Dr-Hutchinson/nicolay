"""
nicolay_benchmark_viewer_5.py
HonestAbe Benchmark Viewer — Nicolay RAG System
For: "Nicolay: Exploring Historic Text Collections with Large Language Models
     and Retrieval Augmented Generation"
     Digital Humanities Quarterly (submitted 2026)

Usage:
    streamlit run nicolay_benchmark_viewer_5.py

Data modes:
  EMBEDDED (default): Canonical five-run results hardcoded. Runs immediately.
  LIVE CSV (optional): Click "Load from GitHub" in sidebar to fetch all five
    merged_run_N.csv files and bootstrap_summary_final.csv from:
    https://github.com/Dr-Hutchinson/nicolay/tree/main/benchmark_data/

Tab structure:
  0 — Overview
  1 — Question Browser
  2 — Grand Summary
  3 — Retrieval & Category Performance
  4 — Type Classification
  5 — Response Fidelity & Hallucinations
  6 — Human Annotation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import io
from datetime import datetime

# ---------------------------------------------------------------------------
# Cell coloring — pure Python, no matplotlib dependency
# ---------------------------------------------------------------------------

def _hex_gradient(val, vmin, vmax, low_rgb, high_rgb):
    if val is None:
        return ""
    try:
        t = max(0.0, min(1.0, (float(val) - vmin) / (vmax - vmin)))
    except (TypeError, ZeroDivisionError):
        return ""
    r = int(low_rgb[0] + t * (high_rgb[0] - low_rgb[0]))
    g = int(low_rgb[1] + t * (high_rgb[1] - low_rgb[1]))
    b = int(low_rgb[2] + t * (high_rgb[2] - low_rgb[2]))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    txt = "#000000" if lum > 140 else "#ffffff"
    return f"background-color: rgba({r},{g},{b},0.85); color: {txt}"

def _apply_gradient(series, vmin, vmax, low_rgb, high_rgb):
    return [_hex_gradient(v, vmin, vmax, low_rgb, high_rgb) for v in series]

_RdYlGn = ((215, 48, 39),  (26, 152, 80))
_Oranges = ((255, 245, 235), (127, 39, 4))
_RdBu    = ((178, 24, 43),  (33, 102, 172))

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HonestAbe Benchmark Viewer",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CANONICAL EMBEDDED DATA
# Source: five-run-benchmark-results.md, 2026-03-28
# Configuration: H4N4 ada-002, 886-chunk corpus, rerank-v4.0-pro, k=5
# ---------------------------------------------------------------------------

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

# RC category: per-question type delta and HD/EC for Chart 4
RC_TYPE_DATA = [
    {"QID": "RC-1 (Q13)", "TypeDelta": -1, "HD": 0.70, "EC": 0.60, "Direction": "Downgrade"},
    {"QID": "RC-2",       "TypeDelta":  0, "HD": None, "EC": None,  "Direction": "Blocked"},
    {"QID": "RC-3",       "TypeDelta": +1, "HD": 0.80, "EC": 0.65,  "Direction": "Upgrade"},
    {"QID": "RC-4",       "TypeDelta": -1, "HD": 0.50, "EC": 0.50,  "Direction": "Downgrade"},
    {"QID": "RC-5",       "TypeDelta": -1, "HD": 0.70, "EC": 0.75,  "Direction": "Downgrade"},
]

# Benchmark-wide type delta buckets for Chart 5
TYPE_DELTA_BUCKETS = [
    {"Delta": "Δ−2", "n": 0,  "HD": None,  "EC": None},
    {"Delta": "Δ−1", "n": 17, "HD": 0.535, "EC": 0.579},
    {"Delta": "Δ0",  "n": 5,  "HD": 0.800, "EC": 0.700},
    {"Delta": "Δ+1", "n": 2,  "HD": 0.767, "EC": 0.767},
]

FLAT_RERANKER_QIDS = {"Q13", "Q5", "Q12", "Q7", "FR-2"}

QUERY_REGISTRY = [
    {"id": "Q1",   "category": "factual_retrieval",    "hay": "A", "nic": "T1",
     "query": "Lincoln noted how many voters from Kansas and Nevada participated in the 1864 election",
     "ideal_docs": [413, 414], "missing": None},
    {"id": "Q2",   "category": "factual_retrieval",    "hay": "D", "nic": "T3",
     "query": "How does Russia factor into Lincoln's speeches?",
     "ideal_docs": [305, 351, 381], "missing": "Eduard de Stoeckl, Alaska purchase negotiations"},
    {"id": "Q3",   "category": "factual_retrieval",    "hay": "D", "nic": "T2",
     "query": "In what ways did Lincoln highlight the contributions of immigrants during the Civil War?",
     "ideal_docs": [390, 349, 350], "missing": None},
    {"id": "Q4",   "category": "analysis",             "hay": "A", "nic": "T2",
     "query": "How did Lincoln incorporate allusions in his Second Inaugural Address?",
     "ideal_docs": [419, 420, 421, 422], "missing": None},
    {"id": "Q5",   "category": "analysis",             "hay": "D", "nic": "T2",
     "query": "How did Lincoln characterize the implications of major Supreme Court decisions before the Civil War?",
     "ideal_docs": [88, 95, 101], "missing": None},
    {"id": "Q7",   "category": "comparative_analysis", "hay": "E", "nic": "T4",
     "query": "How did Lincoln's discussion of slavery evolve between his House Divided speech and his Second Inaugural Address?",
     "ideal_docs": [88, 95, 101, 419, 420, 421, 422], "missing": "Lincoln-Douglas Debates — watch whether retrieved"},
    {"id": "Q8",   "category": "comparative_analysis", "hay": "D", "nic": "T4",
     "query": "How did Lincoln's justification for the Civil War evolve between his First Inaugural and Second Inaugural?",
     "ideal_docs": [185, 191, 197, 202, 419, 420, 421, 422], "missing": None},
    {"id": "Q9",   "category": "comparative_analysis", "hay": "D", "nic": "T4",
     "query": "How did Lincoln's views of African American soldiers change or remain the same over time?",
     "ideal_docs": [288, 295, 367, 374], "missing": "Executive orders on Black troop pay"},
    {"id": "Q10",  "category": "synthesis",            "hay": "D", "nic": "T4",
     "query": "How did Lincoln develop the theme of divine providence throughout his wartime speeches?",
     "ideal_docs": [298, 418, 419, 420, 421, 422], "missing": "Thanksgiving proclamations (not in corpus)"},
    {"id": "Q11",  "category": "synthesis",            "hay": "D", "nic": "T5",
     "query": "How did Lincoln consistently frame the relationship between liberty and law?",
     "ideal_docs": [153, 159, 185, 191, 418, 419], "missing": None},
    {"id": "Q12",  "category": "synthesis",            "hay": "D", "nic": "T5",
     "query": "What themes did Lincoln consistently employ when discussing the Constitution's relationship to slavery?",
     "ideal_docs": [153, 159, 185, 191], "missing": None},
    {"id": "Q13",  "category": "race_citizenship",     "hay": "E", "nic": "T5",
     "query": "How did Lincoln's views on African American citizenship and racial equality evolve across his speeches?",
     "ideal_docs": [288, 295, 367, 374, 413, 414, 419], "missing": "Last Public Address Apr 11 1865 — NOT IN CORPUS"},
    {"id": "R1",   "category": "analysis",             "hay": "D", "nic": "T3",
     "query": "How did Lincoln justify the naval blockade of Confederate ports?",
     "ideal_docs": [218, 272, 300, 345, 359], "missing": "Trent Affair — NOT IN CORPUS"},
    {"id": "R2",   "category": "comparative_analysis", "hay": "D", "nic": "T4",
     "query": "How did Lincoln describe U.S. relations with Great Britain during the Civil War?",
     "ideal_docs": [242, 243, 247, 300, 301, 345, 346, 388], "missing": "Trent Affair — NOT IN CORPUS"},
    {"id": "R3",   "category": "factual_retrieval",    "hay": "A", "nic": "T1",
     "query": "How did Lincoln report on the financial condition of the Post Office Department during the war?",
     "ideal_docs": [311, 312, 364, 365, 401], "missing": None},
    {"id": "AN-5", "category": "analysis",             "hay": "A", "nic": "T2",
     "query": "How did Lincoln develop the labor-capital argument in his First Annual Message?",
     "ideal_docs": [279, 280, 281], "missing": None},
    {"id": "CA-5", "category": "comparative_analysis", "hay": "E", "nic": "T3",
     "query": "How did Lincoln's tone toward the South differ between his First and Second Inaugural Addresses?",
     "ideal_docs": [193, 195, 420, 421, 422], "missing": None},
    {"id": "CA-6", "category": "comparative_analysis", "hay": "D", "nic": "T3",
     "query": "How did Lincoln justify the suspension of habeas corpus during the Civil War?",
     "ideal_docs": [214, 219, 221, 380], "missing": None},
    {"id": "FR-2", "category": "factual_retrieval",    "hay": "D", "nic": "T3",
     "query": "How did Lincoln address wartime taxation, debt, and civic obligation in his Annual Messages?",
     "ideal_docs": [249, 309, 310, 393, 395], "missing": None},
    {"id": "S-4",  "category": "synthesis",            "hay": "E", "nic": "T4",
     "query": "How did Lincoln use the Declaration of Independence as a rhetorical framework across his career?",
     "ideal_docs": [44, 45, 418, 624, 626, 628], "missing": None},
    {"id": "S-5",  "category": "synthesis",            "hay": "D", "nic": "T4",
     "query": "How did Lincoln frame self-government as a test of democratic viability?",
     "ideal_docs": [46, 47, 48, 239, 418], "missing": None},
    {"id": "RC-3", "category": "race_citizenship",     "hay": "E", "nic": "T3",
     "query": "How did Lincoln distinguish between natural rights and political equality in 1858?",
     "ideal_docs": [41, 481, 550, 624, 679], "missing": None},
    {"id": "RC-4", "category": "race_citizenship",     "hay": "D", "nic": "T3",
     "query": "How did Lincoln link emancipation, Black military service, and the future status of freed people?",
     "ideal_docs": [293, 295, 372, 374], "missing": None},
    {"id": "RC-5", "category": "race_citizenship",     "hay": "E", "nic": "T4",
     "query": "What did Lincoln leave unresolved about the future political status of freed people?",
     "ideal_docs": [297, 375, 376, 378, 410, 416], "missing": None},
]
QUERY_REGISTRY_BY_ID = {q["id"]: q for q in QUERY_REGISTRY}

CAT_COLORS = {
    "comparative_analysis": "#4C78A8",
    "factual_retrieval":    "#72B7B2",
    "race_citizenship":     "#E45756",
    "analysis":             "#F58518",
    "synthesis":            "#54A24B",
}

CAT_LABELS = {
    "comparative_analysis": "Comparative Analysis",
    "factual_retrieval":    "Factual Retrieval",
    "race_citizenship":     "Race & Citizenship",
    "analysis":             "Analysis",
    "synthesis":            "Synthesis",
}

DIM_COLORS = {"FA": "#4C78A8", "CA": "#72B7B2", "HD": "#E45756", "EC": "#F58518"}
DIM_LABELS = {
    "FA": "Factual Accuracy",
    "CA": "Citation Accuracy",
    "HD": "Historiographical Depth",
    "EC": "Epistemic Calibration",
}

# ---------------------------------------------------------------------------
# GITHUB DATA LOADING
# ---------------------------------------------------------------------------

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Dr-Hutchinson/nicolay/main/benchmark_data/"
CORPUS_JSON_URL = "https://raw.githubusercontent.com/Dr-Hutchinson/nicolay/main/data/lincoln_speech_corpus_repaired_1.json"
RUN_FILES    = [f"merged_run_{i}.csv" for i in range(5)]
SUMMARY_FILE = "bootstrap_summary_final.csv"


def normalize_run_df(df, run_idx):
    df = df.copy()
    df["run"] = run_idx
    renames = {
        "Query":                        "QueryText",
        "RubricFactualAccuracy":        "FA",
        "RubricCitationAccuracy":       "CA",
        "RubricHistoriographicalDepth": "HD",
        "RubricEpistemicCalibration":   "EC",
        "RecallAt5":                    "R@5",
        "RetrievalKeywordCountTop5":    "KW",
        "RetrievalSemanticCountTop5":   "Sem",
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
    import requests
    results, errors, run_dfs = {}, [], []
    for i, fname in enumerate(RUN_FILES):
        url = GITHUB_RAW_BASE + fname
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            run_dfs.append(normalize_run_df(df, i))
        except Exception as e:
            errors.append(f"{fname}: {e}")
    if run_dfs:
        results["runs"] = pd.concat(run_dfs, ignore_index=True)
    try:
        r = requests.get(GITHUB_RAW_BASE + SUMMARY_FILE, timeout=20)
        r.raise_for_status()
        results["summary"] = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        errors.append(f"{SUMMARY_FILE}: {e}")
    results["errors"] = errors
    return results


@st.cache_data(show_spinner=False)
def fetch_conkling_chunks():
    """Fetch Conkling letter chunks (332–342) from the Lincoln corpus JSON."""
    import requests
    try:
        r = requests.get(CORPUS_JSON_URL, timeout=30)
        r.raise_for_status()
        corpus = r.json()
        # Handle both list and dict formats
        if isinstance(corpus, list):
            chunks = [c for c in corpus if c.get("position") is not None
                      and 332 <= int(c.get("text_id", "999").replace("Text #: ", "").replace("Text #:", "").strip()) <= 342]
            if not chunks:
                # Try numeric index fallback
                chunks = corpus[332:343] if len(corpus) > 342 else []
        elif isinstance(corpus, dict):
            chunks = [v for k, v in corpus.items()
                      if isinstance(v, dict) and 332 <= int(str(k)) <= 342]
        else:
            chunks = []
        return chunks
    except Exception as e:
        return []


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

st.sidebar.title("🏛️ HonestAbe")
st.sidebar.caption("H4N4 · ada-002 · 886 chunks · 5 runs · n=120")
st.sidebar.markdown("---")
st.sidebar.subheader("Live Data")

if "live_loaded" not in st.session_state:
    st.session_state.live_loaded = None

col_btn, col_reload = st.sidebar.columns([2, 1])
with col_btn:
    load_btn = st.button("⬇ Load from GitHub", use_container_width=True)
with col_reload:
    if st.button("↻", help="Clear cache and reload"):
        fetch_github_csvs.clear()
        fetch_conkling_chunks.clear()
        st.session_state.live_loaded = None
        st.rerun()

if load_btn:
    with st.spinner("Fetching from GitHub…"):
        st.session_state.live_loaded = fetch_github_csvs()

loaded = st.session_state.live_loaded or {}
using_live_data = "runs" in loaded

if using_live_data:
    n_runs = loaded["runs"]["run"].nunique()
    n_obs  = len(loaded["runs"])
    st.sidebar.success(f"✓ {n_runs} runs · {n_obs} observations")
    for err in loaded.get("errors", []):
        st.sidebar.warning(err)
elif load_btn:
    st.sidebar.error("Load failed.")
    for err in loaded.get("errors", []):
        st.sidebar.warning(err)
else:
    st.sidebar.info("Embedded data (2026-03-28)")

st.sidebar.markdown("---")
st.sidebar.subheader("Display Options")
show_annotations = st.sidebar.checkbox("Show question labels", value=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Hutchinson, D. (forthcoming). 'Nicolay: Exploring Historic Text Collections "
    "with Large Language Models and Retrieval Augmented Generation.' "
    "*Digital Humanities Quarterly.*"
)

# ---------------------------------------------------------------------------
# WORKING DATAFRAMES
# ---------------------------------------------------------------------------

df_query = pd.DataFrame(PER_QUERY_DATA).sort_values("Total", ascending=False).reset_index(drop=True)
df_cat   = pd.DataFrame(CATEGORY_RESULTS)

# ---------------------------------------------------------------------------
# TAB LAYOUT
# ---------------------------------------------------------------------------

tab_overview, tab_browser, tab_summary, tab_retrieval, tab_type, tab_fidelity, tab_annotation = st.tabs([
    "🏠 Overview",
    "🔎 Question Browser",
    "📊 Grand Summary",
    "📂 Retrieval & Category Performance",
    "📐 Type Classification",
    "🔍 Response Fidelity & Hallucinations",
    "✏️ Human Annotation",
])

# =============================================================================
# TAB 0 — OVERVIEW
# =============================================================================

with tab_overview:
    st.title("HonestAbe Benchmark Viewer")
    st.markdown(
        "This viewer accompanies the article *Nicolay: Exploring Historic Text Collections "
        "with Large Language Models and Retrieval Augmented Generation* "
        "(Hutchinson, *Digital Humanities Quarterly*, forthcoming). "
        "It presents the results of the HonestAbe benchmark, which evaluates the Nicolay RAG system "
        "across 24 questions in five historical query categories."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Benchmark Configuration")
        st.markdown(
            "- **System:** Nicolay RAG (H4N4 configuration)\n"
            "- **Embedding model:** OpenAI Ada-002\n"
            "- **Corpus:** 886 chunks · 22 Lincoln speeches\n"
            "- **Reranker:** Cohere rerank-v4.0-pro\n"
            "- **Runs:** 5 · **Observations:** 120\n"
            "- **Evaluation:** LLM-as-judge (Claude Sonnet 4.6)\n"
            "- **Confidence intervals:** Bootstrap, n=1,000"
        )

        st.subheader("Assessment Dimensions")
        st.markdown(
            "- **FA** Factual Accuracy (0–1)\n"
            "- **CA** Citation Accuracy (0–1)\n"
            "- **HD** Historiographical Depth (0–1)\n"
            "- **EC** Epistemic Calibration (0–1)\n"
            "- **Total** Sum of four dimensions (0–4)"
        )

        st.subheader("Question Categories")
        for cat, label in CAT_LABELS.items():
            n = next(c["n"] for c in CATEGORY_RESULTS if c["category"] == cat)
            color = CAT_COLORS[cat]
            st.markdown(
                f"<span style='display:inline-block;width:12px;height:12px;"
                f"background:{color};border-radius:2px;margin-right:6px;'></span>"
                f"**{label}** (n={n})",
                unsafe_allow_html=True,
            )

    with col2:
        st.subheader("Viewer Tabs")

        tabs_desc = [
            ("🔎 Question Browser",
             "Browse all 24 benchmark questions by category. Select any question to inspect "
             "full pipeline output per run when live data is loaded from GitHub."),
            ("📊 Grand Summary",
             "Canonical benchmark result (2.883 [2.802, 2.961]) with dimension means, "
             "95% bootstrap confidence intervals, per-run stability, and quote verification totals."),
            ("📂 Retrieval & Category Performance",
             "Chart 1: search path composition (keyword vs. semantic) and R@5 by category. "
             "Chart 2: R@5 per question. Chart 3: rubric dimension scores by category."),
            ("📐 Type Classification",
             "Charts 4 and 5 from the article. Chart 4: type classification direction and "
             "HD/EC scores across the Race & Citizenship category. "
             "Chart 5: type classification direction and response quality across the full benchmark."),
            ("🔍 Response Fidelity & Hallucinations",
             "Programmatic quote verification results across all 120 observations. "
             "Targeted exhibit of fabrication cases in the Race & Citizenship category, "
             "with comparison to the Conkling letter source text."),
            ("✏️ Human Annotation",
             "Blind scoring panel. Upload any merged_run_N.csv to score responses "
             "independently and compare against automated rubric scores."),
        ]

        for title, desc in tabs_desc:
            st.markdown(
                f"<div style='border-left:3px solid #b87333;padding:8px 12px;"
                f"margin-bottom:10px;background:rgba(0,0,0,0.03);border-radius:0 6px 6px 0;'>"
                f"<div style='font-weight:600;font-size:14px;'>{title}</div>"
                f"<div style='font-size:13px;color:#555;margin-top:3px;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.subheader("Data Sources")
        st.markdown(
            "- [Benchmark data (GitHub)](https://github.com/Dr-Hutchinson/nicolay/tree/main/benchmark_data/)\n"
            "- [Lincoln corpus JSON (GitHub)](https://github.com/Dr-Hutchinson/nicolay/blob/main/data/lincoln_speech_corpus_repaired_1.json)\n"
            "- [Hay finetuning dataset](https://github.com/Dr-Hutchinson/nicolay/blob/main/data/finetuning_datasets/hays_v4_final.jsonl)\n"
            "- [Nicolay finetuning dataset](https://github.com/Dr-Hutchinson/nicolay/blob/main/data/finetuning_datasets/nicolay_v4_full.jsonl)"
        )

    st.markdown("---")
    st.caption(
        "Canonical result: **2.883 [2.802, 2.961]** · 5 runs · 24 questions · 120 observations · "
        "Bootstrap CI n=1,000 · Data: 2026-03-28"
    )

# =============================================================================
# TAB 1 — QUESTION BROWSER
# =============================================================================

with tab_browser:
    st.header("Question Browser")

    all_cats = ["All categories"] + sorted(set(q["category"] for q in QUERY_REGISTRY))
    selected_cat = st.selectbox(
        "Filter by category",
        options=all_cats,
        format_func=lambda c: "All categories" if c == "All categories" else CAT_LABELS.get(c, c),
    )

    filtered_qs = [q for q in QUERY_REGISTRY
                   if selected_cat == "All categories" or q["category"] == selected_cat]

    def _q_label(q):
        return f"{q['id']}  —  {q['query'][:85]}{'…' if len(q['query']) > 85 else ''}"

    selected_q = st.selectbox("Select question", options=filtered_qs, format_func=_q_label)

    if selected_q:
        qid = selected_q["id"]
        qmeta = QUERY_REGISTRY_BY_ID[qid]
        cat_color = CAT_COLORS.get(qmeta["category"], "#888")

        st.markdown(
            f"<div style='border-left:4px solid {cat_color};padding:10px 16px;"
            f"background:rgba(0,0,0,0.04);border-radius:0 8px 8px 0;margin-bottom:12px;'>"
            f"<div style='font-size:18px;font-weight:700;'>{qid}</div>"
            f"<div style='font-size:13px;color:#888;margin:2px 0 8px;'>"
            f"{CAT_LABELS.get(qmeta['category'], qmeta['category'])}</div>"
            f"<div style='font-size:15px;font-style:italic;'>\"{qmeta['query']}\"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Expected Hay Type", qmeta["hay"])
        mc2.metric("Expected Nicolay Type", qmeta["nic"])
        mc3.metric("Ideal Docs", len(qmeta["ideal_docs"]))
        q_row = next((r for r in PER_QUERY_DATA if r["QID"] == qid), None)
        if q_row:
            mc4.metric("Mean Total (5 runs)", f"{q_row['Total']:.2f}")

        if qmeta.get("missing"):
            st.warning(f"**Corpus gap:** {qmeta['missing']}")

        with st.expander("Ideal document IDs", expanded=False):
            st.code(", ".join(str(d) for d in qmeta["ideal_docs"]))

        if q_row:
            st.markdown("---")
            st.subheader("Aggregate Results (5-run mean)")
            rc1, rc2, rc3, rc4, rc5 = st.columns(5)
            for col, dim, label in zip(
                [rc1, rc2, rc3, rc4, rc5],
                ["Total", "FA", "CA", "HD", "EC"],
                ["Total", "FA", "CA", "HD", "EC"],
            ):
                col.metric(label, f"{q_row[dim]:.2f}",
                           delta=f"SD {q_row['SD']:.3f}" if dim == "Total" else None,
                           delta_color="off")

            run_scores = HEATMAP_DATA.get(qid, [])
            if run_scores:
                fig_mini = go.Figure(go.Heatmap(
                    z=[run_scores],
                    x=["Run 0", "Run 1", "Run 2", "Run 3", "Run 4"],
                    y=[qid],
                    colorscale=[[0, "#d73027"], [0.375, "#fc8d59"], [0.625, "#4575b4"], [1, "#1a6b3c"]],
                    zmin=2.0, zmax=4.0,
                    text=[[f"{v:.2f}" for v in run_scores]],
                    texttemplate="%{text}",
                    showscale=False,
                ))
                fig_mini.update_layout(
                    height=90, margin=dict(l=60, r=20, t=10, b=30),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_mini, use_container_width=True)

            rr1, rr2, rr3 = st.columns(3)
            rr1.metric("R@5", f"{q_row['R@5']:.3f}")
            rr2.metric("Keyword slots (mean)", f"{q_row['KW']:.1f} / 5")
            rr3.metric("Semantic slots (mean)", f"{q_row['Sem']:.1f} / 5")

        st.markdown("---")
        st.subheader("Individual Run Inspection")
        if not using_live_data:
            st.info("Load live data from GitHub (sidebar) to inspect per-run pipeline outputs.")
        else:
            run_df = loaded["runs"]
            q_runs = run_df[run_df["QueryID"] == qid].copy()
            if q_runs.empty:
                st.warning(f"No rows found for {qid}.")
            else:
                run_nums = sorted(q_runs["run"].unique())
                selected_run = st.selectbox(
                    "Select run", options=run_nums,
                    format_func=lambda r: f"Run {r}", key="browser_run_select",
                )
                row = q_runs[q_runs["run"] == selected_run].iloc[0]

                def _get(col, default="—"):
                    v = row.get(col, "")
                    s = str(v).strip()
                    return s if s not in ("", "nan") else default

                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Hay Type", f"{_get('HayTypeGot')} (exp {_get('HayTypeExpected')})")
                p2.metric("Nicolay Type", f"{_get('NicolayTypeGot')} (exp {_get('NicolayTypeExpected')})")
                p3.metric("R@5", _get("R@5"))
                p4.metric("Rubric Total", _get("RubricTotal"))

                dc = st.columns(4)
                for c, (col, lbl) in zip(dc, [("FA", "FA"), ("CA", "CA"), ("HD", "HD"), ("EC", "EC")]):
                    c.metric(lbl, _get(col))

                with st.expander("Hay — InitialAnswer & QueryAssessment", expanded=True):
                    hay_ok = _get("HayTypeCorrect")
                    hay_color = "#2d6a4f" if hay_ok == "True" else "#c1121f"
                    st.markdown(
                        f"<div style='font-size:12px;color:{hay_color};font-weight:600;margin-bottom:6px;'>"
                        f"Type: {_get('HayTypeGot')} (expected {_get('HayTypeExpected')}) — "
                        f"{'Correct' if hay_ok == 'True' else 'Incorrect'}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**InitialAnswer**")
                    st.markdown(
                        f"<div style='background:#eef7ee;padding:10px;border-radius:6px;"
                        f"font-size:13px;border-left:3px solid #52b788;'>{_get('InitialAnswer')}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**QueryAssessment**")
                    st.markdown(
                        f"<div style='background:#f8f4ef;padding:10px;border-radius:6px;"
                        f"font-size:13px;border-left:3px solid #cca855;'>{_get('QueryAssessment')}</div>",
                        unsafe_allow_html=True,
                    )

                with st.expander("Retrieval", expanded=True):
                    ret_cols = st.columns(4)
                    ret_cols[0].metric("KW slots", _get("KW"))
                    ret_cols[1].metric("Sem slots", _get("Sem"))
                    rerank_val = _get("Rerank")
                    ret_cols[2].metric("Reranker spread", f"{float(rerank_val):.3f}" if rerank_val != "—" else "—")
                    ret_cols[3].metric("P@5", _get("PrecisionAt5"))
                    st.code(_get("RetrievalPathTop5"))
                    hc, mc = st.columns(2)
                    hc.markdown(f"Hit: `{_get('IdealDocsHit')}`")
                    mc.markdown(f"Missed: `{_get('IdealDocsMissed')}`")

                with st.expander("Nicolay — Synthesis & Final Answer", expanded=True):
                    nic_ok = _get("NicolayTypeCorrect")
                    nic_color = "#2d6a4f" if nic_ok == "True" else "#c1121f"
                    st.markdown(
                        f"<div style='font-size:12px;color:{nic_color};font-weight:600;margin-bottom:6px;'>"
                        f"Type: {_get('NicolayTypeGot')} (expected {_get('NicolayTypeExpected')}) — "
                        f"{'Correct' if nic_ok == 'True' else 'Incorrect'}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Chain-of-thought trace**")
                    st.markdown(
                        f"<div style='background:#e8f0f8;padding:10px;border-radius:6px;"
                        f"font-size:13px;border-left:3px solid #4c78a8;white-space:pre-wrap;'>"
                        f"{_get('NicolaySynthesisAssessmentRaw')}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Final Answer**")
                    st.markdown(
                        f"<div style='background:#f8f4ef;padding:10px;border-radius:6px;"
                        f"font-size:14px;border-left:3px solid #b87333;'>"
                        f"{_get('FinalAnswerText')}</div>",
                        unsafe_allow_html=True,
                    )

                with st.expander("Quote Verification", expanded=False):
                    qv = st.columns(5)
                    for c, (lbl, col) in zip(qv, [
                        ("Verified", "QuotesVerified"), ("Approx", "QuotesApprox"),
                        ("Displaced", "QuotesDisplaced"), ("Fabricated", "QuotesFabricated"),
                        ("Mislabeled", "QuotesMislabeled"),
                    ]):
                        c.metric(lbl, _get(col, "0"))
                    conf_cols = st.columns(3)
                    conf_cols[0].metric("Confidence Rating", _get("ConfidenceRating"))
                    conf_cols[1].metric("ROUGE-1 max retrieved", _get("Rouge1MaxRetrieved"))
                    conf_cols[2].metric("Calib. Warning", _get("ConfidenceCalibWarning"))

                with st.expander("LLM Rubric Scores & Rationales", expanded=True):
                    sc = st.columns(5)
                    for c, (col, lbl) in zip(sc, [
                        ("FA", "FA"), ("CA", "CA"), ("HD", "HD"), ("EC", "EC"), ("RubricTotal", "Total"),
                    ]):
                        c.metric(lbl, _get(col))
                    for rat_col, lbl, bg in [
                        ("RationaleFactualAccuracy",        "FA Rationale",   "#eef7ee"),
                        ("RationaleCitationAccuracy",       "CA Rationale",   "#e8f0f8"),
                        ("RationaleHistoriographicalDepth", "HD Rationale",   "#f8f4ef"),
                        ("RationaleEpistemicCalibration",   "EC Rationale",   "#fef9ec"),
                        ("RationaleHayDiagnostic",          "Hay Diagnostic", "#fff3cd"),
                    ]:
                        val = _get(rat_col)
                        if val != "—":
                            st.markdown(
                                f"<div style='background:{bg};padding:8px;border-radius:6px;"
                                f"font-size:13px;margin-bottom:5px;'>"
                                f"<b>{lbl}:</b> {val}</div>",
                                unsafe_allow_html=True,
                            )

# =============================================================================
# TAB 2 — GRAND SUMMARY
# =============================================================================

with tab_summary:
    st.header("Grand Summary")
    st.caption("H4N4 ada-002 · 886-chunk corpus · rerank-v4.0-pro · k=5 · 5 runs · n=120")

    # Canonical number callout
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.markdown(
            f"""
            <div style="background:#1a1a2e;border-radius:12px;padding:24px;text-align:center;border:2px solid #b87333;">
            <div style="color:#b87333;font-size:12px;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">
            Mean Total Score</div>
            <div style="color:#f5f0e8;font-size:52px;font-weight:700;line-height:1;">{CANONICAL_ARTICLE_NUMBER:.3f}</div>
            <div style="color:#aaa;font-size:13px;margin-top:6px;">
            95% CI [{CI_LOWER:.3f}, {CI_UPPER:.3f}] · bootstrap n=1,000</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Dimension means
    fig_dims = go.Figure()
    for dim in ["FA", "CA", "HD", "EC"]:
        d = GRAND_MEANS[dim]
        fig_dims.add_trace(go.Bar(
            x=[d["mean"]],
            y=[DIM_LABELS[dim]],
            orientation="h",
            error_x=dict(
                type="data",
                array=[d["ci_hi"] - d["mean"]],
                arrayminus=[d["mean"] - d["ci_lo"]],
                color="rgba(255,255,255,0.5)",
                thickness=2, width=6,
            ),
            marker_color=DIM_COLORS[dim],
            text=[f"{d['mean']:.3f}"],
            textposition="outside",
            textfont=dict(size=13),
            name=DIM_LABELS[dim],
            hovertemplate=(
                f"<b>{DIM_LABELS[dim]}</b><br>"
                f"Mean: {d['mean']:.3f}<br>"
                f"95% CI: [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]<extra></extra>"
            ),
        ))
    fig_dims.update_layout(
        title="Dimension Means with 95% Bootstrap CI",
        xaxis=dict(title="Mean Score (0–1)", range=[0, 1.15]),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        height=300,
        margin=dict(l=20, r=60, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
    )
    st.plotly_chart(fig_dims, use_container_width=True)

    # Dimension metrics row
    col1, col2, col3, col4 = st.columns(4)
    for col, dim in zip([col1, col2, col3, col4], ["FA", "CA", "HD", "EC"]):
        d = GRAND_MEANS[dim]
        col.metric(
            DIM_LABELS[dim],
            f"{d['mean']:.3f}",
            delta=f"[{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]",
            delta_color="off",
        )

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
    ))
    fig_runs.add_hline(
        y=CANONICAL_ARTICLE_NUMBER, line_dash="dash", line_color="#aaa",
        annotation_text=f"Mean: {CANONICAL_ARTICLE_NUMBER:.3f}",
        annotation_position="right",
    )
    fig_runs.add_hrect(
        y0=CI_LOWER, y1=CI_UPPER,
        fillcolor="rgba(184,115,51,0.1)", line_width=0,
    )
    fig_runs.update_layout(
        title="Run-Level Total Score",
        yaxis=dict(title="Mean Total (0–4)", range=[2.6, 3.2]),
        height=260,
        margin=dict(l=20, r=80, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        showlegend=False,
    )
    st.plotly_chart(fig_runs, use_container_width=True)

    st.markdown("---")

    # Per-query heatmap
    st.subheader("Per-Query Rubric Total by Run")
    qids_sorted = df_query["QID"].tolist()
    z_matrix    = [HEATMAP_DATA[qid] for qid in qids_sorted]
    cats_sorted = df_query["Category"].tolist()

    y_labels = [
        f"{qid} ({cat.replace('_', ' ')[:3].upper()})"
        for qid, cat in zip(qids_sorted, cats_sorted)
    ]
    sds_sorted = df_query["SD"].tolist()
    means_sorted = df_query["Total"].tolist()

    fig_hm = go.Figure(go.Heatmap(
        z=z_matrix,
        x=["Run 0", "Run 1", "Run 2", "Run 3", "Run 4"],
        y=y_labels,
        colorscale=[
            [0.0, "#67000d"], [0.25, "#E45756"],
            [0.50, "#F58518"], [0.75, "#4C78A8"], [1.0, "#1a6b3c"],
        ],
        zmin=2.0, zmax=4.0,
        text=[[f"{v:.2f}" for v in row] for row in z_matrix],
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="Score (0–4)", tickvals=[2.0, 2.5, 3.0, 3.5, 4.0]),
        hovertemplate="<b>%{y}</b> — %{x}<br>Score: %{z:.2f}<extra></extra>",
    ))
    fig_hm.update_layout(
        title="Rubric Total by Question × Run (sorted high→low by mean)",
        height=820,
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=11)),
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("---")

    # Quote verification summary
    st.subheader("Quote Verification — n=120 observations")
    qv_cols = st.columns(5)
    qv_data = [
        ("Verified",    597, "✓", "#4C78A8"),
        ("Approximate",   0, "~", "#72B7B2"),
        ("Displaced",     2, "⇌", "#F58518"),
        ("Fabricated",    1, "✗", "#E45756"),
        ("Mislabeled",    0, "?", "#aaa"),
    ]
    for col, (label, n, icon, color) in zip(qv_cols, qv_data):
        col.markdown(
            f"<div style='text-align:center;padding:12px;background:rgba(0,0,0,0.05);border-radius:8px;"
            f"border-top:3px solid {color};'>"
            f"<div style='font-size:22px;'>{icon}</div>"
            f"<div style='font-size:28px;font-weight:700;color:{color};'>{n}</div>"
            f"<div style='font-size:12px;color:#888;'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.caption("1,153 total quote-check events · fabrication rate: 0.8% (1/120) within pipeline verification scope")

    # Per-query summary table
    st.markdown("---")
    st.subheader("Per-Query Summary Table")
    df_display = df_query.copy()
    df_display["Category"] = df_display["Category"].str.replace("_", " ").str.title()
    cols_show = ["QID", "Category", "Total", "SD", "FA", "CA", "HD", "EC", "R@5"]
    st.dataframe(
        df_display[cols_show].style.format({
            "Total": "{:.2f}", "SD": "{:.3f}", "FA": "{:.2f}", "CA": "{:.2f}",
            "HD": "{:.2f}", "EC": "{:.2f}", "R@5": "{:.3f}",
        }).apply(_apply_gradient, vmin=2.0, vmax=4.0,
                 low_rgb=_RdYlGn[0], high_rgb=_RdYlGn[1], subset=["Total"])
          .apply(_apply_gradient, vmin=0, vmax=0.55,
                 low_rgb=_Oranges[0], high_rgb=_Oranges[1], subset=["SD"]),
        use_container_width=True,
        height=680,
    )

# =============================================================================
# TAB 3 — RETRIEVAL & CATEGORY PERFORMANCE
# =============================================================================

with tab_retrieval:
    st.header("Retrieval & Category Performance")

    # --- Chart 1: Search path composition by category ---
    st.subheader("Chart 1 — Search Path Composition and Retrieval@5 by Question Category")

    cat_order = [c["category"] for c in sorted(CATEGORY_RESULTS, key=lambda x: -x["R@5"])]
    labels_cat = [CAT_LABELS[c] for c in cat_order]

    # Compute mean KW/Sem per category from per-query data
    df_q = pd.DataFrame(PER_QUERY_DATA)
    cat_kw_sem = df_q.groupby("Category")[["KW", "Sem"]].mean().reset_index()
    cat_kw_sem_dict = {row["Category"]: row for _, row in cat_kw_sem.iterrows()}

    kw_by_cat  = [cat_kw_sem_dict[c]["KW"]  if c in cat_kw_sem_dict else 2.5 for c in cat_order]
    sem_by_cat = [cat_kw_sem_dict[c]["Sem"] if c in cat_kw_sem_dict else 2.5 for c in cat_order]
    r5_by_cat  = [next(c2["R@5"] for c2 in CATEGORY_RESULTS if c2["category"] == c) for c in cat_order]

    fig_chart1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig_chart1.add_trace(go.Bar(
        name="Keyword slots",
        x=labels_cat,
        y=kw_by_cat,
        marker_color="#4C78A8",
        text=[f"{v:.1f}" for v in kw_by_cat],
        textposition="inside",
        textfont=dict(size=11, color="white"),
    ), secondary_y=False)
    fig_chart1.add_trace(go.Bar(
        name="Semantic slots",
        x=labels_cat,
        y=sem_by_cat,
        marker_color="#72B7B2",
        text=[f"{v:.1f}" for v in sem_by_cat],
        textposition="inside",
        textfont=dict(size=11, color="white"),
    ), secondary_y=False)
    fig_chart1.add_trace(go.Scatter(
        name="R@5",
        x=labels_cat,
        y=r5_by_cat,
        mode="markers+lines+text",
        text=[f"{v:.3f}" for v in r5_by_cat],
        textposition="top center",
        textfont=dict(size=11),
        marker=dict(size=12, color="#b87333", symbol="diamond"),
        line=dict(color="#b87333", width=2, dash="dot"),
    ), secondary_y=True)

    fig_chart1.update_layout(
        barmode="stack",
        title="Chart 1: Search Path Composition and Retrieval@5 by Question Category",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
    )
    fig_chart1.update_yaxes(title_text="Mean Slots (of 5)", range=[0, 6], secondary_y=False)
    fig_chart1.update_yaxes(title_text="R@5", range=[0, 1.15], secondary_y=True)
    st.plotly_chart(fig_chart1, use_container_width=True)

    st.markdown("---")

    # --- Chart 2: R@5 per question ---
    st.subheader("Chart 2 — Retrieval@5 by Question")

    df_q_r5 = df_query.sort_values("R@5", ascending=False).reset_index(drop=True)
    colors_r5 = [CAT_COLORS.get(cat, "#888") for cat in df_q_r5["Category"]]

    fig_chart2 = go.Figure()
    fig_chart2.add_trace(go.Bar(
        x=df_q_r5["QID"],
        y=df_q_r5["R@5"],
        marker_color=colors_r5,
        text=[f"{v:.3f}" for v in df_q_r5["R@5"]],
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate="<b>%{x}</b><br>R@5: %{y:.3f}<extra></extra>",
    ))

    # Category legend traces (invisible, for legend only)
    for cat, color in CAT_COLORS.items():
        fig_chart2.add_trace(go.Bar(
            x=[None], y=[None],
            name=CAT_LABELS[cat],
            marker_color=color,
            showlegend=True,
        ))

    fig_chart2.update_layout(
        title="Chart 2: Retrieval@5 by Question",
        yaxis=dict(title="R@5", range=[0, 1.2]),
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
        barmode="overlay",
    )
    st.plotly_chart(fig_chart2, use_container_width=True)

    st.markdown("---")

    # --- Chart 3: Evaluation scores by question type and assessment dimension ---
    st.subheader("Chart 3 — Evaluation Scores by Question Category and Assessment Dimension")

    cat_order_total = [c["category"] for c in sorted(CATEGORY_RESULTS, key=lambda x: -x["Total"])]
    labels_total = [CAT_LABELS[c] for c in cat_order_total]

    fig_chart3 = go.Figure()
    for dim in ["FA", "CA", "HD", "EC"]:
        vals = [next(c["dim"] for c in
                     [{**r, "dim": r[dim]} for r in CATEGORY_RESULTS]
                     if c["category"] == cat)
                for cat in cat_order_total]
        fig_chart3.add_trace(go.Bar(
            name=DIM_LABELS[dim],
            x=labels_total,
            y=vals,
            marker_color=DIM_COLORS[dim],
            text=[f"{v:.2f}" for v in vals],
            textposition="inside",
            textfont=dict(size=10, color="white"),
        ))

    fig_chart3.update_layout(
        barmode="group",
        title="Chart 3: Evaluation Scores by Question Category and Assessment Dimension",
        yaxis=dict(title="Mean Score (0–1)", range=[0, 1.15]),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
    )
    st.plotly_chart(fig_chart3, use_container_width=True)

    # Category summary table
    st.markdown("---")
    st.subheader("Category Summary")
    df_cat_display = df_cat.copy()
    df_cat_display["Category"] = df_cat_display["category"].map(CAT_LABELS)
    df_cat_display = df_cat_display[["Category", "n", "Total", "FA", "CA", "HD", "EC", "R@5"]].sort_values("Total", ascending=False)
    st.dataframe(
        df_cat_display.style.format({
            "Total": "{:.3f}", "FA": "{:.3f}", "CA": "{:.3f}",
            "HD": "{:.3f}", "EC": "{:.3f}", "R@5": "{:.3f}",
        }).apply(_apply_gradient, vmin=2.4, vmax=3.4,
                 low_rgb=_RdYlGn[0], high_rgb=_RdYlGn[1], subset=["Total"]),
        use_container_width=True,
        height=230,
    )

# =============================================================================
# TAB 4 — TYPE CLASSIFICATION
# =============================================================================

with tab_type:
    st.header("Type Classification")

    # --- Chart 4: RC category ---
    st.subheader("Chart 4 — Type Classification and Response Quality in the Race and Citizenship Category")

    rc_runnable = [d for d in RC_TYPE_DATA if d["HD"] is not None]
    rc_qids     = [d["QID"]      for d in rc_runnable]
    rc_hd       = [d["HD"]       for d in rc_runnable]
    rc_ec       = [d["EC"]       for d in rc_runnable]
    rc_dir      = [d["Direction"] for d in rc_runnable]
    rc_delta    = [d["TypeDelta"] for d in rc_runnable]

    dir_color_map = {"Downgrade": "#E45756", "Exact": "#4C78A8", "Upgrade": "#F58518"}
    rc_colors = [dir_color_map.get(d, "#888") for d in rc_dir]

    fig_chart4 = go.Figure()
    for direction, color in dir_color_map.items():
        mask = [d == direction for d in rc_dir]
        if not any(mask):
            continue
        fig_chart4.add_trace(go.Scatter(
            x=[ec for ec, m in zip(rc_ec, mask) if m],
            y=[hd for hd, m in zip(rc_hd, mask) if m],
            mode="markers+text",
            name=direction,
            marker=dict(size=18, color=color, opacity=0.85,
                        line=dict(width=1.5, color="white")),
            text=[qid for qid, m in zip(rc_qids, mask) if m],
            textposition="top center",
            textfont=dict(size=11),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "EC: %{x:.2f}<br>"
                "HD: %{y:.2f}<extra></extra>"
            ),
        ))

    fig_chart4.add_hline(y=GRAND_MEANS["HD"]["mean"], line_dash="dot", line_color="#888",
                         annotation_text=f"HD mean {GRAND_MEANS['HD']['mean']:.3f}", annotation_position="right")
    fig_chart4.add_vline(x=GRAND_MEANS["EC"]["mean"], line_dash="dot", line_color="#888",
                         annotation_text=f"EC mean {GRAND_MEANS['EC']['mean']:.3f}", annotation_position="top left")

    fig_chart4.update_layout(
        title="Chart 4: Type Classification Direction and Response Quality — Race & Citizenship (n=4 runnable queries)",
        xaxis=dict(title="Epistemic Calibration (EC)", range=[0.3, 1.0]),
        yaxis=dict(title="Historiographical Depth (HD)", range=[0.3, 1.0]),
        height=460,
        legend=dict(title="Classification", font=dict(size=11)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
    )
    st.plotly_chart(fig_chart4, use_container_width=True)
    st.caption("RC-2 (Last Public Address absent from corpus) excluded. Parenthetical Δ values: RC-1 (Δ−1), RC-3 (Δ+1), RC-4 (Δ−1), RC-5 (Δ−1).")

    st.markdown("---")

    # --- Chart 5: Benchmark-wide type classification ---
    st.subheader("Chart 5 — Type Classification Direction and Response Quality Across the HonestAbe Benchmark")

    buckets_runnable = [d for d in TYPE_DELTA_BUCKETS if d["HD"] is not None]
    bucket_labels = [d["Delta"] for d in buckets_runnable]
    bucket_n      = [d["n"]     for d in buckets_runnable]
    bucket_hd     = [d["HD"]    for d in buckets_runnable]
    bucket_ec     = [d["EC"]    for d in buckets_runnable]

    bucket_colors = {"Δ−1": "#E45756", "Δ0": "#4C78A8", "Δ+1": "#F58518"}

    fig_chart5 = go.Figure()
    fig_chart5.add_trace(go.Bar(
        name="HD",
        x=bucket_labels,
        y=bucket_hd,
        marker_color=[bucket_colors.get(b, "#888") for b in bucket_labels],
        marker_opacity=0.9,
        text=[f"HD {v:.3f}" for v in bucket_hd],
        textposition="outside",
        textfont=dict(size=11),
        width=0.3,
        offset=-0.17,
        hovertemplate="<b>%{x}</b><br>HD: %{y:.3f}<extra></extra>",
    ))
    fig_chart5.add_trace(go.Bar(
        name="EC",
        x=bucket_labels,
        y=bucket_ec,
        marker_color=[bucket_colors.get(b, "#888") for b in bucket_labels],
        marker_opacity=0.5,
        text=[f"EC {v:.3f}" for v in bucket_ec],
        textposition="outside",
        textfont=dict(size=11),
        width=0.3,
        offset=0.17,
        hovertemplate="<b>%{x}</b><br>EC: %{y:.3f}<extra></extra>",
    ))

    # n= annotations
    for i, (label, n) in enumerate(zip(bucket_labels, bucket_n)):
        fig_chart5.add_annotation(
            x=label, y=0.08,
            text=f"n={n}",
            showarrow=False,
            font=dict(size=11, color="#555"),
        )

    fig_chart5.add_hline(y=GRAND_MEANS["HD"]["mean"], line_dash="dot", line_color="#E45756",
                         annotation_text=f"HD mean {GRAND_MEANS['HD']['mean']:.3f}",
                         annotation_position="right", annotation_font_color="#E45756")
    fig_chart5.add_hline(y=GRAND_MEANS["EC"]["mean"], line_dash="dot", line_color="#F58518",
                         annotation_text=f"EC mean {GRAND_MEANS['EC']['mean']:.3f}",
                         annotation_position="right", annotation_font_color="#F58518")

    fig_chart5.update_layout(
        title="Chart 5: Type Classification Direction and Response Quality Across the HonestAbe Benchmark (n=24 queries, 120 responses)",
        barmode="overlay",
        yaxis=dict(title="Mean Score (0–1)", range=[0, 1.1]),
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
    )
    st.plotly_chart(fig_chart5, use_container_width=True)

    st.markdown("---")

    # Type accuracy table
    st.subheader("Nicolay Type Classification Accuracy")
    type_acc = [
        {"Type": "T1 (Direct Synthesis)",       "n": 5,  "Accuracy": 1.000},
        {"Type": "T2 (Multi-Passage Synthesis)", "n": 25, "Accuracy": 0.400},
        {"Type": "T3 (Absence / Partial Hit)",  "n": 45, "Accuracy": 0.089},
        {"Type": "T4 (Historiographical)",       "n": 40, "Accuracy": 0.350},
        {"Type": "T5 (Epistemic Calibration)",   "n": 5,  "Accuracy": 0.000},
    ]
    df_ta = pd.DataFrame(type_acc)
    st.dataframe(
        df_ta.style.format({"Accuracy": "{:.1%}", "n": "{:d}"})
                   .apply(_apply_gradient, vmin=0, vmax=1,
                          low_rgb=_RdYlGn[0], high_rgb=_RdYlGn[1], subset=["Accuracy"]),
        use_container_width=True,
        height=220,
    )

    st.markdown("---")

    # HD vs EC scatter
    st.subheader("HD vs. EC — All Queries")
    fig_hdec = go.Figure()
    for cat in CAT_COLORS:
        mask = df_query["Category"] == cat
        sub  = df_query[mask]
        fig_hdec.add_trace(go.Scatter(
            x=sub["EC"],
            y=sub["HD"],
            mode="markers+text" if show_annotations else "markers",
            name=CAT_LABELS[cat],
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
    fig_hdec.add_hline(y=GRAND_MEANS["HD"]["mean"], line_dash="dot", line_color="#888",
                       annotation_text=f"HD mean {GRAND_MEANS['HD']['mean']:.3f}")
    fig_hdec.add_vline(x=GRAND_MEANS["EC"]["mean"], line_dash="dot", line_color="#888",
                       annotation_text=f"EC mean {GRAND_MEANS['EC']['mean']:.3f}")
    fig_hdec.update_layout(
        title="HD vs. EC (bubble size = SD)",
        xaxis=dict(title="Epistemic Calibration (EC)", range=[0.15, 1.05]),
        yaxis=dict(title="Historiographical Depth (HD)", range=[0.15, 1.10]),
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        legend=dict(font=dict(size=9), itemsizing="constant"),
    )
    st.plotly_chart(fig_hdec, use_container_width=True)

# =============================================================================
# TAB 5 — RESPONSE FIDELITY & HALLUCINATIONS
# =============================================================================

with tab_fidelity:
    st.header("Response Fidelity & Hallucinations")

    # --- Verification aggregate ---
    st.subheader("Quote Verification — All 120 Observations")

    qv_summary = [
        ("Verified",    597, "#4C78A8", "✓"),
        ("Approximate",   0, "#72B7B2", "~"),
        ("Displaced",     2, "#F58518", "⇌"),
        ("Fabricated",    1, "#E45756", "✗"),
        ("Mislabeled",    0, "#aaa",    "?"),
    ]
    total_events = 1153

    fig_qv = go.Figure()
    for label, n, color, icon in qv_summary:
        fig_qv.add_trace(go.Bar(
            name=label,
            x=[label],
            y=[n],
            marker_color=color,
            text=[str(n)],
            textposition="outside",
            textfont=dict(size=14),
            hovertemplate=f"<b>{label}</b><br>n={n}<br>{n/total_events:.2%} of {total_events} events<extra></extra>",
        ))

    fig_qv.update_layout(
        title=f"Quote Verification Results (n={total_events} quote-check events, 120 observations)",
        yaxis=dict(title="Count"),
        showlegend=False,
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        margin=dict(t=50, b=20),
    )
    st.plotly_chart(fig_qv, use_container_width=True)

    # Verification by category (from FA audit)
    st.subheader("Fabrication Rate by Question Category")
    st.caption("Proportion of quoted strings (≥5 tokens) in FinalAnswerText confirmed absent from corpus, per category.")

    cat_fab_data = [
        {"Category": "Factual Retrieval",    "Rate": 0.000, "n_flags": 0},
        {"Category": "Analysis",             "Rate": 0.071, "n_flags": 3},
        {"Category": "Comparative Analysis", "Rate": 0.075, "n_flags": 4},
        {"Category": "Race & Citizenship",   "Rate": 0.075, "n_flags": 4},
        {"Category": "Synthesis",            "Rate": 0.142, "n_flags": 6},
    ]
    cat_fab_colors = [
        CAT_COLORS["factual_retrieval"],
        CAT_COLORS["analysis"],
        CAT_COLORS["comparative_analysis"],
        CAT_COLORS["race_citizenship"],
        CAT_COLORS["synthesis"],
    ]

    fig_fab_cat = go.Figure(go.Bar(
        x=[d["Category"] for d in cat_fab_data],
        y=[d["Rate"] for d in cat_fab_data],
        marker_color=cat_fab_colors,
        text=[f"{d['Rate']:.1%}" for d in cat_fab_data],
        textposition="outside",
        textfont=dict(size=12),
        hovertemplate="<b>%{x}</b><br>Rate: %{y:.1%}<extra></extra>",
    ))
    fig_fab_cat.update_layout(
        title="Fabrication Rate by Question Category",
        yaxis=dict(title="Rate", tickformat=".0%", range=[0, 0.22]),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
    )
    st.plotly_chart(fig_fab_cat, use_container_width=True)

    st.markdown("---")

    # --- RC-4 Hallucination Exhibit ---
    st.subheader("RC-4 Fabrication Exhibit")
    st.markdown(
        "**Question:** How did Lincoln link emancipation, Black military service, "
        "and the future status of freed people?"
    )
    st.markdown(
        "RC-4 achieved R@5=0.933 (all three target documents retrieved) across all five runs, "
        "yet produced confirmed fabricated quotations in two runs. "
        "The fabrications draw on the Conkling letter (August 26, 1863), "
        "the primary retrieved source for this question."
    )

    col_fab1, col_fab2 = st.columns(2)

    with col_fab1:
        st.markdown(
            "<div style='border-left:4px solid #E45756;padding:10px 14px;"
            "background:rgba(228,87,86,0.06);border-radius:0 8px 8px 0;margin-bottom:10px;'>"
            "<div style='font-weight:700;font-size:13px;color:#E45756;margin-bottom:6px;'>"
            "Run 0 — Fabricated quotation</div>"
            "<div style='font-style:italic;font-size:13px;'>"
            "\"ought to be a step toward their ultimate admission to citizenship\""
            "</div>"
            "<div style='font-size:11px;color:#888;margin-top:6px;'>"
            "Attributed to: Conkling letter (Lincoln 1863)<br>"
            "Pipeline verification: QuotesFabricated=0 · CA=0.75"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    with col_fab2:
        st.markdown(
            "<div style='border-left:4px solid #E45756;padding:10px 14px;"
            "background:rgba(228,87,86,0.06);border-radius:0 8px 8px 0;margin-bottom:10px;'>"
            "<div style='font-weight:700;font-size:13px;color:#E45756;margin-bottom:6px;'>"
            "Run 1 — Fabricated quotation</div>"
            "<div style='font-style:italic;font-size:13px;'>"
            "\"The service of colored soldiers in the war of the rebellion, is the very motive "
            "which, in my judgment, will secure to them the right of suffrage\""
            "</div>"
            "<div style='font-size:11px;color:#888;margin-top:6px;'>"
            "Attributed to: Conkling letter (Lincoln 1863)<br>"
            "Pipeline verification: QuotesFabricated=0 · CA=0.75"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # Conkling letter source text
    st.markdown("**Conkling Letter — Source Text (Corpus Chunks 332–342)**")
    st.caption("Load from GitHub to display the actual Conkling letter chunks from the Lincoln corpus.")

    if not using_live_data:
        st.info("Load live data from GitHub (sidebar) to display Conkling letter source text.")
    else:
        with st.spinner("Fetching Conkling letter chunks…"):
            conkling_chunks = fetch_conkling_chunks()

        if conkling_chunks:
            for chunk in conkling_chunks:
                # Normalize field names across possible schema variants
                text_id  = chunk.get("text_id", chunk.get("id", ""))
                source   = chunk.get("source", "")
                raw_text = chunk.get("full_text", chunk.get("text", ""))
                st.markdown(
                    f"<div style='border:1px solid #ddd;border-radius:6px;padding:10px 14px;"
                    f"margin-bottom:8px;background:rgba(0,0,0,0.02);'>"
                    f"<div style='font-size:11px;color:#888;margin-bottom:4px;'>"
                    f"{text_id} · {source}</div>"
                    f"<div style='font-size:13px;'>{raw_text}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.warning(
                "Could not retrieve Conkling letter chunks from the corpus JSON. "
                "Verify the corpus file is accessible at the GitHub URL."
            )

        # RC-4 final answers for the two fabrication runs, from live data
        st.markdown("---")
        st.markdown("**RC-4 Final Answer Text — Runs 0 and 1 (from benchmark data)**")
        run_df = loaded["runs"]
        rc4_rows = run_df[(run_df["QueryID"] == "RC-4") & (run_df["run"].isin([0, 1]))].copy()
        if not rc4_rows.empty:
            for _, row in rc4_rows.sort_values("run").iterrows():
                run_num  = int(row["run"])
                fa_text  = str(row.get("FinalAnswerText", "")).strip()
                nic_type = str(row.get("NicolayTypeGot", "—")).strip()
                ca_score = row.get("CA", row.get("RubricCitationAccuracy", "—"))
                st.markdown(
                    f"<div style='border-left:4px solid #b87333;padding:10px 14px;"
                    f"background:rgba(184,115,51,0.05);border-radius:0 8px 8px 0;margin-bottom:12px;'>"
                    f"<div style='font-weight:700;font-size:13px;margin-bottom:4px;'>"
                    f"Run {run_num} · Nicolay Type: {nic_type} · CA: {ca_score}</div>"
                    f"<div style='font-size:13px;white-space:pre-wrap;'>{fa_text}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.warning("RC-4 rows for runs 0 and 1 not found in loaded data.")

    st.markdown("---")

    # Pipeline gap note
    st.subheader("Verification Pipeline Scope")
    st.markdown(
        "The quote verification pipeline checks strings extracted from the Match Analysis "
        "Key Quote fields against the 886-chunk corpus using multi-stage string matching. "
        "Quotes appearing only in FinalAnswerText — not routed through Match Analysis entries — "
        "are not counted in the `QuotesFabricated` metric, even when absent from the corpus. "
        "The CA dimension of the LLM rubric provides a complementary measure of citation fidelity."
    )

    pipe_gap_data = [
        {"Check":  "Match Analysis key quotes",      "Scope": "✓ Checked",   "Metric": "QuotesFabricated"},
        {"Check":  "FinalAnswer quotes (unanchored)", "Scope": "✗ Not checked", "Metric": "CA (LLM rubric)"},
        {"Check":  "Paraphrased hallucinations",      "Scope": "✗ Not checked", "Metric": "HD / CA (LLM rubric)"},
        {"Check":  "Date / metadata errors",          "Scope": "✗ Not checked", "Metric": "CA (LLM rubric)"},
        {"Check":  "Speaker-attribution errors",      "Scope": "✗ Not checked", "Metric": "CA (LLM rubric)"},
    ]
    st.dataframe(pd.DataFrame(pipe_gap_data), use_container_width=True, height=220)

# =============================================================================
# TAB 6 — HUMAN ANNOTATION
# =============================================================================

with tab_annotation:
    st.header("Human Annotation Panel")
    st.caption(
        "Blind scoring protocol. Upload any merged_run_N.csv. "
        "Score each response before expanding the LLM auto-scores section."
    )

    if "annotation_scores" not in st.session_state:
        st.session_state.annotation_scores = {}
    if "annotation_idx" not in st.session_state:
        st.session_state.annotation_idx = 0
    if "annotation_df" not in st.session_state:
        st.session_state.annotation_df = None

    upload = st.file_uploader(
        "Upload benchmark CSV (merged_run_N.csv)", type=["csv"],
        help="Any merged_run_N.csv works as-is.",
    )

    if upload:
        try:
            ann_df = pd.read_csv(upload)
            if "Query" in ann_df.columns and "QueryText" not in ann_df.columns:
                ann_df = ann_df.rename(columns={"Query": "QueryText"})
            required_cols = {"QueryID", "Category", "QueryText", "FinalAnswerText"}
            missing_cols  = required_cols - set(ann_df.columns)
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                st.session_state.annotation_df = ann_df
                st.success(f"Loaded {len(ann_df)} responses.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if st.session_state.annotation_df is not None:
        ann_df  = st.session_state.annotation_df
        n_total = len(ann_df)
        n_scored = len(st.session_state.annotation_scores)
        idx      = st.session_state.annotation_idx

        st.progress(n_scored / n_total if n_total > 0 else 0,
                    text=f"Scored {n_scored} of {n_total} responses")

        nav1, nav2, nav3, nav4 = st.columns([1, 1, 3, 1])
        with nav1:
            if st.button("Prev", key="ann_prev") and idx > 0:
                st.session_state.annotation_idx -= 1
                st.rerun()
        with nav2:
            if st.button("Next", key="ann_next") and idx < n_total - 1:
                st.session_state.annotation_idx += 1
                st.rerun()
        with nav3:
            jump_to = st.selectbox(
                "Jump to", options=list(range(n_total)), index=idx,
                format_func=lambda i: f"{ann_df.iloc[i]['QueryID']} ({i+1}/{n_total})",
                key="ann_jump",
            )
            if jump_to != idx:
                st.session_state.annotation_idx = jump_to
                st.rerun()
        with nav4:
            row_qid_nav = ann_df.iloc[idx]["QueryID"]
            st.markdown("**Scored**" if row_qid_nav in st.session_state.annotation_scores else "**Unscored**")

        st.markdown("---")
        ann_row = ann_df.iloc[idx]
        qid     = ann_row["QueryID"]
        prior   = st.session_state.annotation_scores.get(qid, {})

        def _ann_get(col, default="--"):
            v = ann_row.get(col, "")
            s = str(v).strip()
            return s if s not in ("", "nan") else default

        qmeta_ann    = QUERY_REGISTRY_BY_ID.get(qid, {})
        cat_color_ann = CAT_COLORS.get(ann_row.get("Category", ""), "#888")

        st.markdown(
            f"<div style='border-left:4px solid {cat_color_ann};padding:8px 14px;"
            f"background:rgba(0,0,0,0.04);border-radius:0 8px 8px 0;margin-bottom:10px;'>"
            f"<b style='font-size:16px;'>{qid}</b>"
            f"<span style='font-size:12px;color:#888;margin-left:10px;'>"
            f"{ann_row.get('Category','').replace('_',' ').title()}</span><br>"
            f"<i style='font-size:14px;'>{ann_row.get('QueryText','')}</i>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.subheader("Pipeline Context")
        st.caption(
            "Evidentiary hierarchy: "
            "(1) FinalAnswerText — primary for FA and CA; "
            "(2) SynthesisAssessmentRaw — primary for HD and EC; "
            "(3) InitialAnswer — diagnostic only."
        )

        with st.expander("1. Hay — InitialAnswer (diagnostic only)", expanded=True):
            hay_ok  = _ann_get("HayTypeCorrect")
            hay_col = "#2d6a4f" if hay_ok == "True" else "#c1121f"
            st.markdown(
                f"<div style='font-size:12px;color:{hay_col};font-weight:600;margin-bottom:4px;'>"
                f"Hay type: {_ann_get('HayTypeGot')} (expected {_ann_get('HayTypeExpected')}) — "
                f"{'Correct' if hay_ok=='True' else 'Incorrect'}</div>",
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

        with st.expander("2. Retrieval", expanded=False):
            rc = st.columns(4)
            rc[0].metric("R@5",       _ann_get("RecallAt5"))
            rc[1].metric("P@5",       _ann_get("PrecisionAt5"))
            rc[2].metric("KW slots",  _ann_get("RetrievalKeywordCountTop5"))
            rc[3].metric("Sem slots", _ann_get("RetrievalSemanticCountTop5"))
            st.code(_ann_get("RetrievalPathTop5"))
            hc2, mc2 = st.columns(2)
            hc2.markdown(f"Hit: `{_ann_get('IdealDocsHit')}`")
            mc2.markdown(f"Missed: `{_ann_get('IdealDocsMissed')}`")
            if qmeta_ann and qmeta_ann.get("missing"):
                st.warning(f"Corpus gap: {qmeta_ann['missing']}")

        with st.expander("3. SynthesisAssessmentRaw — primary for HD and EC", expanded=True):
            nic_ok  = _ann_get("NicolayTypeCorrect")
            nic_col = "#2d6a4f" if nic_ok == "True" else "#c1121f"
            st.markdown(
                f"<div style='font-size:12px;color:{nic_col};font-weight:600;margin-bottom:4px;'>"
                f"Nicolay type: {_ann_get('NicolayTypeGot')} (expected {_ann_get('NicolayTypeExpected')}) — "
                f"{'Correct' if nic_ok=='True' else 'Incorrect'}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='background:#e8f0f8;padding:10px;border-radius:6px;"
                f"font-size:13px;border-left:3px solid #4c78a8;white-space:pre-wrap;'>"
                f"{_ann_get('NicolaySynthesisAssessmentRaw')}</div>",
                unsafe_allow_html=True,
            )

        with st.expander("4. FinalAnswerText — primary for FA and CA", expanded=True):
            st.markdown(
                "<div style='font-size:11px;color:#666;margin-bottom:6px;'>"
                "Hard rule: QuotesFabricated ≥ 1 caps CA at 0.50.</div>",
                unsafe_allow_html=True,
            )
            qv_ann = st.columns(5)
            for c, (lbl, col) in zip(qv_ann, [
                ("Verified", "QuotesVerified"), ("Approx", "QuotesApprox"),
                ("Displaced", "QuotesDisplaced"), ("Fabricated", "QuotesFabricated"),
                ("Mislabeled", "QuotesMislabeled"),
            ]):
                val   = _ann_get(col, "0")
                color = "#c1121f" if lbl in ("Fabricated", "Displaced") and val not in ("0", "--") else "#333"
                c.markdown(
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

        with st.expander("5. LLM Auto-Scores (reveal after scoring)", expanded=False):
            st.warning("Blind protocol: score the response above before expanding.")
            sc_cols = st.columns(5)
            for c, (col, lbl) in zip(sc_cols, [
                ("RubricFactualAccuracy", "FA"), ("RubricCitationAccuracy", "CA"),
                ("RubricHistoriographicalDepth", "HD"), ("RubricEpistemicCalibration", "EC"),
                ("RubricTotal", "Total"),
            ]):
                c.metric(lbl, _ann_get(col))
            for rat_col, lbl, bg in [
                ("RationaleFactualAccuracy",        "FA",   "#eef7ee"),
                ("RationaleCitationAccuracy",       "CA",   "#e8f0f8"),
                ("RationaleHistoriographicalDepth", "HD",   "#f8f4ef"),
                ("RationaleEpistemicCalibration",   "EC",   "#fef9ec"),
                ("RationaleHayDiagnostic",          "Hay",  "#fff3cd"),
            ]:
                val = _ann_get(rat_col)
                if val != "--":
                    st.markdown(
                        f"<div style='background:{bg};padding:7px;border-radius:5px;"
                        f"font-size:12px;margin-bottom:5px;'>"
                        f"<b>{lbl}:</b> {val}</div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("---")
        st.subheader("Your Scores")

        dim_guidance = {
            "FA": ("Factual Accuracy",
                   "1.0 All correct | 0.75 Minor errors | 0.5 Significant errors | 0.25 Major distortions | 0.0 Fabrication"),
            "CA": ("Citation Accuracy",
                   "1.0 All verified | 0.75 Minor issues | 0.5 Displaced/unsupported | 0.0 Fabricated · Hard rule: Fabricated≥1 caps at 0.50"),
            "HD": ("Historiographical Depth",
                   "1.0 Sophisticated | 0.75 Solid framing | 0.5 Descriptive only | 0.25 Superficial | 0.0 No engagement"),
            "EC": ("Epistemic Calibration",
                   "1.0 Explicitly calibrated | 0.75 Generally appropriate | 0.5 Overconfident | 0.0 Systematically overconfident"),
        }

        score_vals = {}
        for dim in ["FA", "CA", "HD", "EC"]:
            label, rubric = dim_guidance[dim]
            st.markdown(f"**{label}**")
            st.caption(rubric)
            score_vals[dim] = st.select_slider(
                f"Score {label}",
                options=[0.0, 0.25, 0.50, 0.75, 1.0],
                value=prior.get(dim, 0.75),
                key=f"ann_slider_{qid}_{dim}",
                label_visibility="collapsed",
            )

        total_score = sum(score_vals[d] for d in ["FA", "CA", "HD", "EC"])
        st.markdown(f"**Total: {total_score:.2f} / 4.00**")

        notes = st.text_area(
            "Notes", value=prior.get("notes", ""),
            height=70, key=f"ann_notes_{qid}",
        )

        save_col, clear_col = st.columns([2, 1])
        with save_col:
            if st.button("Save Score", type="primary", key="ann_save"):
                st.session_state.annotation_scores[qid] = {
                    **score_vals, "Total": total_score,
                    "notes": notes, "timestamp": datetime.now().isoformat(),
                }
                st.success(f"Saved: {qid} — Total {total_score:.2f}/4.00")
                if idx < n_total - 1:
                    st.session_state.annotation_idx += 1
                    st.rerun()
        with clear_col:
            if st.button("Clear Score", key="ann_clear"):
                if qid in st.session_state.annotation_scores:
                    del st.session_state.annotation_scores[qid]
                    st.rerun()

        # Export
        st.markdown("---")
        st.subheader("Export")
        if st.session_state.annotation_scores:
            export_rows = []
            for scored_qid, scores in st.session_state.annotation_scores.items():
                row_match = ann_df[ann_df["QueryID"] == scored_qid]
                if not row_match.empty:
                    llm_fa  = float(row_match.iloc[0].get("RubricFactualAccuracy",        0) or 0)
                    llm_ca  = float(row_match.iloc[0].get("RubricCitationAccuracy",        0) or 0)
                    llm_hd  = float(row_match.iloc[0].get("RubricHistoriographicalDepth",  0) or 0)
                    llm_ec  = float(row_match.iloc[0].get("RubricEpistemicCalibration",    0) or 0)
                    llm_tot = round(llm_fa + llm_ca + llm_hd + llm_ec, 2)
                    h_tot   = scores["Total"]
                    export_rows.append({
                        "QueryID": scored_qid,
                        "Category": row_match.iloc[0]["Category"],
                        "HumanFA": scores["FA"], "HumanCA": scores["CA"],
                        "HumanHD": scores["HD"], "HumanEC": scores["EC"],
                        "HumanTotal": h_tot,
                        "LLM_FA": llm_fa, "LLM_CA": llm_ca,
                        "LLM_HD": llm_hd, "LLM_EC": llm_ec,
                        "LLM_Total": llm_tot,
                        "Delta": round(h_tot - llm_tot, 2),
                        "Notes": scores.get("notes", ""),
                        "Timestamp": scores.get("timestamp", ""),
                    })

            export_df = pd.DataFrame(export_rows)
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download annotation CSV ({len(export_rows)} responses)",
                data=csv_bytes,
                file_name=f"human_annotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

            if len(export_rows) > 1:
                st.subheader("Human vs. LLM")
                correction_rate = (export_df["Delta"].abs() > 0.24).mean()
                mean_delta      = export_df["Delta"].mean()
                cr1, cr2 = st.columns(2)
                cr1.metric("Correction rate (|Δ| > 0.25)", f"{correction_rate:.1%}")
                cr2.metric("Mean delta (Human − LLM)",      f"{mean_delta:+.3f}")
                dim_cols = st.columns(4)
                for dc, dim in zip(dim_cols, ["FA", "CA", "HD", "EC"]):
                    if f"Human{dim}" in export_df.columns and f"LLM_{dim}" in export_df.columns:
                        d = (export_df[f"Human{dim}"] - export_df[f"LLM_{dim}"]).mean()
                        dc.metric(dim, f"{d:+.3f}")
                st.dataframe(
                    export_df[["QueryID", "HumanTotal", "LLM_Total", "Delta", "Notes"]].style.format({
                        "HumanTotal": "{:.2f}", "LLM_Total": "{:.2f}", "Delta": "{:+.2f}",
                    }).apply(_apply_gradient, vmin=-1, vmax=1,
                             low_rgb=_RdBu[0], high_rgb=_RdBu[1], subset=["Delta"]),
                    use_container_width=True,
                )
        else:
            st.caption("No scores saved yet.")

    else:
        st.markdown(
            "1. Upload any `merged_run_N.csv` from the benchmark directory.\n"
            "2. Score each dimension using the inline rubric.\n"
            "3. Expand LLM Auto-Scores only after scoring.\n"
            "4. Export the comparison CSV when complete."
        )

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "HonestAbe Benchmark Viewer · H4N4 ada-002 · 886-chunk corpus · rerank-v4.0-pro · k=5  \n"
    "Canonical result: **2.883 [2.802, 2.961]** (95% CI, bootstrap n=1,000, 5 runs, n=120 obs)  \n"
    "Hutchinson, D. (forthcoming). *Digital Humanities Quarterly.*"
)
