"""
Nicolay Formal Benchmark — Streamlit App
=========================================
Runs the 25-query formal benchmark against the Nicolay RAG pipeline.
Captures all defined metrics: Hay-layer, retrieval-layer, Nicolay-layer, quote verification,
BLEU/ROUGE NLP scores, and a manual qualitative rubric.

System state: Hay v3 + Nicolay v3 + Cohere rerank-v4.0-pro + full chunk text + k=5
Corpus: lincoln_speech_corpus_reindex_keep.json (886 chunks)

v7.7 changes:
  - Quote verification pipeline unified with Streamlit app v1.9.
  - Sliding-window search (_sliding_window_verify) ported; catches mid-sentence
    quoting and light sentence condensation the old stack missed.
  - Stage ordering fixed: ALL cited-chunk methods (strict → loose → sliding →
    token-coverage) now exhaust before any corpus-wide displacement scan.
    Prevents false displacements where a quote IS in the cited chunk but needed
    loose/sliding to match while another chunk matched strict segments first.
  - Fuzzy-punctuation / edit-distance anchor stage removed (subsumed by sliding
    window + loose segments).
  - Return dict schema and outcome strings unchanged — downstream consumers
    (verify_all_quotes, result aggregation, debug table) are fully compatible.

v5.4 changes:
  - Benchmark expanded from 16 to 25 runnable questions (corpus-validated, 2026-03-07)
  - Q5 revised: Dred Scott nationalization argument (replaces diffuse broad framing)
  - Q11 revised: Constitutional obligation / Fugitive Slave Law (replaces abstract liberty/law)
  - New questions: FR-2, AN-5, CA-5, CA-6, S-4, S-5, RC-3, RC-4, RC-5
  - RC-2 included but flagged blocked (Last Public Address absent from corpus)
  - Groups: core (retained), revised (Q5-rev, Q11-rev), new (FR-2 through RC-5)
"""

import streamlit as st
import json
import re
import os
import csv
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
import msgpack
import hashlib


def _decode_bytes(obj):
    """Recursively convert bytes to str for msgpack-loaded objects."""
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8")
        except Exception:
            return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return { _decode_bytes(k): _decode_bytes(v) for k,v in obj.items() }
    if isinstance(obj, list):
        return [ _decode_bytes(x) for x in obj ]
    return obj


def load_corpus_any(corpus_path: str) -> list[dict]:
    """Load corpus from JSON or msgpack and return list-of-dict items."""
    p = Path(corpus_path)
    if not p.exists():
        raise FileNotFoundError(f"Corpus path not found: {corpus_path}")
    suffix = p.suffix.lower()

    if suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            data = data["data"]
        if not isinstance(data, list):
            raise ValueError(f"JSON corpus must be a list of records; got {type(data)}")
        return data

    if suffix in {".msgpack", ".mpack"}:
        with open(p, "rb") as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            objs = [o for o in unpacker]
        if not objs:
            raise ValueError("Msgpack corpus is empty (no top-level objects).")
        data = objs[-1]  # tolerate header/metadata objects in the stream
        data = _decode_bytes(data)
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            data = data["data"]
        if not isinstance(data, list):
            raise ValueError(f"Msgpack corpus must be a list of records; got {type(data)}")
        return data

    # Fallback: try JSON then msgpack
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    with open(p, "rb") as f:
        data = msgpack.unpackb(f.read(), raw=False)
    data = _decode_bytes(data)
    if not isinstance(data, list):
        raise ValueError(f"Unsupported corpus format at {corpus_path} (suffix={suffix})")
    return data


def _safe_preview(text: str, n: int = 160) -> str:
    """One-line preview for debugging tables."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return (text[:n] + "…") if len(text) > n else text


def _sha1_10(text: str) -> str:
    """Short content fingerprint used to spot corpus/index mismatches."""
    if text is None:
        text = ""
    h = hashlib.sha1(str(text).encode("utf-8", errors="replace")).hexdigest()
    return h[:10]


def _chunk_signature(corpus: dict, tid: int) -> dict:
    """Summarize what *this corpus* thinks chunk `tid` is."""
    chunk = corpus.get(tid) if corpus else None
    if not chunk:
        return {
            "tid": tid, "exists": False, "source": "", "text_id": "",
            "len": 0, "sha1_10": "", "preview": ""
        }
    full_text = chunk.get("full_text", chunk.get("text", "")) or ""
    return {
        "tid": tid,
        "exists": True,
        "source": str(chunk.get("source", "")),
        "text_id": str(chunk.get("text_id", "")),
        "len": int(len(full_text)),
        "sha1_10": _sha1_10(full_text),
        "preview": _safe_preview(full_text, 160),
    }


def _extract_int_from_text_id(val) -> Optional[int]:
    """Parse an int from 'Text #: 420' or '420' or any string containing digits."""
    if val is None:
        return None
    s = str(val).strip()
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def _extract_cited_ids(nicolay_output: dict) -> list[int]:
    """Extract all integer Text IDs Nicolay claims in Match Analysis."""
    ma = nicolay_output.get("Match Analysis", {}) if isinstance(nicolay_output, dict) else {}
    if not isinstance(ma, dict):
        return []
    out = []
    for _, mv in ma.items():
        if isinstance(mv, dict):
            num = _extract_int_from_text_id(mv.get("Text ID", ""))
            if num is not None:
                out.append(num)
    # Preserve order but unique
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq



# NLP evaluation
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer

# ── U12 confidence signals (ported from main app, pure computation) ────────────────────

def _u12_extract_synth_type_num(synthesis_assessment_str):
    """Extract integer synthesis type (1-5) from Nicolay's synthesis_assessment field."""
    m = re.search(r'Type\s+(\d)', str(synthesis_assessment_str), re.IGNORECASE)
    return int(m.group(1)) if m else 3


def _u12_compute_corpus_grounding(final_answer_text, reranked_list, top_n=3):
    """ROUGE-1/2 between FinalAnswer and top-N reranked chunk Key Quote text."""
    if not final_answer_text or not reranked_list:
        return None
    ref_chunks = reranked_list[:top_n]
    reference_text = " ".join(
        r.get("Key Quote", "") or r.get("full_text", "") or r.get("Full Text", "")
        for r in ref_chunks
    ).strip()
    if not reference_text:
        return None
    try:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
        s = scorer.score(reference_text, final_answer_text)
        return {
            "rouge1": round(s["rouge1"].fmeasure, 4),
            "rouge2": round(s["rouge2"].fmeasure, 4),
        }
    except Exception:
        return None


def _u12_analyze_reranker_scores(reranked_list):
    """Score ceiling, spread, source diversity, calibration warning (flat+high = bad)."""
    if not reranked_list:
        return None
    scores = [
        r.get("Relevance Score", r.get("relevance_score", 0))
        for r in reranked_list[:5]
    ]
    scores = [s for s in scores if s is not None]
    if not scores:
        return None

    # n_distinct_sources: prefer source-string diversity when available;
    # fall back to counting distinct text_id / doc_id values.
    # The reranked_proxy built in compute_confidence_signals() carries
    # Source="" (unavailable at that stage), so we use doc-ID diversity instead.
    sources = [r.get("Source", "") or r.get("source", "") for r in reranked_list[:5]]
    n_distinct_by_source = len(set(s for s in sources if s))
    if n_distinct_by_source == 0:
        # Fall back to distinct doc IDs (text_id_num / Key Quote as proxy-uniqueness)
        doc_ids = [r.get("doc_id") or r.get("text_id_num") or r.get("Key Quote", "")
                   for r in reranked_list[:5]]
        n_distinct = len(set(d for d in doc_ids if d))
        # If still 0, the list length itself is the lower-bound (all entries distinct)
        if n_distinct == 0:
            n_distinct = len(reranked_list[:5])
    else:
        n_distinct = n_distinct_by_source

    max_s  = max(scores)
    min_s  = min(scores)
    spread = round(max_s - min_s, 4)
    calib_warning = (max_s >= 0.70 and spread <= 0.08)
    return {
        "max_score":          round(max_s, 4),
        "min_score":          round(min_s, 4),
        "spread":             spread,
        "n_distinct_sources": n_distinct,
        "calibration_warning": calib_warning,
    }


def _u12_compute_overall_confidence(
    verified_quotes, displaced_quotes, unverified_quotes,
    rouge_data, reranker_data, synth_type_num
):
    """Return (rating, icon, explanation): 'high' / 'medium' / 'low'."""
    has_unverified = len(unverified_quotes) > 0
    calib_warning  = reranker_data.get("calibration_warning", False) if reranker_data else False
    r1             = rouge_data.get("rouge1", 0.0) if rouge_data else 0.0
    direct_type    = synth_type_num in (1, 2)
    absence_type   = synth_type_num == 3

    if has_unverified:
        return ("low", "🔴",
                "One or more quotes could not be verified against the corpus.")
    if calib_warning:
        return ("low", "🔴",
                "High-confidence retrieval with flat spread — corpus may lack ideal documents.")
    if direct_type and r1 < 0.25:
        return ("low", "🔴",
                "Type 1/2 response with weak corpus grounding (ROUGE-1 < 0.25).")
    if absence_type and not has_unverified and not calib_warning:
        return ("high", "✅",
                "Absence response: corpus limits correctly identified, quotes verified.")
    rouge_threshold = 0.45 if direct_type else 0.30
    if not has_unverified and not calib_warning and r1 >= rouge_threshold:
        return ("high", "✅",
                "All quotes verified and response closely tracks retrieved sources.")
    return ("medium", "⚠️",
            "Mixed signals — review source passages before relying on specific claims.")


def _count_distinct_sources_from_corpus(retrieved_ids, corpus):
    """
    Count distinct *speech-level* source documents among retrieved chunk IDs.

    Chunks share the same source string when they come from the same speech
    (e.g. chunks 412, 413, 414 all carry "Fourth Annual Message. December 6, 1864").
    This is the correct diversity metric: 3 chunks from one speech = 1 distinct source.

    Falls back to len(retrieved_ids) if corpus is unavailable.
    """
    if not corpus or not retrieved_ids:
        return len(retrieved_ids)
    sources = set()
    for tid in retrieved_ids:
        chunk = corpus.get(int(tid)) if isinstance(tid, (int, str)) else None
        if chunk:
            src_str = (chunk.get("source") or chunk.get("Source") or "").strip()
            if src_str:
                sources.add(src_str)
    # If no source strings found (corpus keying issue), fall back to chunk count
    return len(sources) if sources else len(retrieved_ids)


def _count_distinct_sources_from_qv(qv_list):
    """
    Derive distinct speech-level source count from stored quote_verification results.
    Each qv item carries cited_chunk_source from the verifier — strip chunk-level
    metadata to get the speech title and count unique speeches.
    Available without corpus I/O; used in display panel and as fallback.
    """
    sources = set()
    for qv in qv_list:
        src = (qv.get("cited_chunk_source") or "").strip()
        if src:
            # Normalize: "Source: Fourth Annual Message. December 6, 1864" → title only
            src = src.replace("Source:", "").strip()
            sources.add(src)
    return len(sources) if sources else 0


def compute_confidence_signals(qresult, corpus=None):
    """
    Compute all U12 confidence signals from a completed qresult dict.
    Returns a flat dict of confidence_* fields ready for storage in results/CSV/Sheets.

    corpus: optional dict {int_id: chunk_dict} — when provided, enables accurate
            speech-level source diversity counting (chunks from the same speech
            counted once). Pass corpus_for_verify from run_pipeline_for_query.
            If omitted, falls back to qv cited_chunk_source strings.
    """
    final_text   = qresult.get("nicolay_final_answer_text", "") or ""
    synth_raw    = qresult.get("nicolay_synthesis_assessment_raw", "") or ""
    synth_type   = _u12_extract_synth_type_num(synth_raw)

    qv_list    = qresult.get("quote_verification", [])
    verified   = [q for q in qv_list if q.get("outcome") in ("verified", "approximate_quote")]
    displaced  = [q for q in qv_list if q.get("outcome") in ("displacement", "approximate_displacement")]
    unverified = [q for q in qv_list if q.get("outcome") in ("fabrication", "source_mislabeled")]

    # Build a thin reranked_list from stored scores for reranker analysis
    reranker_scores = qresult.get("reranker_scores", [])[:5]
    retrieved_ids   = qresult.get("retrieved_doc_ids", [])[:5]
    nicolay_out     = qresult.get("nicolay_output", {}) or {}
    match_analysis  = nicolay_out.get("Match Analysis", {}) if isinstance(nicolay_out, dict) else {}

    reranked_proxy = []
    for s, tid in zip(reranker_scores, retrieved_ids):
        kq = ""
        if isinstance(match_analysis, dict):
            for mv in match_analysis.values():
                if isinstance(mv, dict):
                    mid = _extract_int_from_text_id(mv.get("Text ID", ""))
                    if mid == tid:
                        kq = mv.get("Key Quote", "") or ""
                        break
        reranked_proxy.append({"Relevance Score": s, "Source": "", "Key Quote": kq, "doc_id": tid})

    rouge_data    = _u12_compute_corpus_grounding(final_text, reranked_proxy) if final_text else None
    reranker_data = _u12_analyze_reranker_scores(reranked_proxy)

    # Distinct speech-level sources: corpus lookup > qv fallback > chunk count
    if corpus:
        n_distinct = _count_distinct_sources_from_corpus(retrieved_ids, corpus)
    else:
        n_distinct = _count_distinct_sources_from_qv(qv_list)
        if n_distinct == 0:
            n_distinct = (reranker_data or {}).get("n_distinct_sources") or len(retrieved_ids)

    rating, icon, explanation = _u12_compute_overall_confidence(
        verified, displaced, unverified, rouge_data, reranker_data, synth_type
    )

    return {
        "confidence_rating":        rating,
        "confidence_icon":          icon,
        "confidence_explanation":   explanation,
        "confidence_synth_type":    synth_type,
        "confidence_rouge1":        (rouge_data or {}).get("rouge1"),
        "confidence_rouge2":        (rouge_data or {}).get("rouge2"),
        "confidence_calib_warning": (reranker_data or {}).get("calibration_warning", False),
        "confidence_spread":        (reranker_data or {}).get("spread"),
        "confidence_n_sources":     n_distinct,
        "confidence_max_score":     (reranker_data or {}).get("max_score"),
    }


# API clients
import openai
import cohere

# Download NLTK data quietly
for resource in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nicolay Benchmark",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — archival/documentary aesthetic appropriate for a Lincoln corpus tool
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Source+Sans+3:wght@300;400;600&family=Source+Code+Pro:wght@400&display=swap');

    html, body, [class*="css"] {
        font-family: 'Source Sans 3', sans-serif;
    }
    h1, h2, h3 { font-family: 'Playfair Display', serif; }
    .stCode, code, pre { font-family: 'Source Code Pro', monospace; font-size: 0.85em; }

    .metric-card {
        background: #f8f6f1;
        border-left: 3px solid #8b6f47;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 2px;
    }
    .pass { color: #2d6a2d; font-weight: 600; }
    .fail { color: #8b1a1a; font-weight: 600; }
    .warn { color: #7a5c00; font-weight: 600; }
    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
        color: #3d2b1f;
        border-bottom: 1px solid #c9b99a;
        padding-bottom: 0.25rem;
        margin: 1rem 0 0.5rem 0;
    }
    .query-badge {
        background: #3d2b1f;
        color: #f8f6f1;
        padding: 0.15rem 0.5rem;
        border-radius: 3px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'Source Code Pro', monospace;
    }
    .bleu-rouge-table th { background: #3d2b1f; color: #f8f6f1; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL PAIRS REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
# Each entry maps a display label to its hay/nicolay model IDs and a short
# label tag used in CSV/Sheets ModelConfigTag values.
# Add new pairs here as additional fine-tuning iterations complete.

MODEL_PAIRS = {
    "H3N3 — Hay v3 + Nicolay v3 (baseline)": {
        "hay":     "ft:gpt-4.1-mini-2025-04-14:personal:hays-v3:DEcb9s4u",
        "nicolay": "ft:gpt-4.1-mini-2025-04-14:personal:nicolay-v3:DEccNnWt",
        "label":   "H3N3",
    },
    "H4N3 — Hay v4 + Nicolay v3": {
        "hay":     "ft:gpt-4.1-mini-2025-04-14:personal:hays-v4:DI4PJ4Zt",
        "nicolay": "ft:gpt-4.1-mini-2025-04-14:personal:nicolay-v3:DEccNnWt",
        "label":   "H4N3",
    },
    "H4N4 — Hay v4 + Nicolay v4": {
        "hay":     "ft:gpt-4.1-mini-2025-04-14:personal:hays-v4:DI4PJ4Zt",
        "nicolay": "ft:gpt-4.1-mini-2025-04-14:personal:nicolay-v4:DIPD9hh5",
        "label":   "H4N4",
    },
}

# Default active pair — overridden by sidebar selection at runtime.
# These module-level variables remain the single source of truth used by
# run_pipeline_for_query(), log_query_to_sheets(), and empty_results().
_DEFAULT_PAIR_KEY = "H3N3 — Hay v3 + Nicolay v3 (baseline)"
HAY_MODEL     = MODEL_PAIRS[_DEFAULT_PAIR_KEY]["hay"]
NICOLAY_MODEL = MODEL_PAIRS[_DEFAULT_PAIR_KEY]["nicolay"]

COHERE_RERANK_MODEL = "rerank-v4.0-pro"
RERANK_K = 5

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK QUERY LIST
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_QUERIES = [
    # ── Factual Retrieval (5) ─────────────────────────────────────────────────
    {
        "id": "Q1", "group": "core",
        "query": "Lincoln noted how many voters from Kansas and Nevada participated in the 1864 election",
        "category": "factual_retrieval",
        "expected_hay_type": "A", "expected_nicolay_type": "T1",
        "ideal_docs_new": [413, 414], "ideal_docs_original": [77], "ideal_docs_count": 2,
        "critical_missing_evidence": None,
        "watchlist": ["Hay spurious field", "Hay hallucination propagation (33,762 figure)"],
    },
    {
        "id": "Q2", "group": "core",
        "query": "How does Russia factor into Lincoln's speeches?",
        "category": "factual_retrieval",
        "expected_hay_type": "D", "expected_nicolay_type": "T3",
        "ideal_docs_new": [305, 351, 381], "ideal_docs_original": [52, 63, 68], "ideal_docs_count": 3,
        "critical_missing_evidence": "Eduard de Stoeckl, Alaska purchase negotiations",
        "watchlist": ["Chunk 351 Russia/Japan retrieval gap"],
    },
    {
        "id": "Q3", "group": "core",
        "query": "In what ways did Lincoln highlight the contributions of immigrants during the Civil War?",
        "category": "factual_retrieval",
        "expected_hay_type": "D", "expected_nicolay_type": "T2",
        "ideal_docs_new": [390, 349, 350], "ideal_docs_original": [62, 63, 69], "ideal_docs_count": 3,
        "critical_missing_evidence": None,
        "watchlist": [],
    },
    {
        "id": "R3", "group": "core",
        "query": "How did Lincoln report on the financial condition of the Post Office Department during the war?",
        "category": "factual_retrieval",
        "expected_hay_type": "D", "expected_nicolay_type": "T2",
        "ideal_docs_new": [311, 312, 364, 365, 401], "ideal_docs_original": [55, 56, 64, 65, 71], "ideal_docs_count": 5,
        "critical_missing_evidence": None,
        "watchlist": ["Numerical progression: $8.3M → near-self-sustaining → $12.4M"],
    },
    {
        "id": "FR-2", "group": "new",
        "query": "How did Lincoln characterize the relationship between wartime taxation, public debt, and the financial obligations of citizens in his Annual Messages to Congress?",
        "category": "factual_retrieval",
        "expected_hay_type": "D", "expected_nicolay_type": "T3",
        "ideal_docs_new": [249, 309, 310, 393, 395], "ideal_docs_original": None, "ideal_docs_count": 5,
        "critical_missing_evidence": None,
        "watchlist": [
            "Tests fiscal-civic argument synthesis vs. raw number retrieval",
            "Docs 357/392 (raw fiscal tables) may retrieve but are not ideal targets — watch for T1 overshoot",
            "Doc 395: 'citizens cannot be much oppressed by a debt which they owe to themselves'",
        ],
    },
    # ── Analysis (5) ─────────────────────────────────────────────────────────
    {
        "id": "Q4", "group": "core",
        "query": "How did Lincoln incorporate allusions in his Second Inaugural Address?",
        "category": "analysis",
        "expected_hay_type": "A", "expected_nicolay_type": "T2",
        "ideal_docs_new": [419, 420, 421, 422], "ideal_docs_original": [77, 78], "ideal_docs_count": 4,
        "critical_missing_evidence": None,
        "watchlist": ["Biblical allusion depth in chunks 421-422", "Fabrication risk at 562-word response edge"],
    },
    {
        "id": "Q5", "group": "revised",
        "query": "How did Lincoln argue in the 1858 debates that the Dred Scott decision was part of a larger design to nationalize slavery throughout the United States?",
        "category": "analysis",
        "expected_hay_type": "D", "expected_nicolay_type": "T3",
        "ideal_docs_new": [89, 96, 97, 487, 619, 726], "ideal_docs_original": [15, 16, 17], "ideal_docs_count": 6,
        "critical_missing_evidence": None,
        "watchlist": [
            "Revised from broad Dred Scott framing (Run 0: reranker mean=0.949, ROUGE-1 ratio=1.231)",
            "Nationalization argument: docs 487 (Freeport), 619 (Galesburg), 726 (Quincy)",
            "House Divided conspiracy framing: docs 89, 96, 97",
        ],
    },
    {
        "id": "Q6", "group": "core",
        "query": "How did Lincoln explain his administration's approach to the Fugitive Slave Law?",
        "category": "analysis",
        "expected_hay_type": "A", "expected_nicolay_type": "T1",
        "ideal_docs_new": [185, 191, 197, 202], "ideal_docs_original": [33, 34, 35, 36], "ideal_docs_count": 4,
        "critical_missing_evidence": None,
        "watchlist": ["Run 0 calibration decoupling case: P@5=0 yet rubric 3.25 — retain for article evidence"],
    },
    {
        "id": "R1", "group": "core",
        "query": "How did Lincoln justify the naval blockade of Confederate ports?",
        "category": "analysis",
        "expected_hay_type": "D", "expected_nicolay_type": "T3",
        "ideal_docs_new": [218, 272, 300, 345, 359], "ideal_docs_original": [39, 48, 53, 61, 63], "ideal_docs_count": 5,
        "critical_missing_evidence": "Trent Affair entirely absent from corpus — Mason, Slidell, San Jacinto references NOT IN CORPUS",
        "watchlist": ["Trent Affair gap recognition by Nicolay"],
    },
    {
        "id": "AN-5", "group": "new",
        "query": "How did Lincoln argue in his First Annual Message that the relationship between labor and capital in a free society differed fundamentally from the assumptions underlying the slave-labor system?",
        "category": "analysis",
        "expected_hay_type": "A", "expected_nicolay_type": "T2",
        "ideal_docs_new": [279, 280, 281], "ideal_docs_original": None, "ideal_docs_count": 3,
        "critical_missing_evidence": None,
        "watchlist": [
            "Tight ideal doc set: 3 consecutive First Annual Message chunks only",
            "Doc 280: 'Labor is prior to, and independent of, capital' — key anchor phrase",
            "Peoria has moral anti-slavery argument but NOT the capital-labor economic framing — watch for diffuse retrieval",
        ],
    },
    # ── Comparative Analysis (6) ──────────────────────────────────────────────
    {
        "id": "Q7", "group": "core",
        "query": "How did Lincoln's discussion of slavery evolve between his House Divided speech and his Second Inaugural Address?",
        "category": "comparative_analysis",
        "expected_hay_type": "E", "expected_nicolay_type": "T4",
        "ideal_docs_new": [88, 95, 101, 419, 420, 421, 422], "ideal_docs_original": [15, 16, 17, 77, 78], "ideal_docs_count": 7,
        "critical_missing_evidence": None,
        "watchlist": ["Fabrication confirmed Run 0 (QuotesFabricated=1)", "T4→T2 floor — key failure case", "Hay Contrastive over-classification"],
    },
    {
        "id": "Q8", "group": "core",
        "query": "How did Lincoln's justification for the Civil War evolve between his First Inaugural and Second Inaugural?",
        "category": "comparative_analysis",
        "expected_hay_type": "E", "expected_nicolay_type": "T4",
        "ideal_docs_new": [185, 191, 197, 202, 419, 420, 421, 422], "ideal_docs_original": [33, 34, 35, 36, 77, 78], "ideal_docs_count": 8,
        "critical_missing_evidence": None,
        "watchlist": ["T4→T2 floor pattern from Run 0", "Hay Contrastive over-classification"],
    },
    {
        "id": "Q9", "group": "core",
        "query": "How did Lincoln's views of African American soldiers change or remain the same over time?",
        "category": "comparative_analysis",
        "expected_hay_type": "D", "expected_nicolay_type": "T3",
        "ideal_docs_new": [288, 295, 367, 374], "ideal_docs_original": [51, 52, 65, 66], "ideal_docs_count": 4,
        "critical_missing_evidence": None,
        "watchlist": ["T3→T2 floor from Run 0", "Docs 288/367 systematically missed in Run 0"],
    },
    {
        "id": "R2", "group": "core",
        "query": "How did Lincoln describe U.S. relations with Great Britain during the Civil War?",
        "category": "comparative_analysis",
        "expected_hay_type": "D", "expected_nicolay_type": "T3",
        "ideal_docs_new": [242, 243, 247, 300, 301, 345, 346, 388], "ideal_docs_original": [43, 44, 53, 54, 61, 62, 69], "ideal_docs_count": 8,
        "critical_missing_evidence": "Trent Affair absent (most diplomatically significant U.S.-British episode of the war)",
        "watchlist": ["Trent Affair gap recognition by Nicolay", "T3→T2 floor from Run 0"],
    },
    {
        "id": "CA-5", "group": "new",
        "query": "How did Lincoln's characterization of the South and the causes of the conflict differ between his First Inaugural Address and his Second Inaugural Address?",
        "category": "comparative_analysis",
        "expected_hay_type": "E", "expected_nicolay_type": "T3",
        "ideal_docs_new": [193, 195, 420, 421, 422], "ideal_docs_original": None, "ideal_docs_count": 5,
        "critical_missing_evidence": None,
        "watchlist": [
            "Doc 185 (First Inaugural opening procedural chunk) excluded from ideal set — no argumentative content",
            "Docs 193/195 carry conciliatory framing; docs 420-422 carry Second Inaugural causal argument",
            "Both ends reliably retrieved in Run 0 — cleaner T3 test than Q7/Q8",
        ],
    },
    {
        "id": "CA-6", "group": "new",
        "query": "How did Lincoln constitutionally justify his suspension of habeas corpus and exercise of war powers in his 1861 message to Congress, and how did he reaffirm this authority in later wartime addresses?",
        "category": "comparative_analysis",
        "expected_hay_type": "D", "expected_nicolay_type": "T3",
        "ideal_docs_new": [214, 219, 221, 380], "ideal_docs_original": None, "ideal_docs_count": 4,
        "critical_missing_evidence": None,
        "watchlist": [
            "Strong 1861 anchor: docs 219 (suspension announcement), 221 (constitutional justification), 214 (war power)",
            "Doc 380 (3rd Annual Message): 1863 reaffirmation — 'war power is still our main reliance'",
            "1864 evolution claim: corpus thin — well-calibrated response should hedge; calibration test",
        ],
    },
    # ── Synthesis (5) ────────────────────────────────────────────────────────
    {
        "id": "Q10", "group": "core",
        "query": "How did Lincoln develop the theme of divine providence throughout his wartime speeches?",
        "category": "synthesis",
        "expected_hay_type": "D", "expected_nicolay_type": "T4",
        "ideal_docs_new": [298, 418, 419, 420, 421, 422], "ideal_docs_original": [53, 76, 77, 78], "ideal_docs_count": 6,
        "critical_missing_evidence": None,
        "watchlist": [
            "Lowest Run 0 scorer (2.25/4.0) — doc misattribution confirmed",
            "'The Almighty has His own purposes' placed in wrong speech — key failure case for article",
            "T4→T2 floor compounds with factual error",
        ],
    },
    {
        "id": "Q11", "group": "revised",
        "query": "How did Lincoln use the concept of constitutional obligation to justify enforcement of laws he personally opposed, such as the Fugitive Slave Law?",
        "category": "synthesis",
        "expected_hay_type": "D", "expected_nicolay_type": "T4",
        "ideal_docs_new": [153, 159, 185, 191, 418, 419], "ideal_docs_original": [27, 28, 33, 34, 76, 77], "ideal_docs_count": 6,
        "critical_missing_evidence": None,
        "watchlist": [
            "Revised from abstract liberty/law framing (Run 0: P@5=0, R@5=0 yet rubric 3.25 — measuring confabulation)",
            "Concrete argumentative move: right answer exists; harder to confabulate",
            "Docs 153/159 (Cooper Union) systematically missed — watch retrieval pattern",
        ],
    },
    {
        "id": "Q12", "group": "core",
        "query": "What themes did Lincoln consistently employ when discussing the Constitution's relationship to slavery?",
        "category": "synthesis",
        "expected_hay_type": "D", "expected_nicolay_type": "T4",
        "ideal_docs_new": [153, 159, 185, 191], "ideal_docs_original": [27, 28, 33, 34], "ideal_docs_count": 4,
        "critical_missing_evidence": None,
        "watchlist": ["High reranker/low P@5 calibration decoupling case from Run 0", "Lincoln-Douglas Debate chunk retrieval", "Cooper Union additional chunks"],
    },
    {
        "id": "S-4", "group": "new",
        "query": "How did Lincoln's use of the Declaration of Independence as a founding argument shift from his pre-war debates with Douglas to his wartime addresses, and what did this shift accomplish rhetorically?",
        "category": "synthesis",
        "expected_hay_type": "E", "expected_nicolay_type": "T4",
        "ideal_docs_new": [44, 45, 418, 624, 626, 628], "ideal_docs_original": None, "ideal_docs_count": 6,
        "critical_missing_evidence": None,
        "watchlist": [
            "Peoria Declaration passages: docs 44/45 (pre-war anchor)",
            "Galesburg 5th Debate (docs 624-632): richest for Lincoln's extended Declaration defense vs. Douglas",
            "Doc 418 (Gettysburg): wartime reinterpretation capstone — single chunk",
            "HD discriminator: Garry Wills / Declaration-as-reinterpretation thesis standard in literature",
        ],
    },
    {
        "id": "S-5", "group": "new",
        "query": "How did Lincoln argue that the Civil War was fundamentally a test of whether democratic self-government could survive, and where did he make this case most explicitly?",
        "category": "synthesis",
        "expected_hay_type": "D", "expected_nicolay_type": "T4",
        "ideal_docs_new": [46, 47, 48, 239, 418], "ideal_docs_original": None, "ideal_docs_count": 5,
        "critical_missing_evidence": None,
        "watchlist": [
            "Pre-war anchor strong: Peoria docs 43-50 (self-government experiment framing)",
            "Wartime anchor thinner: doc 239 (July 4th Message 1861) is the strongest wartime articulation",
            "Partial retrieval likely — calibration interest: does system hedge appropriately?",
        ],
    },
    # ── Race and Citizenship (4 runnable + 1 blocked) ────────────────────────
    {
        "id": "Q13", "group": "core",
        "query": "How did Lincoln's views on African American citizenship and racial equality evolve across his speeches?",
        "category": "race_citizenship",
        "expected_hay_type": "E", "expected_nicolay_type": "T5",
        "ideal_docs_new": [288, 295, 367, 374, 413, 414, 419], "ideal_docs_original": [51, 52, 65, 66, 77, 78], "ideal_docs_count": 7,
        "critical_missing_evidence": "Last Public Address (Apr 11, 1865) — conditional suffrage statement NOT IN CORPUS",
        "watchlist": [
            "Complete retrieval collapse in Run 0 (P@5=0, R@5=0, rubric 3.25) — corpus stress-test / confabulation resistance case",
            "Retain as explicitly untestable until Last Public Address added",
            "Jonesboro chunks 517-518 retrieval (racial hierarchy statement)",
            "Historiographical nuance: does Nicolay handle limiting statements appropriately?",
        ],
    },
    {
        "id": "RC-3", "group": "new",
        "query": "How did Lincoln simultaneously affirm African Americans' natural rights under the Declaration of Independence while arguing against full political equality in the 1858 debates?",
        "category": "race_citizenship",
        "expected_hay_type": "E", "expected_nicolay_type": "T3",
        "ideal_docs_new": [41, 481, 550, 624, 679], "ideal_docs_original": None, "ideal_docs_count": 5,
        "critical_missing_evidence": None,
        "watchlist": [
            "Doc 481 (Freeport): 'no reason in the world why the negro is not entitled to all the natural rights enumerated in the Declaration' — key anchor",
            "Doc 41 (Peoria): 'the poor negro has some natural right to himself' — 1854 formulation",
            "Docs 481/550/679: liberty/equality tension — 'physical difference... forever forbid living on the footing of perfect equality'",
            "HD discriminator: Foner/Oakes liberty-equality distinction — does Nicolay navigate or collapse?",
        ],
    },
    {
        "id": "RC-4", "group": "new",
        "query": "How did Lincoln justify the emancipation of enslaved people and the use of Black soldiers as military policy, and what did he suggest this service implied for their future status?",
        "category": "race_citizenship",
        "expected_hay_type": "D", "expected_nicolay_type": "T3",
        "ideal_docs_new": [293, 295, 372, 374], "ideal_docs_original": None, "ideal_docs_count": 4,
        "critical_missing_evidence": None,
        "watchlist": [
            "Doc 295 (Conkling Letter): 'Some of them seem willing to fight for you' — military service pivot",
            "Doc 374 (3rd Annual Message): 'full one hundred thousand are now in the United States military service'",
            "'Future status' clause is calibration test: response should acknowledge implied rather than explicit citizenship claim",
        ],
    },
    {
        "id": "RC-5", "group": "new",
        "query": "How did Lincoln address the future political status of formerly enslaved people in his wartime Annual Messages and public letters, and what did he leave unresolved?",
        "category": "race_citizenship",
        "expected_hay_type": "E", "expected_nicolay_type": "T4",
        "ideal_docs_new": [297, 375, 376, 378, 410, 416], "ideal_docs_original": None, "ideal_docs_count": 6,
        "critical_missing_evidence": "Last Public Address (Apr 11, 1865) absent — explicit suffrage statement NOT IN CORPUS",
        "watchlist": [
            "'What did he leave unresolved' clause is calibration load-bearing — strongest EC discriminator in new set",
            "Well-scored response acknowledges corpus silence on explicit suffrage; fabricated response invents Lincoln's position",
            "Doc 297 (Conkling close): implied future status via military service",
            "Docs 375-376: reconstruction plan; doc 410: 13th Amendment advocacy; doc 416: reconstruction terms",
        ],
    },
    {
        "id": "RC-2", "group": "new",
        "query": "What specific conditions did Lincoln attach to his support for limited Black suffrage in his final public address?",
        "category": "race_citizenship",
        "expected_hay_type": "A", "expected_nicolay_type": "T1",
        "ideal_docs_new": [], "ideal_docs_original": None, "ideal_docs_count": 0,
        "critical_missing_evidence": "BLOCKED — Last Public Address (Apr 11, 1865) NOT IN CORPUS. Do not run until added.",
        "watchlist": [
            "BLOCKED QUESTION — corpus gap confirmed programmatically",
            "Do not run until Last Public Address is added to corpus",
            "Once unblocked: sharp T1/T2 question; one correct answer; minimal fabrication risk",
        ],
    },
]

QUERY_IDS = [q["id"] for q in BENCHMARK_QUERIES]
QUERY_BY_ID = {q["id"]: q for q in BENCHMARK_QUERIES}

# ─────────────────────────────────────────────────────────────────────────────
# CORPUS LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_corpus(path: str) -> dict:
    """Load lincoln_speech_corpus_reindex_keep.json → {int(text_id): chunk_dict}."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    corpus = {}
    for item in raw:
        # Support both 'text_id' formats: "Text #: 413" or integer 413
        tid = item.get("text_id", "")
        if isinstance(tid, str) and "Text #:" in tid:
            num = int(tid.split("Text #:")[1].strip())
        elif isinstance(tid, str) and tid.isdigit():
            num = int(tid)
        elif isinstance(tid, int):
            num = tid
        else:
            continue
        corpus[num] = item
    return corpus


def _get_or_load_corpus(corpus_file: str) -> dict:
    """
    Get the shared corpus dict from session_state, loading it on demand if absent.

    This is the single authoritative corpus-access function for all display code.
    It mirrors how the main app works: load eagerly, keep in memory, never gate
    display on a prior pipeline run.

    Returns {int_id: chunk_dict} or {} if corpus_file is not found/loadable.
    """
    _key = f"_corpus_shared_{corpus_file}"
    cached = st.session_state.get(_key)
    if cached:
        return cached
    if not corpus_file or not Path(corpus_file).exists():
        return {}
    try:
        _raw = load_corpus_any(corpus_file)
        _d: dict = {}
        for _item in _raw:
            _tid = _item.get("text_id", "")
            _m = re.search(r"(\d+)", str(_tid))
            if _m:
                _d[int(_m.group(1))] = _item
        st.session_state[_key] = _d
        return _d
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# HAY API CALL
# ─────────────────────────────────────────────────────────────────────────────

def call_hay(query: str, client: openai.OpenAI, system_prompt: str) -> dict:
    """
    Call Hay v3 fine-tuned model. Returns all five fields with safe .get() defaults.
    Pipeline dependency note: initial_answer seeds HyDE semantic search independently
    of keyword results — semantic search runs regardless of keyword success.
    """
    try:
        response = client.chat.completions.create(
            model=HAY_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            max_tokens=1500
        )
        raw_text = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        raw_text = re.sub(r'^```(?:json)?\s*', '', raw_text)
        raw_text = re.sub(r'\s*```$', '', raw_text)

        parsed = json.loads(raw_text)
        return {
            "initial_answer":    parsed.get("initial_answer", ""),
            "weighted_keywords": parsed.get("weighted_keywords", {}),
            "year_keywords":     parsed.get("year_keywords", []),
            "text_keywords":     parsed.get("text_keywords", []),
            "query_assessment":  parsed.get("query_assessment", ""),
            "_raw": raw_text,
            "_error": None
        }
    except json.JSONDecodeError as e:
        return {
            "initial_answer": query, "weighted_keywords": {},
            "year_keywords": [], "text_keywords": [], "query_assessment": "",
            "_raw": raw_text if 'raw_text' in dir() else "", "_error": f"JSON parse error: {e}"
        }
    except Exception as e:
        return {
            "initial_answer": query, "weighted_keywords": {},
            "year_keywords": [], "text_keywords": [], "query_assessment": "",
            "_raw": "", "_error": str(e)
        }


# ─────────────────────────────────────────────────────────────────────────────
# METRICS CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def extract_hay_type(query_assessment: str) -> Optional[str]:
    """Extract Type A/B/C/D/E from Hay's query_assessment prose.

    Hay v3 writes full labels ('Inferential Retrieval') rather than letter
    designations ('Type B'), so we match both formats.
    """
    # First try the letter format: "Type A", "Type B" etc.
    m = re.search(r'Type\s+([ABCDE])\b', query_assessment, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: match Hay v3's prose label format
    label_map = {
        "direct retrieval": "A",
        "inferential retrieval": "B",
        "absence recognition": "C",
        "multi-passage synthesis": "D",
        "contrastive": "E",
        "historiographical": "E",
    }
    qa_lower = query_assessment.lower()
    for label, letter in label_map.items():
        if label in qa_lower:
            return letter
    return None


def extract_nicolay_type(synthesis_assessment: str) -> Optional[str]:
    """Extract T1–T5 from Nicolay's synthesis_assessment string."""
    m = re.search(r'Type\s*([1-5])', synthesis_assessment, re.IGNORECASE)
    return f"T{m.group(1)}" if m else None


def compute_retrieval_metrics(reranked: list[dict], ideal_docs: list[int],
                               ideal_docs_original: list[int] = None) -> dict:
    """
    Compute precision@5, recall@5, ceiling-adjusted precision.
    Auto-detects which ID set matches the corpus in use:
    - If retrieved IDs overlap with ideal_docs_original → use original IDs
    - Otherwise use ideal_docs (886-chunk new IDs)
    """
    if not reranked:
        return {
            "precision_at_5": 0.0, "recall_at_5": 0.0,
            "ceiling_adjusted_precision": 0.0,
            "ideal_docs_hit": [], "ideal_docs_missed": list(ideal_docs),
            "ideal_docs_set_used": "new"
        }
    # Auto-detect: check if any retrieved ID appears in original IDs
    retrieved_ids_set = {r["text_id_num"] for r in reranked if r.get("text_id_num")}
    if ideal_docs_original and retrieved_ids_set & set(ideal_docs_original):
        ideal_docs = ideal_docs_original
        id_set_label = "original"
    else:
        id_set_label = "new"

    retrieved_ids = [r["text_id_num"] for r in reranked]
    ideal_set = set(ideal_docs)
    retrieved_set = set(retrieved_ids)

    hits = list(ideal_set & retrieved_set)
    misses = list(ideal_set - retrieved_set)

    k = len(reranked)
    precision = len(hits) / k if k > 0 else 0.0
    recall = len(hits) / len(ideal_set) if ideal_set else 0.0

    # Ceiling-adjusted: max possible precision given ideal set size vs k
    max_possible = min(len(ideal_set), k) / k if k > 0 else 0.0
    ceiling_adj = (precision / max_possible) if max_possible > 0 else 0.0

    return {
        "precision_at_5": round(precision, 4),
        "recall_at_5": round(recall, 4),
        "ceiling_adjusted_precision": round(ceiling_adj, 4),
        "ideal_docs_hit": sorted(hits),
        "ideal_docs_missed": sorted(misses),
        "ideal_docs_set_used": id_set_label
    }


def check_hay_spurious_fields(hay_output: dict) -> list:
    """Detect fields outside the 5-field Hay schema."""
    known = {"initial_answer", "weighted_keywords", "year_keywords", "text_keywords", "query_assessment", "_raw", "_error"}
    return [k for k in hay_output if k not in known]


def check_nicolay_schema(nicolay_output: dict) -> dict:
    """Check all 6 Nicolay schema sections are present."""
    required = ["User Query Analysis", "Initial Answer Review", "Match Analysis",
                "Meta Analysis", "FinalAnswer", "Model Feedback"]
    present = {f: f in nicolay_output for f in required}
    return {"complete": all(present.values()), "fields": present}


def get_final_answer_text(nicolay_output: dict) -> str:
    """Extract FinalAnswer text regardless of whether it's a string or nested dict."""
    fa = nicolay_output.get("FinalAnswer", "")
    if isinstance(fa, dict):
        return fa.get("Text", fa.get("text", str(fa)))
    return str(fa) if fa else ""


def get_synthesis_assessment(nicolay_output: dict) -> str:
    # In Nicolay v3, synthesis_assessment is under "User Query Analysis"
    # (confirmed from log_nicolay_model_output in main app).
    # Also check "Meta Analysis" as fallback for older schema variants.
    for section in ["User Query Analysis", "Meta Analysis"]:
        val = nicolay_output.get(section, {})
        if isinstance(val, dict):
            s = val.get("synthesis_assessment", "")
            if s:
                return s
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# QUOTE VERIFICATION  (v7.7 — unified stack ported from Streamlit app v1.9)
# ─────────────────────────────────────────────────────────────────────────────
#
# PATCH v7.6→v7.7 — Port of the working verification pipeline
# ────────────────────────────────────────────────────────────
# Replaces the v7.6 pipeline with the stage ordering and sliding-window
# logic proven in the Streamlit app (v1.9).  Key changes:
#
#   1. ALL cited-chunk methods now run before any corpus-wide displacement
#      scan.  This prevents false displacements where a quote IS in the
#      cited chunk but needs loose/sliding/token-coverage to match, yet
#      some other chunk happens to match strict segments first.
#
#   2. Sliding-window search (_sliding_window_verify) ported from v1.9.
#      Catches mid-sentence quoting and light sentence condensation that
#      the old edit-distance anchor stage handled imperfectly.
#
#   3. Fuzzy-punctuation / edit-distance anchor stage (old Stage 3.5)
#      removed — subsumed by sliding window + loose segments.
#
#   4. Corpus-wide displacement now runs strict segments → loose segments
#      → sliding window → token-coverage, mirroring the cited-chunk order.
#
#   5. Return dict schema and outcome strings unchanged — verify_all_quotes
#      and all downstream consumers (result aggregation, debug table) are
#      compatible without modification.
#
#   6. Normalization functions renamed to match v1.9 internal names but
#      old names retained as aliases for any stray call sites.
# ─────────────────────────────────────────────────────────────────────────────


def normalize_for_quote_matching(text: str) -> str:
    """
    Strict normalization for quote matching (equivalent to _norm_strict in v1.9).
    NFKD unicode → quote-glyph normalization → dash collapse →
    editorial bracket removal → lowercase → whitespace collapse.
    Does NOT strip punctuation (preserves verbatim-match precision).
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    # LaTeX-style double-backtick/double-apostrophe quoting (19th-century typesetting
    # convention present in Lincoln corpus) → plain ASCII double quotes.
    # Must run BEFORE single-backtick replacement so `` and '' are caught as pairs.
    text = text.replace("``", '"').replace("''", '"')
    # Quote glyphs → plain ASCII equivalents
    text = text.replace("`", "'")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Ellipsis variants → literal "..."
    text = text.replace("\u2026", "...")
    # Literal \n escape sequences and real newlines → space
    text = text.replace("\\n", " ").replace("\n", " ").replace("\r", " ")
    # Multi-dash runs and single dashes → space
    text = re.sub(r'[-\u2013\u2014]{2,}', ' ', text)
    text = re.sub(r'[-\u2013\u2014]', ' ', text)
    # Editorial brackets removed
    text = re.sub(r'\[.*?\]', '', text)
    # Collapse whitespace, lowercase
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    # Strip quote-delimiter characters at word boundaries so that typographic
    # variation between corpus and model output (e.g. corpus backtick→" vs
    # Nicolay curly-single→') never downgrades a verbatim quote to approximate.
    # Word-internal apostrophes (possessives, contractions) are preserved.
    text = re.sub(r'(?<!\w)["\']|["\'](?!\w)', '', text)
    return text


# Stopwords for token-coverage heuristic
_STOPWORDS = {
    "the","a","an","and","or","but","if","then","than","to","of","in","on",
    "for","with","by","at","as","is","are","was","were","be","been","being",
    "it","its","this","that","these","those","we","you","i","he","she","they",
    "them","his","her","their","our","us","my","your","not","no","so","do",
    "does","did","from","into","over","under","up","down","out","about",
    "because","which","who","whom","what","when","where","why","how",
}


def normalize_for_quote_matching_loose(text: str) -> str:
    """
    Loose normalization (equivalent to _norm_loose in v1.9).
    Everything normalize_for_quote_matching does, plus full punctuation removal.
    Used as second-pass to prevent false failures from backticks, curly quotes,
    and minor punctuation variants in corpus text.
    """
    text = normalize_for_quote_matching(text)
    if not text:
        return ""
    # Strip all non-alphanumeric characters (keep spaces)
    text = re.sub(r"[^0-9a-z\s]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _quote_segments_for_matching(passage: str) -> list[str]:
    """
    Split a passage on ellipsis markers into ordered normalized segments (strict).
    Each non-trivial segment must appear in order in the target chunk.

    Handles trailing "..." that models append as truncation markers.
    Strips surrounding quote characters so curly-quote-wrapped passages
    don't include quote chars in the segment string.
    """
    if not passage:
        return []
    p = unicodedata.normalize("NFKD", str(passage)).strip()
    p = p.strip('\u201c\u201d\u2018\u2019"\'').strip()
    parts = re.split(r'(?:\.\.\.|…)', p)
    segs = [normalize_for_quote_matching(part) for part in parts
            if len(normalize_for_quote_matching(part)) >= 10]
    if not segs:
        p2 = re.sub(r'(?:\.\.\.|…)+', ' ', p)
        s2 = normalize_for_quote_matching(p2)
        if s2:
            segs = [s2]
    return segs


def _quote_segments_for_matching_loose(passage: str) -> list[str]:
    """
    Split a passage on ellipsis markers into ordered normalized segments (loose).
    Same logic as strict version but uses loose normalization and a higher
    minimum segment length (18 chars) to compensate for punctuation removal.
    """
    if not passage:
        return []
    p = unicodedata.normalize("NFKD", str(passage)).strip()
    p = p.strip('\u201c\u201d\u2018\u2019"\'').strip()
    parts = re.split(r'(?:\.\.\.|…)', p)
    segs = [normalize_for_quote_matching_loose(part) for part in parts
            if len(normalize_for_quote_matching_loose(part)) >= 18]
    if not segs:
        p2 = re.sub(r'(?:\.\.\.|…)+', ' ', p)
        s2 = normalize_for_quote_matching_loose(p2)
        if s2:
            segs = [s2]
    return segs


def _contains_segments_in_order(haystack_norm: str, segs_norm: list[str]) -> bool:
    """Return True iff all segments occur in haystack in order."""
    if not haystack_norm or not segs_norm:
        return False
    pos = 0
    for seg in segs_norm:
        idx = haystack_norm.find(seg, pos)
        if idx == -1:
            return False
        pos = idx + len(seg)
    return True


def _content_tokens(norm_text: str) -> list[str]:
    """Tokenize and drop common stopwords; used for approximate matching heuristics."""
    if not norm_text:
        return []
    return [t for t in norm_text.split() if len(t) >= 3 and t not in _STOPWORDS]


def _token_coverage(quote_norm_loose: str, chunk_norm_loose: str) -> float:
    """
    Fraction of content tokens in the quote that appear in the chunk.
    Content tokens = non-stopword tokens of length ≥ 3.
    Returns 0.0 if the quote has fewer than 6 content tokens.
    """
    q = set(_content_tokens(quote_norm_loose))
    if len(q) < 6:
        return 0.0
    c = set(_content_tokens(chunk_norm_loose))
    if not c:
        return 0.0
    return len(q & c) / max(1, len(q))


def _sliding_window_verify(norm_quote: str, norm_chunk: str,
                            min_window: int = 30, step: int = 10):
    """
    Sliding-window substring search (ported from Streamlit app v1.9).

    Catches two failure modes that segment matching misses:
      1. Mid-sentence quoting — Nicolay starts a quote part-way through a
         sentence, so the opening anchor is absent even though most text
         is verbatim.
      2. Light sentence condensation — Nicolay drops a clause while
         preserving the surrounding text verbatim.

    Tries progressively shorter windows (60 → min_window chars) and returns
    (True, matched_segment) on the first hit, (False, '') otherwise.
    A min_window of 30 chars is long enough to be highly distinctive in
    19th-century prose while still catching condensed openings.
    Operates on strict-normalized text.
    """
    q_len = len(norm_quote)
    for window in range(min(60, q_len), min_window - 1, -step):
        for start in range(0, q_len - window + 1, step):
            segment = norm_quote[start:start + window]
            if segment in norm_chunk:
                return True, segment
    return False, ''


def verify_quote(cited_passage: str, cited_chunk: dict, corpus: dict) -> dict:
    """
    Seven-stage quote verification pipeline (v7.7 — unified with Streamlit v1.9).

    Outcomes:
      verified               — match in cited chunk (strict, loose, sliding, or token ≥0.95)
      approximate_quote      — high token overlap (0.86–0.95) in cited chunk; genuine
                                content but Nicolay rendered a compression in quote marks
      displacement           — match in a different corpus chunk (strict, loose, sliding,
                                or token ≥0.95)
      approximate_displacement — token coverage 0.90–0.95 in a different chunk
      fabrication            — no match found anywhere in corpus

    Stage ordering (all cited-chunk methods exhaust before corpus scan):
      1. Strict segments-in-order (cited chunk)
      2. Loose segments-in-order (cited chunk)
      3. Sliding window (cited chunk) — catches mid-sentence starts, condensation
      4. Token coverage ≥ 0.95 in cited chunk → verified
      5. Token coverage ≥ 0.86 in cited chunk → approximate_quote
      6. Corpus-wide displacement: strict → loose → sliding → token coverage
      7. Fabrication fallback
    """
    cited_chunk_present = bool(cited_chunk)
    cited_text = cited_chunk.get("full_text", cited_chunk.get("text", "")) if cited_chunk else ""
    cited_source = str(cited_chunk.get("source", "")) if cited_chunk else ""
    cited_text_id = str(cited_chunk.get("text_id", "")) if cited_chunk else ""

    if not cited_passage or len(str(cited_passage).strip()) < 5:
        return {
            "outcome": "too_short",
            "in_cited_chunk": None,
            "in_any_chunk": None,
            "fabricated": None,
            "note": "Passage too short to verify",
            "cited_chunk_present": cited_chunk_present,
            "cited_chunk_source": cited_source,
            "cited_chunk_text_id": cited_text_id,
            "cited_chunk_text_len": len(cited_text) if cited_text else 0,
        }

    # ── Precompute normalizations ──────────────────────────────────────────
    segs_strict    = _quote_segments_for_matching(cited_passage)
    segs_loose     = _quote_segments_for_matching_loose(cited_passage)
    norm_q_strict  = normalize_for_quote_matching(cited_passage)
    norm_q_loose   = normalize_for_quote_matching_loose(cited_passage)
    norm_ch_strict = normalize_for_quote_matching(cited_text) if cited_text else ""
    norm_ch_loose  = normalize_for_quote_matching_loose(cited_text) if cited_text else ""

    # Helper: build debug dict for cited-chunk verified outcomes
    def _cited_debug(method, note="", extra=None):
        d = {
            "outcome": "verified",
            "in_cited_chunk": True,
            "in_any_chunk": True,
            "fabricated": False,
            "match_source_id": cited_text_id or "unknown",
            "match_method": method,
            "cited_chunk_present": cited_chunk_present,
            "cited_chunk_source": cited_source,
            "cited_chunk_text_id": cited_text_id,
            "cited_chunk_text_len": len(cited_text) if cited_text else 0,
            "cited_chunk_preview": _safe_preview(cited_text, 140) if cited_text else "",
        }
        if note:
            d["note"] = note
        if extra:
            d.update(extra)
        return d

    # ── Stage 1 — Strict segments-in-order (cited chunk) ──────────────────
    if norm_ch_strict and segs_strict and _contains_segments_in_order(norm_ch_strict, segs_strict):
        return _cited_debug("strict_segments", extra={
            "norm_segments": segs_strict[:3],
            "num_segments": len(segs_strict),
        })

    # ── Stage 2 — Loose segments-in-order (cited chunk) ───────────────────
    if norm_ch_loose and segs_loose and _contains_segments_in_order(norm_ch_loose, segs_loose):
        return _cited_debug("loose_segments",
                            note="VERIFIED — punctuation-insensitive match in cited chunk",
                            extra={
                                "loose_segments": segs_loose[:3],
                                "num_loose_segments": len(segs_loose),
                            })

    # ── Stage 3 — Sliding window (cited chunk) ────────────────────────────
    if norm_ch_strict:
        sw_found, sw_seg = _sliding_window_verify(norm_q_strict, norm_ch_strict)
        if sw_found:
            return _cited_debug("sliding_window",
                                note="VERIFIED — sliding-window substring match in cited chunk",
                                extra={"sliding_match_segment": sw_seg[:80]})

    # ── Stage 4 — Token coverage ≥ 0.95 (cited chunk) → verified ─────────
    if norm_ch_loose:
        quote_loose_joined = " ".join(segs_loose) if segs_loose else norm_q_loose
        cov = _token_coverage(quote_loose_joined, norm_ch_loose)
        if cov >= 0.95:
            return _cited_debug("token_coverage",
                                note="VERIFIED — near-exact token coverage in cited chunk",
                                extra={"approx_score": float(cov)})

        # ── Stage 5 — Token coverage ≥ 0.86 (cited chunk) → approximate ──
        if cov >= 0.86:
            return {
                "outcome": "approximate_quote",
                "in_cited_chunk": True,
                "in_any_chunk": True,
                "fabricated": False,
                "match_source_id": cited_text_id or "unknown",
                "match_method": "token_coverage",
                "approx_score": float(cov),
                "note": "APPROXIMATE — high token coverage in cited chunk",
                "cited_chunk_present": cited_chunk_present,
                "cited_chunk_source": cited_source,
                "cited_chunk_text_id": cited_text_id,
                "cited_chunk_text_len": len(cited_text) if cited_text else 0,
                "cited_chunk_preview": _safe_preview(cited_text, 140) if cited_text else "",
            }

    # ── Stage 6 — Corpus-wide displacement search ─────────────────────────
    # Mirrors the cited-chunk order: strict → loose → sliding → token-coverage.
    # Only runs if we have a corpus to scan.
    if corpus:
        # Track best token-coverage hit across entire corpus for final fallback
        best_tc = (0.0, None, None, None)  # (coverage, cid, text_id, source)

        for cid, chunk in corpus.items():
            chunk_text = chunk.get("full_text", chunk.get("text", "")) or ""
            if not chunk_text:
                continue
            c_strict = normalize_for_quote_matching(chunk_text)
            c_loose  = normalize_for_quote_matching_loose(chunk_text)
            c_tid    = str(chunk.get("text_id", ""))
            c_src    = str(chunk.get("source", ""))

            # 6a — Strict segments
            if segs_strict and _contains_segments_in_order(c_strict, segs_strict):
                return {
                    "outcome": "displacement",
                    "in_cited_chunk": False,
                    "in_any_chunk": True,
                    "fabricated": False,
                    "match_source_id": c_tid,
                    "match_chunk_num": cid,
                    "match_chunk_source": c_src,
                    "match_method": "strict_segments",
                    "note": "DISPLACEMENT — strict match found in different chunk",
                    "cited_chunk_present": cited_chunk_present,
                    "cited_chunk_source": cited_source,
                    "cited_chunk_text_id": cited_text_id,
                    "cited_chunk_text_len": len(cited_text) if cited_text else 0,
                    "cited_chunk_preview": _safe_preview(cited_text, 140) if cited_text else "",
                    "norm_segments": segs_strict[:3],
                    "num_segments": len(segs_strict),
                }

            # 6b — Loose segments
            if segs_loose and _contains_segments_in_order(c_loose, segs_loose):
                return {
                    "outcome": "displacement",
                    "in_cited_chunk": False,
                    "in_any_chunk": True,
                    "fabricated": False,
                    "match_source_id": c_tid,
                    "match_chunk_num": cid,
                    "match_chunk_source": c_src,
                    "match_method": "loose_segments",
                    "note": "DISPLACEMENT — punctuation-insensitive match found in different chunk",
                    "cited_chunk_present": cited_chunk_present,
                    "cited_chunk_source": cited_source,
                    "cited_chunk_text_id": cited_text_id,
                    "cited_chunk_text_len": len(cited_text) if cited_text else 0,
                    "cited_chunk_preview": _safe_preview(cited_text, 140) if cited_text else "",
                    "loose_segments": segs_loose[:3],
                    "num_loose_segments": len(segs_loose),
                }

            # 6c — Sliding window
            sw_found, sw_seg = _sliding_window_verify(norm_q_strict, c_strict)
            if sw_found:
                return {
                    "outcome": "displacement",
                    "in_cited_chunk": False,
                    "in_any_chunk": True,
                    "fabricated": False,
                    "match_source_id": c_tid,
                    "match_chunk_num": cid,
                    "match_chunk_source": c_src,
                    "match_method": "sliding_window",
                    "note": "DISPLACEMENT — sliding-window match found in different chunk",
                    "cited_chunk_present": cited_chunk_present,
                    "cited_chunk_source": cited_source,
                    "cited_chunk_text_id": cited_text_id,
                    "cited_chunk_text_len": len(cited_text) if cited_text else 0,
                    "cited_chunk_preview": _safe_preview(cited_text, 140) if cited_text else "",
                    "sliding_match_segment": sw_seg[:80],
                }

            # 6d — Track best token-coverage for post-loop evaluation
            tc = _token_coverage(
                " ".join(segs_loose) if segs_loose else norm_q_loose,
                c_loose
            )
            if tc > best_tc[0]:
                best_tc = (tc, cid, c_tid, c_src)

        # Evaluate best corpus-wide token coverage
        if best_tc[1] is not None:
            if best_tc[0] >= 0.95:
                return {
                    "outcome": "displacement",
                    "in_cited_chunk": False,
                    "in_any_chunk": True,
                    "fabricated": False,
                    "match_source_id": best_tc[2],
                    "match_chunk_num": best_tc[1],
                    "match_chunk_source": best_tc[3],
                    "match_method": "token_coverage",
                    "approx_score": float(best_tc[0]),
                    "note": "DISPLACEMENT — near-exact token coverage in different chunk",
                    "cited_chunk_present": cited_chunk_present,
                    "cited_chunk_source": cited_source,
                    "cited_chunk_text_id": cited_text_id,
                    "cited_chunk_text_len": len(cited_text) if cited_text else 0,
                    "cited_chunk_preview": _safe_preview(cited_text, 140) if cited_text else "",
                }
            if best_tc[0] >= 0.90:
                return {
                    "outcome": "approximate_displacement",
                    "in_cited_chunk": False,
                    "in_any_chunk": True,
                    "fabricated": False,
                    "match_source_id": best_tc[2],
                    "match_chunk_num": best_tc[1],
                    "match_chunk_source": best_tc[3],
                    "match_method": "token_coverage",
                    "approx_score": float(best_tc[0]),
                    "note": "APPROXIMATE DISPLACEMENT — high token coverage in different chunk",
                    "cited_chunk_present": cited_chunk_present,
                    "cited_chunk_source": cited_source,
                    "cited_chunk_text_id": cited_text_id,
                    "cited_chunk_text_len": len(cited_text) if cited_text else 0,
                    "cited_chunk_preview": _safe_preview(cited_text, 140) if cited_text else "",
                }

    # ── Stage 7 — Fabrication fallback ────────────────────────────────────
    return {
        "outcome": "fabrication",
        "in_cited_chunk": False,
        "in_any_chunk": False,
        "fabricated": True,
        "note": "FABRICATION — passage not found in corpus",
        "cited_chunk_present": cited_chunk_present,
        "cited_chunk_source": cited_source,
        "cited_chunk_text_id": cited_text_id,
        "cited_chunk_text_len": len(cited_text) if cited_text else 0,
        "cited_chunk_preview": _safe_preview(cited_text, 140) if cited_text else "",
        "norm_segments": segs_strict[:3],
        "num_segments": len(segs_strict),
        "loose_segments": segs_loose[:3],
        "num_loose_segments": len(segs_loose),
    }
def _source_matches_refs(corpus_source: str, claimed_source: str) -> bool:
    """
    Compare a corpus chunk's actual source against Nicolay's claimed source label.
    Returns True if they refer to the same document (token-Jaccard ≥ 0.35).

    Handles the common cases:
      "At Peoria, Illinois. October 16, 1854."  vs  "Peoria Address (1854)"   → 0.50 ✓
      "First Annual Message. December 3, 1861"  vs  same with Source: prefix  → 1.00 ✓
    And correctly rejects:
      "Lincoln-Douglas Debates, Fourth Debate, Charleston, 1858"
        vs "Speech to the Young Men's Lyceum, Springfield, 1838"              → 0.11 ✗
    """
    if not corpus_source or not claimed_source:
        return True  # cannot check → don't penalise

    def _norm(s):
        import re as _re
        s = _re.sub(r'^source:\s*', '', str(s).strip(), flags=_re.IGNORECASE)
        s = _re.sub(r'[.,;:\-()]', ' ', s)
        s = _re.sub(r'\s+', ' ', s).strip().lower()
        return s

    ta = set(t for t in _norm(corpus_source).split() if len(t) > 2)
    tb = set(t for t in _norm(claimed_source).split() if len(t) > 2)
    if not ta or not tb:
        return True
    return len(ta & tb) / max(len(ta), len(tb)) >= 0.35


def verify_all_quotes(nicolay_output: dict, reranked: list[dict], corpus: dict) -> list[dict]:
    """Run quote verification for all Match Analysis entries (with debug metadata)."""
    match_analysis = nicolay_output.get("Match Analysis", {})
    if not isinstance(match_analysis, dict):
        return []

    # Helpful for diagnosing: is Nicolay citing an ID that wasn't even retrieved?
    reranked_by_num = {r.get("text_id_num"): r for r in (reranked or []) if r.get("text_id_num") is not None}
    results: list[dict] = []

    for match_key, match_val in match_analysis.items():
        if not isinstance(match_val, dict):
            continue

        key_passage = match_val.get("Key Quote", match_val.get("Key Passage", ""))  # v3 uses "Key Quote"
        text_id_str = match_val.get("Text ID", "")

        cited_num = _extract_int_from_text_id(text_id_str)
        cited_chunk = corpus.get(cited_num) if (cited_num is not None and corpus) else None
        cited_present = bool(cited_chunk)

        verification = verify_quote(key_passage, cited_chunk, corpus)

        # Was this ID actually part of the reranked top-k list?
        reranked_rec = reranked_by_num.get(cited_num) if cited_num is not None else None

        # Source mislabeling check: if quote text verified but claimed source
        # doesn't match corpus source → override outcome to source_mislabeled.
        # This catches parametric laundering: real text, fabricated attribution.
        claimed_source = str(match_val.get("Source", "") or "").strip()
        actual_source  = str(cited_chunk.get("source", "")) if cited_chunk else ""
        if (verification.get("outcome") == "verified"
                and claimed_source
                and not _source_matches_refs(actual_source, claimed_source)):
            verification = {**verification, "outcome": "source_mislabeled",
                            "note": "MISLABELED — quote text verified but source attribution "
                                    f"does not match corpus source. "
                                    f"Claimed: '{claimed_source[:80]}' | "
                                    f"Actual: '{actual_source[:80]}'"}

        results.append({
            "match": match_key,
            "text_id": text_id_str,
            "cited_num": cited_num,
            "cited_chunk_present": cited_present,
            "cited_chunk_source": actual_source,
            "cited_chunk_text_id": str(cited_chunk.get("text_id", "")) if cited_chunk else "",
            "cited_passage": key_passage,
            "cited_in_reranked_topk": bool(reranked_rec),
            "cited_reranked_rank": reranked_rec.get("rank") if isinstance(reranked_rec, dict) else None,
            **verification,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# BLEU / ROUGE SCORING
# ─────────────────────────────────────────────────────────────────────────────

def compute_bleu_rouge(
    generated_response: str,
    reranked: list[dict],
    corpus: dict,
    ideal_docs: list[int]
) -> dict:
    """
    Compute BLEU and ROUGE scores for the Nicolay FinalAnswer.

    Two reference sets:
    1. Retrieved docs (top-5 reranked full texts) — measures how faithfully
       the response uses what was actually retrieved.
    2. Ideal docs (ground-truth chunks) — measures alignment with what
       *should* have been used.

    BLEU uses sentence_bleu with smoothing (method1) to avoid zero scores
    on short n-gram overlaps. ROUGE uses rouge_scorer with ROUGE-1, ROUGE-2, ROUGE-L.

    Note for DHQ article framing: BLEU/ROUGE measure lexical overlap, not
    historiographical quality. High ROUGE on retrieved docs + low ROUGE on ideal
    docs is a diagnostic signal (faithfully synthesized wrong chunks). These scores
    complement, not replace, the qualitative rubric.
    """
    if not generated_response:
        return {"error": "No generated response to score"}

    # Tokenize
    hypothesis_tokens = nltk.word_tokenize(generated_response.lower())

    # Build reference texts
    retrieved_texts = [r.get("full_text", "") for r in reranked if r.get("full_text")]
    ideal_texts = [
        corpus[i].get("full_text", corpus[i].get("text", ""))
        for i in ideal_docs if i in corpus
    ]

    def score_against(reference_texts: list[str], label: str) -> dict:
        if not reference_texts:
            return {f"bleu_{label}": None, f"rouge1_{label}": None,
                    f"rouge2_{label}": None, f"rougeL_{label}": None}

        # BLEU: compare against each reference individually, take max
        # (captures best-matching chunk rather than requiring all to match)
        smoother = SmoothingFunction().method1
        ref_token_lists = [nltk.word_tokenize(t.lower()) for t in reference_texts if t]
        bleu_scores = []
        for ref_tokens in ref_token_lists:
            try:
                score = sentence_bleu([ref_tokens], hypothesis_tokens,
                                      smoothing_function=smoother)
                bleu_scores.append(score)
            except Exception:
                pass
        bleu_max = max(bleu_scores) if bleu_scores else 0.0
        bleu_avg = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

        # ROUGE: score against concatenated references for aggregate signal,
        # also report per-chunk max
        scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        for ref_text in reference_texts:
            try:
                s = scorer_obj.score(ref_text, generated_response)
                rouge1_scores.append(s['rouge1'].fmeasure)
                rouge2_scores.append(s['rouge2'].fmeasure)
                rougeL_scores.append(s['rougeL'].fmeasure)
            except Exception:
                pass

        return {
            f"bleu_max_{label}": round(bleu_max, 4),
            f"bleu_avg_{label}": round(bleu_avg, 4),
            f"rouge1_max_{label}": round(max(rouge1_scores), 4) if rouge1_scores else 0.0,
            f"rouge1_avg_{label}": round(sum(rouge1_scores)/len(rouge1_scores), 4) if rouge1_scores else 0.0,
            f"rouge2_max_{label}": round(max(rouge2_scores), 4) if rouge2_scores else 0.0,
            f"rouge2_avg_{label}": round(sum(rouge2_scores)/len(rouge2_scores), 4) if rouge2_scores else 0.0,
            f"rougeL_max_{label}": round(max(rougeL_scores), 4) if rougeL_scores else 0.0,
            f"rougeL_avg_{label}": round(sum(rougeL_scores)/len(rougeL_scores), 4) if rougeL_scores else 0.0,
        }

    retrieved_scores = score_against(retrieved_texts, "retrieved")
    ideal_scores = score_against(ideal_texts, "ideal")

    # Diagnostic ratio: ROUGE-1 avg on retrieved vs ideal
    # Helps identify "faithful but wrong source" failure mode
    r1_retr = retrieved_scores.get("rouge1_avg_retrieved", 0) or 0
    r1_ideal = ideal_scores.get("rouge1_avg_ideal", 0) or 0
    diagnostic_ratio = round(r1_retr / r1_ideal, 3) if r1_ideal > 0 else None

    return {
        **retrieved_scores,
        **ideal_scores,
        "rouge1_retrieved_vs_ideal_ratio": diagnostic_ratio,
        "_note": (
            "BLEU/ROUGE measure lexical overlap only. "
            "High ratio (retrieved >> ideal) may indicate faithful synthesis of wrong chunks. "
            "Use alongside qualitative rubric, not as standalone quality measure."
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS PERSISTENCE — Google Sheets primary, session_state working store
# ─────────────────────────────────────────────────────────────────────────────
#
# Streamlit Cloud has an ephemeral filesystem — local file writes do not
# survive reruns or sleep cycles. Google Sheets is the durable store.
#
# Architecture:
#   • st.session_state.results  — live working dict for the current session
#   • benchmark_logger          — appends one row per query to "benchmark_results"
#   • On startup: optionally reads prior rows back from Sheets to resume
#
# The "Results Directory" sidebar option is retained for local dev convenience
# but is not used on Cloud. The CSV export button serialises session_state
# directly so it works regardless of filesystem.

RESULTS_FILE = "nicolay_benchmark_results.json"   # local dev fallback only
CSV_FILE = "nicolay_benchmark_summary.csv"


def empty_results() -> dict:
    return {
        "run_metadata": {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "hay_model": HAY_MODEL,
            "nicolay_model": NICOLAY_MODEL,
            "cohere_model": COHERE_RERANK_MODEL,
            "corpus": "lincoln_speech_corpus_reindex_keep.json",
            "k": RERANK_K,
            "notes": ""
        },
        "queries": {}
    }


def load_results(results_file: str = RESULTS_FILE) -> dict:
    """Load from local file (local dev only). Returns empty results if not found."""
    if Path(results_file).exists():
        try:
            with open(results_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return empty_results()


def save_results(results: dict, results_file: str = RESULTS_FILE):
    """Write to local file (local dev only). Silently no-ops on Cloud."""
    try:
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
    except Exception:
        pass  # Expected on Streamlit Cloud — Sheets is the real store


def log_query_to_sheets(benchmark_logger, qresult: dict):
    """
    Append one benchmark query result row to the 'benchmark_results_data' Google Sheet.
    Uses DataLogger.record_api_outputs(record_dict) from data_logging.py.

    DataLogger.record_api_outputs() behavior:
    - Adds a 'Timestamp' key (datetime.now()) to the dict, overwriting any existing one
    - Converts the dict to a single-row DataFrame
    - Calls sheet.set_dataframe(df, (end_row, 1), copy_head=False, extend=True)

    IMPORTANT: pygsheets set_dataframe with copy_head=False writes values positionally,
    not by column name. The sheet MUST have headers already set in row 1 matching the
    column order in the record dict. If headers are absent or mismatched, data will be
    written to wrong columns. Set up headers manually or via init_benchmark_sheet_headers()
    below before the first run.

    Sheet columns (must be present in row 1 of benchmark_results_data sheet, in this order):
    Timestamp, QueryID, Query, Category, HayModel, NicolayModel,
    HayTypeExpected, HayTypeGot, HayTypeCorrect, HayKeywordCount,
    HaySpuriousFields, HayTrainingBleed, InitialAnswer, QueryAssessment,
    RetrievedDocIDs, RetrievalSearchTypes, RerankerScores,
    PrecisionAt5, RecallAt5, CeilingAdjustedPrecision,
    IdealDocsHit, IdealDocsMissed,
    NicolayTypeExpected, NicolayTypeGot, NicolayTypeCorrect,
    SchemaComplete, FinalAnswerWordCount, FinalAnswerText,
    QuotesVerified, QuotesDisplaced, QuotesFabricated,
    BleuMaxRetrieved, BleuAvgRetrieved,
    Rouge1MaxRetrieved, Rouge1AvgRetrieved,
    Rouge1MaxIdeal, Rouge1AvgIdeal,
    Rouge1RetrievedVsIdealRatio,
    RougeL_MaxRetrieved, RougeL_MaxIdeal,
    CriticalMissingEvidence,
    RubricFactualAccuracy, RubricCitationAccuracy,
    RubricHistoriographicalDepth, RubricEpistemicCalibration,
    RubricTotal, EvaluatorNotes
    """
    if benchmark_logger is None:
        return

    hay_out = qresult.get("hay_output", {})

    # Derive the active model IDs from the stamped model_config_tag in qresult.
    # This is the ground truth for what models actually ran — reading the global
    # HAY_MODEL / NICOLAY_MODEL here would log the *current* sidebar selection,
    # which may differ from what was active when this query executed (e.g. after
    # a pair switch between runs or on Cloud where globals reset on rerun).
    _tag = qresult.get("model_config_tag", "")
    def _extract_model_from_tag(tag, key):
        import re as _re_tag
        m = _re_tag.search(rf"{key}=([^|]+)", tag)
        return m.group(1) if m else ""
    _hay_model_logged     = _extract_model_from_tag(_tag, "hay")     or str(HAY_MODEL)
    _nicolay_model_logged = _extract_model_from_tag(_tag, "nicolay") or str(NICOLAY_MODEL)

    # Build record — Timestamp MUST be the first key so DataLogger's automatic
    # datetime.now() overwrites it in-place at column 0 (positional write).
    # If Timestamp is absent or last, it appends at the end, shifting every
    # data value one column to the left relative to the sheet headers.
    # All values must be JSON-serializable primitives (str, int, float, bool).
    record = {
        "Timestamp": "",        # DataLogger overwrites this with datetime.now()
        "QueryID": str(qresult.get("id", "")),
        "Query": str(qresult.get("query", "")),
        "Category": str(qresult.get("category", "")),
        # HV-5: Model configuration tag — stamped at pipeline start, source of truth
        "ModelConfigTag": str(_tag),
        "HayModel": _hay_model_logged,
        "NicolayModel": _nicolay_model_logged,
        # Pipeline reliability
        "PipelineAttempts": int(qresult.get("pipeline_attempts", 1) or 1),
        "PipelineRetryLog": str(qresult.get("pipeline_retry_log", "") or ""),
        # Hay layer
        "HayTypeExpected": str(qresult.get("expected_hay_type", "")),
        "HayTypeGot": str(qresult.get("hay_task_type_raw", "") or ""),
        "HayTypeCorrect": str(qresult.get("hay_task_type_correct", "")),
        "HayKeywordCount": int(qresult.get("hay_keyword_count", 0) or 0),
        "HaySpuriousFields": json.dumps(qresult.get("hay_spurious_fields", [])),
        "HayTrainingBleed": str(qresult.get("hay_training_bleed", False)),
        # HV-1: full initial_answer as top-level field (also in hay_output nested dict)
        "InitialAnswer": str(qresult.get("hay_initial_answer", "") or hay_out.get("initial_answer", "")),
        # HV-2: full query_assessment string
        "QueryAssessment": str(qresult.get("hay_query_assessment_raw", "") or hay_out.get("query_assessment", "")),
        # Retrieval layer
        "RetrievedDocIDs": json.dumps(qresult.get("retrieved_doc_ids", [])),
        "RetrievalSearchTypes": json.dumps(qresult.get("retrieval_search_types", [])),
        # HV-3: top-5 retrieval path summary
        "RetrievalPathTop5": str(qresult.get("retrieval_path_top5", "")),
        "RetrievalKeywordCountTop5": int(qresult.get("retrieval_keyword_count_top5", 0) or 0),
        "RetrievalSemanticCountTop5": int(qresult.get("retrieval_semantic_count_top5", 0) or 0),
        "RerankerScores": json.dumps([
            round(s, 6) if isinstance(s, float) else s
            for s in qresult.get("reranker_scores", [])
        ]),
        # HV-4: reranker score distribution (top-5)
        "RerankerScoreMaxTop5": float(qresult.get("reranker_score_max_top5") or 0),
        "RerankerScoreMinTop5": float(qresult.get("reranker_score_min_top5") or 0),
        "RerankerScoreMeanTop5": float(qresult.get("reranker_score_mean_top5") or 0),
        "PrecisionAt5": float(qresult.get("precision_at_5", 0) or 0),
        "RecallAt5": float(qresult.get("recall_at_5", 0) or 0),
        "CeilingAdjustedPrecision": float(qresult.get("ceiling_adjusted_precision", 0) or 0),
        "IdealDocsHit": json.dumps(qresult.get("ideal_docs_hit", [])),
        "IdealDocsMissed": json.dumps(qresult.get("ideal_docs_missed", [])),
        # Nicolay layer
        "NicolayTypeExpected": str(qresult.get("expected_nicolay_type", "")),
        "NicolayTypeGot": str(qresult.get("nicolay_synthesis_type_raw", "") or ""),
        "NicolayTypeCorrect": str(qresult.get("nicolay_synthesis_type_correct", "")),
        # CF-3: raw synthesis_assessment string before regex extraction
        "NicolaySynthesisAssessmentRaw": str(qresult.get("nicolay_synthesis_assessment_raw", "")),
        "SchemaComplete": str(qresult.get("nicolay_schema_complete", "")),
        "FinalAnswerWordCount": int(qresult.get("nicolay_final_answer_wordcount", 0) or 0),
        # HV-6: full FinalAnswer text (added v5.3)
        "FinalAnswerText": str(qresult.get("nicolay_final_answer_text", "") or ""),
        # Quote verification
        "QuotesVerified": int(qresult.get("quotes_verified_count", 0) or 0),
        "QuotesApprox": int(qresult.get("quotes_approx_count", 0) or 0),
        "QuotesDisplaced": int(qresult.get("quotes_displaced_count", 0) or 0),
        "QuotesApproxDisplaced": int(qresult.get("quotes_approx_displaced_count", 0) or 0),
        "QuotesFabricated": int(qresult.get("quotes_fabricated_count", 0) or 0),
        # BLEU/ROUGE
        "BleuMaxRetrieved": float(qresult.get("bleu_max_retrieved", 0) or 0),
        "BleuAvgRetrieved": float(qresult.get("bleu_avg_retrieved", 0) or 0),
        "Rouge1MaxRetrieved": float(qresult.get("rouge1_max_retrieved", 0) or 0),
        "Rouge1AvgRetrieved": float(qresult.get("rouge1_avg_retrieved", 0) or 0),
        "Rouge1MaxIdeal": float(qresult.get("rouge1_max_ideal", 0) or 0),
        "Rouge1AvgIdeal": float(qresult.get("rouge1_avg_ideal", 0) or 0),
        "Rouge1RetrievedVsIdealRatio": float(qresult.get("rouge1_retrieved_vs_ideal_ratio", 0) or 0),
        "RougeL_MaxRetrieved": float(qresult.get("rougeL_max_retrieved", 0) or 0),
        "RougeL_MaxIdeal": float(qresult.get("rougeL_max_ideal", 0) or 0),
        # Critical missing evidence
        "CriticalMissingEvidence": str(qresult.get("critical_missing_evidence", "") or ""),
        # Qualitative rubric (may be empty until manually scored)
        "RubricFactualAccuracy": str(qresult.get("rubric_factual_accuracy", "") if qresult.get("rubric_factual_accuracy") is not None else ""),
        "RubricCitationAccuracy": str(qresult.get("rubric_citation_accuracy", "") if qresult.get("rubric_citation_accuracy") is not None else ""),
        "RubricHistoriographicalDepth": str(qresult.get("rubric_historiographical_depth", "") if qresult.get("rubric_historiographical_depth") is not None else ""),
        "RubricEpistemicCalibration": str(qresult.get("rubric_epistemic_calibration", "") if qresult.get("rubric_epistemic_calibration") is not None else ""),
        "RubricTotal": str(qresult.get("rubric_total", "") if qresult.get("rubric_total") is not None else ""),
        "EvaluatorNotes": str(qresult.get("evaluator_notes", "")),
        # U12 confidence signals (v5.5)
        "ConfidenceRating":       str(qresult.get("confidence_rating", "") or ""),
        "ConfidenceRouge1":       float(qresult.get("confidence_rouge1") or 0),
        "ConfidenceRouge2":       float(qresult.get("confidence_rouge2") or 0),
        "ConfidenceCalibWarning": str(qresult.get("confidence_calib_warning", False)),
        "ConfidenceSpread":       float(qresult.get("confidence_spread") or 0),
        "ConfidenceNSources":     int(qresult.get("confidence_n_sources") or 0),
        "ConfidenceMaxScore":     float(qresult.get("confidence_max_score") or 0),
    }

    try:
        benchmark_logger.record_api_outputs(record)
        # Clear any previous error for this query
        st.session_state[f"_debug_sheets_err_{record.get('QueryID', '')}"] = "none recorded"
    except Exception as e:
        err_msg = f"Sheets logging failed for {record.get('QueryID', '?')}: {type(e).__name__}: {e}"
        st.warning(f"⚠️ {err_msg}")
        try:
            st.session_state[f"_debug_sheets_err_{record.get('QueryID', '')}"] = err_msg
        except Exception:
            pass


def init_benchmark_sheet_headers(gc_client):
    """
    Write the required column headers to row 1 of the 'benchmark_results_data' sheet.
    Call this once before the first benchmark run to ensure positional alignment
    with DataLogger.record_api_outputs() which uses set_dataframe(copy_head=False).

    Safe to call again — checks if headers are already present before writing.
    """
    if gc_client is None:
        return
    headers = [
        "Timestamp", "QueryID", "Query", "Category", "ModelConfigTag",
        "PipelineAttempts", "PipelineRetryLog",
        "HayModel", "NicolayModel",
        "HayTypeExpected", "HayTypeGot", "HayTypeCorrect", "HayKeywordCount",
        "HaySpuriousFields", "HayTrainingBleed", "InitialAnswer", "QueryAssessment",
        "RetrievedDocIDs", "RetrievalSearchTypes",
        "RetrievalPathTop5", "RetrievalKeywordCountTop5", "RetrievalSemanticCountTop5",
        "RerankerScores", "RerankerScoreMaxTop5", "RerankerScoreMinTop5", "RerankerScoreMeanTop5",
        "PrecisionAt5", "RecallAt5", "CeilingAdjustedPrecision",
        "IdealDocsHit", "IdealDocsMissed",
        "NicolayTypeExpected", "NicolayTypeGot", "NicolayTypeCorrect",
        "NicolaySynthesisAssessmentRaw",
        "SchemaComplete", "FinalAnswerWordCount", "FinalAnswerText",
        "QuotesVerified", "QuotesApprox", "QuotesDisplaced", "QuotesApproxDisplaced", "QuotesFabricated",
        "BleuMaxRetrieved", "BleuAvgRetrieved",
        "Rouge1MaxRetrieved", "Rouge1AvgRetrieved",
        "Rouge1MaxIdeal", "Rouge1AvgIdeal",
        "Rouge1RetrievedVsIdealRatio",
        "RougeL_MaxRetrieved", "RougeL_MaxIdeal",
        "CriticalMissingEvidence",
        "RubricFactualAccuracy", "RubricCitationAccuracy",
        "RubricHistoriographicalDepth", "RubricEpistemicCalibration",
        "RubricTotal", "EvaluatorNotes",
        # U12 confidence signals (v5.5)
        "ConfidenceRating", "ConfidenceRouge1", "ConfidenceRouge2",
        "ConfidenceCalibWarning", "ConfidenceSpread", "ConfidenceNSources",
        "ConfidenceMaxScore",
    ]
    try:
        sh = gc_client.open_by_key("1uQx9ERAHL0EKaI5QEpfywGxS40TpQ0Ao0HLfJe74auo").sheet1
        existing_row1 = sh.get_row(1, returnas="matrix")
        if existing_row1 and existing_row1[0] == "Timestamp":
            return  # Headers already present
        sh.update_row(1, headers)
        st.info("📋 Sheet headers initialized.")
    except Exception as e:
        st.warning(f"⚠️ Could not initialize sheet headers: {e}")


def update_rubric_in_sheets(benchmark_logger, gc_client, qid: str, rubric_data: dict):
    """
    Update rubric scores for an already-logged row in benchmark_results sheet.
    Finds the row by QueryID and updates the rubric columns in place.
    Falls back to appending a new row if the query isn't found.
    """
    if benchmark_logger is None or gc_client is None:
        return
    try:
        sh = gc_client.open_by_key("1uQx9ERAHL0EKaI5QEpfywGxS40TpQ0Ao0HLfJe74auo").sheet1
        all_rows = sh.get_all_records()
        headers = sh.row(1)  # Get header row

        # Find column indices for rubric fields
        rubric_cols = {
            "RubricFactualAccuracy": None, "RubricCitationAccuracy": None,
            "RubricHistoriographicalDepth": None, "RubricEpistemicCalibration": None,
            "RubricTotal": None, "EvaluatorNotes": None
        }
        for i, h in enumerate(headers, 1):
            if h in rubric_cols:
                rubric_cols[h] = i

        # Find the row with matching QueryID
        for row_idx, row in enumerate(all_rows, 2):  # data starts at row 2
            if row.get("QueryID") == qid:
                for col_name, col_idx in rubric_cols.items():
                    if col_idx and col_name in rubric_data:
                        sh.update_value((row_idx, col_idx), str(rubric_data[col_name]))
                return
        # Not found — shouldn't happen but log it
        st.warning(f"Could not find {qid} in benchmark_results_data sheet to update rubric.")
    except Exception as e:
        st.warning(f"⚠️ Rubric sheet update failed: {e}")


def export_csv(results: dict, csv_file: str = CSV_FILE) -> bytes:
    """Serialize results to CSV bytes for Streamlit download. No file write needed."""
    rows = []
    for qid, qdata in results.get("queries", {}).items():
        rows.append({
            "id": qid,
            "category": qdata.get("category", ""),
            "model_config_tag": qdata.get("model_config_tag", ""),
            "pipeline_attempts": qdata.get("pipeline_attempts", ""),
            "pipeline_retry_log": qdata.get("pipeline_retry_log", ""),
            "expected_hay_type": qdata.get("expected_hay_type", ""),
            "hay_type_raw": qdata.get("hay_task_type_raw", ""),
            "hay_type_correct": qdata.get("hay_task_type_correct", ""),
            "hay_keyword_count": qdata.get("hay_keyword_count", ""),
            "hay_spurious_fields": str(qdata.get("hay_spurious_fields", [])),
            "hay_initial_answer": qdata.get("hay_initial_answer", ""),
            "hay_query_assessment_raw": qdata.get("hay_query_assessment_raw", ""),
            "precision_at_5": qdata.get("precision_at_5", ""),
            "recall_at_5": qdata.get("recall_at_5", ""),
            "ceiling_adjusted_precision": qdata.get("ceiling_adjusted_precision", ""),
            "retrieval_path_top5": qdata.get("retrieval_path_top5", ""),
            "retrieval_keyword_count_top5": qdata.get("retrieval_keyword_count_top5", ""),
            "retrieval_semantic_count_top5": qdata.get("retrieval_semantic_count_top5", ""),
            "reranker_score_max_top5": qdata.get("reranker_score_max_top5", ""),
            "reranker_score_min_top5": qdata.get("reranker_score_min_top5", ""),
            "reranker_score_mean_top5": qdata.get("reranker_score_mean_top5", ""),
            "expected_nicolay_type": qdata.get("expected_nicolay_type", ""),
            "nicolay_type_raw": qdata.get("nicolay_synthesis_type_raw", ""),
            "nicolay_type_correct": qdata.get("nicolay_synthesis_type_correct", ""),
            "nicolay_synthesis_assessment_raw": qdata.get("nicolay_synthesis_assessment_raw", ""),
            "schema_complete": qdata.get("nicolay_schema_complete", ""),
            "final_answer_wordcount": qdata.get("nicolay_final_answer_wordcount", ""),
            "quotes_verified": qdata.get("quotes_verified_count", ""),
            "quotes_approx": qdata.get("quotes_approx_count", ""),
            "quotes_displaced": qdata.get("quotes_displaced_count", ""),
            "quotes_approx_displaced": qdata.get("quotes_approx_displaced_count", ""),
            "quotes_fabricated": qdata.get("quotes_fabricated_count", ""),
            "bleu_max_retrieved": qdata.get("bleu_max_retrieved", ""),
            "rouge1_max_retrieved": qdata.get("rouge1_max_retrieved", ""),
            "rouge1_max_ideal": qdata.get("rouge1_max_ideal", ""),
            "rouge1_retrieved_vs_ideal_ratio": qdata.get("rouge1_retrieved_vs_ideal_ratio", ""),
            "rubric_factual_accuracy": qdata.get("rubric_factual_accuracy", ""),
            "rubric_citation_accuracy": qdata.get("rubric_citation_accuracy", ""),
            "rubric_historiographical_depth": qdata.get("rubric_historiographical_depth", ""),
            "rubric_epistemic_calibration": qdata.get("rubric_epistemic_calibration", ""),
            "rubric_total": qdata.get("rubric_total", ""),
            "evaluator_notes": qdata.get("evaluator_notes", ""),
        })
    if not rows:
        return b""
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────
#
# Delegates entirely to run_rag_pipeline()

class _NoOpColBERT:
    """
    Stub ColBERT object passed to run_rag_pipeline() so it never tries to
    instantiate ColBERTSearcher (which would trigger Astra DB initialization).
    The pipeline checks `colbert_searcher is not None` before instantiating —
    passing this stub satisfies that check. perform_colbert_search=False ensures
    the stub's search() method is never actually called.
    """
    def search(self, *args, **kwargs):
        return pd.DataFrame()

# That function handles all data loading, Hay, retrieval, reranking, and Nicolay
# with the correct module signatures. The benchmark layer adds metrics on top.

def reranked_df_to_list(reranked_df: pd.DataFrame, corpus: dict = None) -> list[dict]:
    """
    Convert reranked_df (columns: Rank, Search Type, Text ID, Source, Summary,
    Key Quote, Relevance Score) into the list-of-dicts format used by benchmark
    metrics.

    DEBUG goals:
      • Preserve the raw Text ID emitted by the pipeline.
      • Record how/if we remapped IDs (and why).
      • Surface cases where multiple chunks share a Source label (lossy mapping risk).

    ID remapping:
      If a `corpus` dict is provided (keyed by *new* int IDs), we verify whether the
      parsed ID exists in that corpus. If it does not, we attempt to resolve via
      Source→new_id mapping (best-effort, but potentially lossy if Source is not unique).
    """
    # Build a source→new_id lookup from the corpus for ID remapping (best-effort)
    source_to_new_id = {}
    if corpus:
        for new_id, chunk in corpus.items():
            src = str(chunk.get("source", "")).strip()
            if src:
                source_to_new_id.setdefault(src, new_id)

    records: list[dict] = []
    for _, row in reranked_df.iterrows():
        raw_id = str(row.get("Text ID", "")).strip()
        row_source = str(row.get("Source", "")).strip()
        rank = row.get("Rank", None)

        # Parse raw ID to integer
        parsed_id = _extract_int_from_text_id(raw_id)

        # Remap to new corpus ID if the parsed ID is not present in the new corpus
        remapped_id = parsed_id
        remap_reason = None
        if corpus and parsed_id is not None and parsed_id not in corpus:
            remapped = source_to_new_id.get(row_source)
            if remapped is not None:
                remapped_id = remapped
                remap_reason = "source_match"
            else:
                remap_reason = "missing_in_corpus_no_source_match"

        # Normalize search type label — pipeline column is "Search Type"
        raw_search_type = str(row.get("Search Type", "")).strip()
        search_type = raw_search_type if raw_search_type else "Unknown"

        records.append({
            # Canonical fields used by metrics
            "rank": rank,
            "text_id_num": remapped_id,
            "text_id_str": f"Text #: {remapped_id}" if remapped_id is not None else raw_id,
            "source": row.get("Source", ""),
            "full_text": row.get("Key Quote", ""),  # (note) may actually be Key Quote, not full chunk text
            "reranker_score": row.get("Relevance Score"),
            "_search_type": search_type,

            # Debug-only fields (safe to ignore elsewhere)
            "_raw_text_id": raw_id,
            "_parsed_text_id": parsed_id,
            "_remapped": bool(remap_reason and remapped_id != parsed_id),
            "_remap_reason": remap_reason or "",
            "_row_source": row_source,
            "_parsed_id_in_corpus": bool(corpus and parsed_id is not None and parsed_id in corpus),
            "_remapped_id_in_corpus": bool(corpus and remapped_id is not None and remapped_id in corpus),
        })
    return records


def run_pipeline_for_query(
    qdef: dict,
    openai_api_key: str,
    cohere_api_key: str,
    corpus_file: str = "",
    status_cb=None,
    hay_model: str = "",
    nicolay_model: str = "",
) -> dict:
    """
    Run the full Nicolay pipeline for one benchmark query and return a complete
    result record. Delegates to run_rag_pipeline() — no retrieval reimplementation.

    Parameters
    ----------
    qdef          : benchmark query definition dict
    openai_api_key: passed through to run_rag_pipeline
    cohere_api_key: passed through to run_rag_pipeline
    corpus_file   : path to lincoln_speech_corpus_reindex_keep.(json|msgpack) (886-chunk corpus).
                    Used here for metrics and quote verification. Note: run_rag_pipeline()
                    loads its own corpus internally via load_lincoln_speech_corpus() in
                    data_utils.py — if retrieval returns only ~80 documents, that function
                    is also pointing at the wrong file and must be fixed there too.
    status_cb     : optional callable(str) for progress UI updates
    """
    from modules.rag_pipeline import run_rag_pipeline

    def status(msg):
        if status_cb:
            status_cb(msg)

    # Resolve active model IDs: use explicitly passed values (from call site
    # where selected_pair_key is in scope) or fall back to module globals.
    if not hay_model:
        hay_model = HAY_MODEL
    if not nicolay_model:
        nicolay_model = NICOLAY_MODEL

    qid = qdef["id"]
    query = qdef["query"]
    result = {
        "id": qid, "query": query,
        "category": qdef["category"],
        "expected_hay_type": qdef["expected_hay_type"],
        "expected_nicolay_type": qdef["expected_nicolay_type"],
        "ideal_docs_new": qdef["ideal_docs_new"],
        # HV-5: Model configuration tag — records exact pipeline state for this run.
        # Essential for multi-version comparison; retrofitting later requires re-running.
        "model_config_tag": f"hay={hay_model}|nicolay={nicolay_model}|reranker={COHERE_RERANK_MODEL}|k={RERANK_K}|colbert=disabled",
    }

    # ── 1. RUN FULL PIPELINE (with retry) ────────────────────────────────────
    # Nicolay v3 occasionally emits malformed JSON (unescaped characters or
    # structural errors in the response string). This is stochastic — temperature=0
    # does not guarantee identical outputs on every call. The retry wrapper catches
    # these transient parse failures and re-runs the full pipeline up to MAX_RETRIES
    # times before giving up. Each attempt is logged for article transparency.
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 2

    import time

    pipeline_out = None
    pipeline_attempts = []   # list of {attempt, error, error_type} for logging

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt > 1:
                status(f"🔄 Retry {attempt}/{MAX_RETRIES} for {qid} (prev: {pipeline_attempts[-1]['error_type']})...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                status("🔍 Running pipeline (Hay → retrieval → rerank → Nicolay)...")

            pipeline_out = run_rag_pipeline(
                user_query=query,
                perform_keyword_search=True,
                perform_semantic_search=True,
                perform_colbert_search=False,
                perform_reranking=True,
                colbert_searcher=_NoOpColBERT(),
                openai_api_key=openai_api_key,
                cohere_api_key=cohere_api_key,
                top_n_results=10,
            )
            # Success — log the attempt count and break
            pipeline_attempts.append({"attempt": attempt, "error": None, "error_type": None})
            break

        except Exception as e:
            err_type = type(e).__name__
            err_msg = str(e)
            pipeline_attempts.append({"attempt": attempt, "error": err_msg, "error_type": err_type})
            status(f"⚠️ Pipeline attempt {attempt} failed ({err_type}): {err_msg[:120]}")

            if attempt == MAX_RETRIES:
                # All retries exhausted — record full attempt log and bail
                result["pipeline_error"] = err_msg
                result["pipeline_error_type"] = err_type
                result["pipeline_attempts"] = attempt
                result["pipeline_retry_log"] = json.dumps(pipeline_attempts)
                status(f"❌ Pipeline failed after {MAX_RETRIES} attempts for {qid}.")
                return result

    # Record retry metadata on result (useful even on first-attempt success)
    result["pipeline_attempts"] = len(pipeline_attempts)
    result["pipeline_retry_log"] = json.dumps(pipeline_attempts)
    # Surface retry events in Streamlit UI for immediate visibility
    if len(pipeline_attempts) > 1:
        st.warning(
            f"⚠️ **{qid} required {len(pipeline_attempts)} attempts** — "
            f"{len(pipeline_attempts)-1} JSON parse failure(s) before success. "
            f"See retry log in debug expander."
        )

    # ── 2. UNPACK PIPELINE OUTPUT ─────────────────────────────────────────────
    hay_output      = pipeline_out.get("hay_output", {})
    reranked_df     = pipeline_out.get("reranked_results", pd.DataFrame())
    nicolay_output  = pipeline_out.get("nicolay_output", {})

    # DEBUG: capture raw reranked_df BEFORE any transformation so the debug
    # expander in the UI can show exactly what the pipeline returned.
    # This is the ground truth for diagnosing column name and ID issues.
    try:
        st.session_state[f"_debug_reranked_df_{qid}"] = reranked_df.copy()
    except Exception:
        pass

    # ── 2b. LOAD CORPUS for metrics and quote verification ────────────────────
    # Load directly from corpus_file (the 886-chunk reindexed JSON) rather than
    # via load_lincoln_speech_corpus(), which points at the old 80-doc corpus.
    corpus_for_verify = {}
    corpus_load_error = None
    if corpus_file and Path(corpus_file).exists():
        try:
            _raw = load_corpus_any(corpus_file)
            for item in _raw:
                tid = item.get("text_id", "")
                m = re.search(r"(\d+)", str(tid))
                if m:
                    corpus_for_verify[int(m.group(1))] = item
        except Exception as e:
            corpus_load_error = str(e)
    else:
        corpus_load_error = f"corpus_file not found or not set: {repr(corpus_file)}"

    # DEBUG: log corpus key count and any load error
    try:
        _size_msg = f"{len(corpus_for_verify)} keys (range: {min(corpus_for_verify) if corpus_for_verify else 'N/A'} – {max(corpus_for_verify) if corpus_for_verify else 'N/A'})"
        if corpus_load_error:
            _size_msg += f" ⚠️ LOAD ERROR: {corpus_load_error}"
        st.session_state[f"_debug_corpus_size_{qid}"] = _size_msg
        # Cache corpus for match card full-text display in Tab 1
        st.session_state[f"_corpus_{qid}"] = corpus_for_verify
    except Exception:
        pass

    # DEBUG: corpus file fingerprint (proves which corpus we verified against)
    try:
        _p = Path(corpus_file)
        _stat = _p.stat() if _p.exists() else None
        st.session_state[f"_debug_corpus_fingerprint_{qid}"] = {
            "corpus_file": str(corpus_file),
            "resolved_path": str(_p.resolve()) if _p.exists() else str(corpus_file),
            "exists": bool(_p.exists()),
            "size_bytes": int(_stat.st_size) if _stat else None,
            "mtime_iso": datetime.fromtimestamp(_stat.st_mtime).isoformat() if _stat else None,
        }
    except Exception:
        pass


    # ── 3. HAY METRICS ────────────────────────────────────────────────────────
    status("📊 Computing Hay metrics...")
    result["hay_output"] = hay_output
    hay_type = extract_hay_type(hay_output.get("query_assessment", ""))
    result["hay_task_type_raw"] = hay_type
    result["hay_task_type_correct"] = (hay_type == qdef["expected_hay_type"]) if hay_type else None
    result["hay_keyword_count"] = len(hay_output.get("weighted_keywords", {}))
    result["hay_spurious_fields"] = check_hay_spurious_fields(hay_output)
    result["hay_training_bleed"] = "This trains" in hay_output.get("query_assessment", "")

    # ── 4. RETRIEVAL METRICS ──────────────────────────────────────────────────
    # Convert reranked_df to list-of-dicts for benchmark metric functions.
    # Pass corpus so reranked_df_to_list can remap old parent IDs to new chunk IDs.
    reranked = reranked_df_to_list(reranked_df, corpus=corpus_for_verify) if not reranked_df.empty else []

    result["retrieved_doc_ids"] = [r["text_id_num"] for r in reranked]
    result["retrieval_search_types"] = [r.get("_search_type") or "Unknown" for r in reranked]
    result["reranker_scores"] = [r.get("reranker_score") for r in reranked]

    # HV-3: Top-5 retrieval path as a compact comma-separated string.
    # e.g. "Semantic,Semantic,Keyword,Semantic,Keyword"
    # Makes BM25 vs Hay keyword share immediately readable in CSV/Sheets.
    _top5_types = result["retrieval_search_types"][:5]
    result["retrieval_path_top5"] = ",".join(str(t) for t in _top5_types) if _top5_types else ""
    _keyword_count_top5 = sum(1 for t in _top5_types if str(t).lower() == "keyword")
    _semantic_count_top5 = sum(1 for t in _top5_types if str(t).lower() == "semantic")
    result["retrieval_keyword_count_top5"] = _keyword_count_top5
    result["retrieval_semantic_count_top5"] = _semantic_count_top5

    # HV-4: Reranker score distribution for top-5 results.
    # Low max score (<0.7) is a retrieval confidence warning flag.
    _scores_top5 = [s for s in result["reranker_scores"][:5] if s is not None]
    result["reranker_score_max_top5"] = round(max(_scores_top5), 6) if _scores_top5 else None
    result["reranker_score_min_top5"] = round(min(_scores_top5), 6) if _scores_top5 else None
    result["reranker_score_mean_top5"] = round(sum(_scores_top5) / len(_scores_top5), 6) if _scores_top5 else None

    # DEBUG: ID audit — shows raw IDs, parsed IDs, remaps, and corpus membership
    try:
        _audit_rows = []
        for r in reranked:
            _audit_rows.append({
                "rank": r.get("rank"),
                "raw_text_id": r.get("_raw_text_id"),
                "parsed_text_id": r.get("_parsed_text_id"),
                "final_text_id": r.get("text_id_num"),
                "search_type": r.get("_search_type"),
                "row_source": r.get("_row_source"),
                "remapped": r.get("_remapped"),
                "remap_reason": r.get("_remap_reason"),
                "parsed_in_corpus": r.get("_parsed_id_in_corpus"),
                "final_in_corpus": r.get("_remapped_id_in_corpus"),
            })
        st.session_state[f"_debug_id_audit_{qid}"] = _audit_rows

        _retrieved_ids = [r.get("text_id_num") for r in reranked if r.get("text_id_num") is not None]
        _missing_retrieved = sorted([tid for tid in _retrieved_ids if tid not in corpus_for_verify])
        st.session_state[f"_debug_missing_retrieved_{qid}"] = _missing_retrieved
    except Exception:
        pass


    metrics = compute_retrieval_metrics(reranked, qdef["ideal_docs_new"], qdef.get("ideal_docs_original"))
    result.update(metrics)

    # ── 5. NICOLAY METRICS ────────────────────────────────────────────────────
    status("✍️ Computing Nicolay metrics...")
    result["nicolay_output"] = nicolay_output

    synth_str = get_synthesis_assessment(nicolay_output)
    nicolay_type = extract_nicolay_type(synth_str)
    result["nicolay_synthesis_type_raw"] = nicolay_type
    # CF-3: Preserve the raw synthesis_assessment string before it is parsed.
    # Essential for diagnosing the T2 floor — confirms whether Nicolay is genuinely
    # self-classifying as T2 or whether the regex is failing to parse a correct value.
    result["nicolay_synthesis_assessment_raw"] = synth_str
    result["nicolay_synthesis_type_correct"] = (
        (nicolay_type == qdef["expected_nicolay_type"]) if nicolay_type else None
    )

    # HV-1: Capture Hay's initial_answer at the Nicolay-metrics stage so it is
    # available as a standalone field in logs/CSV (it is also inside hay_output
    # but buried in a nested dict — surfacing it top-level simplifies analysis).
    result["hay_initial_answer"] = str(hay_output.get("initial_answer", ""))
    # HV-2: Capture the full query_assessment string from Hay.  The single-letter
    # type extraction (A/B/C/D/E) discards nuance in Hay's prose description.
    result["hay_query_assessment_raw"] = str(hay_output.get("query_assessment", ""))

    schema_check = check_nicolay_schema(nicolay_output)
    result["nicolay_schema_complete"] = schema_check["complete"]
    result["nicolay_schema_fields"] = schema_check["fields"]

    final_answer_text = get_final_answer_text(nicolay_output)
    result["nicolay_final_answer_text"] = final_answer_text  # HV-6: full FinalAnswer text
    result["nicolay_final_answer_wordcount"] = len(final_answer_text.split()) if final_answer_text else 0

    match_analysis = nicolay_output.get("Match Analysis", {})
    relevance_map = {}
    if isinstance(match_analysis, dict):
        for mk, mv in match_analysis.items():
            if isinstance(mv, dict):
                relevance_map[mk] = mv.get("Relevance Assessment", "")
    result["nicolay_relevance_assessments"] = relevance_map

    # ── 6. QUOTE VERIFICATION ─────────────────────────────────────────────────
    # corpus_for_verify was built at step 2b above (new 886-chunk corpus).
    status("✅ Verifying quotes...")


    # DEBUG: Nicolay citation IDs vs corpus keys (catches ID drift instantly)
    try:
        _cited_ids = _extract_cited_ids(nicolay_output)
        _missing_cited = sorted([tid for tid in _cited_ids if tid not in corpus_for_verify])
        st.session_state[f"_debug_cited_ids_{qid}"] = _cited_ids
        st.session_state[f"_debug_missing_cited_{qid}"] = _missing_cited

        # Spot-check what the corpus thinks a few key IDs are (retrieved + cited)
        _retrieved_ids = [r.get("text_id_num") for r in reranked if r.get("text_id_num") is not None]
        _spot_ids = []
        for tid in (_retrieved_ids[:5] + _cited_ids[:5]):
            if tid is not None and tid not in _spot_ids:
                _spot_ids.append(tid)
        st.session_state[f"_debug_corpus_spotcheck_{qid}"] = [_chunk_signature(corpus_for_verify, tid) for tid in _spot_ids]
    except Exception:
        pass

    qv = verify_all_quotes(nicolay_output, reranked, corpus_for_verify)
    result["quote_verification"] = qv
    outcomes = [q.get("outcome") for q in qv]
    result["quotes_verified_count"]     = outcomes.count("verified")
    result["quotes_approx_count"]        = outcomes.count("approximate_quote")
    result["quotes_displaced_count"]     = outcomes.count("displacement")
    result["quotes_approx_displaced_count"] = outcomes.count("approximate_displacement")
    result["quotes_fabricated_count"]    = outcomes.count("fabrication")
    result["quotes_mislabeled_count"]    = outcomes.count("source_mislabeled")

    # ── 7. BLEU / ROUGE ───────────────────────────────────────────────────────
    status("📊 Computing BLEU/ROUGE scores...")
    # Use whichever ideal ID set matches the live corpus
    _ideal_for_bleu = (
        qdef.get("ideal_docs_original")
        if metrics.get("ideal_docs_set_used") == "original"
        else qdef["ideal_docs_new"]
    )
    bleu_rouge = compute_bleu_rouge(
        final_answer_text, reranked, corpus_for_verify, _ideal_for_bleu
    )
    result.update({k: v for k, v in bleu_rouge.items() if k != "_note"})
    result["bleu_rouge_note"] = bleu_rouge.get("_note", "")

    # ── 8. QUALITATIVE RUBRIC (filled manually in UI) ─────────────────────────
    result["rubric_factual_accuracy"] = None
    result["rubric_citation_accuracy"] = None
    result["rubric_historiographical_depth"] = None
    result["rubric_epistemic_calibration"] = None
    result["rubric_total"] = None
    result["evaluator_notes"] = ""
    result["critical_missing_evidence"] = qdef.get("critical_missing_evidence")
    result["watchlist"] = qdef.get("watchlist", [])

    # ── 9. CONFIDENCE SIGNALS ───────────────────────────────────────────────────────────────────────────
    # U12-equivalent five-signal confidence assessment. ROUGE is computed
    # against Key Quote text recovered from Nicolay's Match Analysis; a
    # richer recomputation runs in the Tab 1 display panel.
    status("📊 Computing confidence signals...")
    try:
        conf = compute_confidence_signals(result, corpus=corpus_for_verify)
        result.update(conf)
    except Exception as _conf_err:
        result.update({
            "confidence_rating": "unknown", "confidence_icon": "?",
            "confidence_explanation": f"Signal computation failed: {_conf_err}",
            "confidence_synth_type": None, "confidence_rouge1": None,
            "confidence_rouge2": None, "confidence_calib_warning": False,
            "confidence_spread": None, "confidence_n_sources": None,
            "confidence_max_score": None,
        })

    status(f"✓ {qid} complete")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown('<h1 style="font-family:Playfair Display,serif;color:#3d2b1f;">Nicolay Benchmark</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#7a6a5a;font-size:0.95rem;">Hay v4 + Nicolay v4 · Cohere rerank-v4.0-pro · 886-chunk corpus · 25 queries</p>', unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # API keys loaded from st.secrets — no sidebar fields exposed.
        # OpenAI: st.secrets["openai_api_key"] (or env fallback for local dev)
        # Cohere: st.secrets["cohere_api_key"] (or env fallback for local dev)
        def _get_secret(key: str, env_var: str) -> str:
            try:
                return st.secrets[key]
            except Exception:
                return os.environ.get(env_var, "")

        openai_key = _get_secret("openai_api_key", "OPENAI_API_KEY")
        cohere_key = _get_secret("cohere_api_key", "COHERE_API_KEY")

        # Auto-detect corpus across common repo layouts (root, Data/, data/)
        # Glob-based so it works regardless of case or Cloud working directory.
        _base = "lincoln_speech_corpus_reindex_keep"
        _candidates = [
            f"Data/{_base}.json", f"data/{_base}.json", f"{_base}.json",
            f"Data/{_base}.msgpack", f"data/{_base}.msgpack", f"{_base}.msgpack",
        ]
        # Also search via glob in case capitalisation differs on Cloud
        import glob as _glob
        _glob_hits = (
            _glob.glob(f"**/{_base}.json",    recursive=True) +
            _glob.glob(f"**/{_base}.msgpack", recursive=True)
        )
        _all_candidates = _candidates + [p for p in _glob_hits if p not in _candidates]
        _corpus_default = next((p for p in _all_candidates if Path(p).exists()), f"Data/{_base}.msgpack")
        corpus_file = st.text_input("Corpus path (JSON or MSGPACK)", value=_corpus_default)

        # ── Corpus path debug expander (always visible) ──────────────────────
        with st.expander("🗂️ Corpus path debug", expanded=not Path(corpus_file).exists()):
            import os as _os_dbg
            st.caption(f"**cwd:** `{_os_dbg.getcwd()}`")
            st.caption(f"**corpus_file value:** `{corpus_file}`")
            st.caption(f"**Path(corpus_file).exists():** `{Path(corpus_file).exists()}`")
            _dbg_found = [p for p in _all_candidates if Path(p).exists()]
            if _dbg_found:
                st.caption("**Candidate paths found on disk:**")
                for _dp in _dbg_found:
                    st.caption(f"  • `{_dp}` ({Path(_dp).stat().st_size // 1024} KB)")
            else:
                st.warning("No candidate corpus file found on disk. Check repo structure.")
                # Show top-level directory listing to diagnose Cloud layout
                try:
                    _top = sorted(_os_dbg.listdir("."))
                    st.caption("**Top-level directory listing:**")
                    st.code("\n".join(_top))
                    # Check one level deeper for Data/ or data/
                    for _sub in ("Data", "data", "pages"):
                        if _os_dbg.path.isdir(_sub):
                            _sub_files = sorted(_os_dbg.listdir(_sub))
                            st.caption(f"**{_sub}/ contents:**")
                            st.code("\n".join(_sub_files))
                except Exception as _le:
                    st.caption(f"Directory listing failed: {_le}")

        # ── Corpus load ───────────────────────────────────────────────────────
        _shared_key = f"_corpus_shared_{corpus_file}"
        _corpus_shared_current = st.session_state.get(_shared_key)

        if Path(corpus_file).exists():
            if not _corpus_shared_current:
                try:
                    _raw_shared = load_corpus_any(corpus_file)
                    _corpus_shared = {}
                    for _item in _raw_shared:
                        _tid = _item.get("text_id", "")
                        _m = re.search(r"(\d+)", str(_tid))
                        if _m:
                            _corpus_shared[int(_m.group(1))] = _item
                    st.session_state[_shared_key] = _corpus_shared
                    _corpus_shared_current = _corpus_shared
                except Exception as _ce:
                    st.session_state[_shared_key] = {}
                    st.error(f"⚠️ Corpus load failed: {_ce}")
            _corpus_size = len(_corpus_shared_current) if _corpus_shared_current else 0
            if _corpus_size >= 700:
                st.caption(f"✅ Corpus loaded — {_corpus_size} chunks in memory")
            elif _corpus_size > 0:
                st.warning(f"⚠️ Corpus loaded but only {_corpus_size} chunks — expected ~886. Check file.")
            else:
                st.error("❌ Corpus loaded 0 chunks — check corpus path.")
        else:
            if _corpus_shared_current:
                # Corpus was loaded in a prior run under a path that has since changed
                _corpus_size = len(_corpus_shared_current)
                st.caption(f"✅ Corpus in memory from prior load — {_corpus_size} chunks (path may have changed)")
            else:
                st.error(f"❌ Corpus file not found: `{corpus_file}` — full-text display will be unavailable.")
        st.warning(
            "**Pipeline corpus note:** The benchmark script loads the corpus above "
            "for metrics and quote verification. But `run_rag_pipeline()` loads its "
            "own corpus internally via `modules/data_utils.py → load_lincoln_speech_corpus()`. "
            "If retrieval returns IDs only in the 0–79 range, that function is pointing "
            "at the old 80-doc corpus — fix the path in `data_utils.py` to point at "
            "`lincoln_speech_corpus_reindex_keep.json`.",
            icon="⚠️"
        )
        st.markdown("---")
        st.markdown("### 📊 Google Sheets Logging")
        st.caption("Results are saved to Sheets on each query — durable across reruns on Streamlit Cloud.")
        sheets_enabled = st.checkbox("Enable Sheets logging", value=True)

        # Initialize Google Sheets logger
        # Reads credentials from st.secrets["gcp_service_account"] (set in Streamlit Cloud secrets)
        benchmark_logger = None
        gc_client = None
        if sheets_enabled:
            try:
                from google.oauth2 import service_account as gcp_sa
                import pygsheets
                if "gcp_service_account" in st.secrets:
                    _creds = gcp_sa.Credentials.from_service_account_info(
                        st.secrets["gcp_service_account"],
                        scopes=["https://www.googleapis.com/auth/drive"]
                    )
                    gc_client = pygsheets.authorize(custom_credentials=_creds)
                    from modules.data_logging import DataLogger
                    benchmark_logger = DataLogger(gc=gc_client, sheet_name="benchmark_results_data")
                    st.success("✅ Sheets connected")

                    # ── SHEETS DIAGNOSTICS ────────────────────────────────
                    if st.button("📋 Init Sheet Headers"):
                        init_benchmark_sheet_headers(gc_client)

                    if st.button("🔍 List All Accessible Spreadsheets"):
                        try:
                            all_sheets = gc_client.spreadsheet_titles()
                            st.write("**Spreadsheets visible to service account:**")
                            for title in all_sheets:
                                marker = " ← THIS ONE" if title == "benchmark_results_data" else ""
                                st.write(f"• `{repr(title)}`{marker}")
                            if "benchmark_results_data" not in all_sheets:
                                st.error("❌ No spreadsheet titled exactly 'benchmark_results_data' found. Check title spelling and sharing permissions.")
                        except Exception as e:
                            st.error(f"❌ Could not list spreadsheets: {type(e).__name__}: {e}")

                    if st.button("🧪 Test Sheets Write (with verification)"):
                        try:
                            # Step 1: Open sheet and show exactly which file it resolved to
                            sh = gc_client.open_by_key("1uQx9ERAHL0EKaI5QEpfywGxS40TpQ0Ao0HLfJe74auo")
                            st.write(f"**Resolved spreadsheet title:** `{sh.title}`")
                            st.write(f"**Spreadsheet ID (URL):** `{sh.id}`")
                            ws = sh.sheet1
                            st.write(f"**Sheet tab name:** `{ws.title}`  |  **Rows before write:** `{ws.rows}`")

                            # Step 2: Direct low-level write — bypass DataLogger entirely
                            # to confirm pygsheets itself can write to this sheet.
                            next_row = len(ws.get_all_records()) + 2
                            ws.update_value(f"A{next_row}", f"DIRECT_TEST_{datetime.now().isoformat()}")
                            st.success(f"✅ Direct write succeeded at row {next_row} — check column A of the sheet.")

                            # Step 3: Now test DataLogger path
                            test_record = {
                                "QueryID": "TEST_DATALOGGER",
                                "Query": "DataLogger path test",
                                "Category": "debug",
                                "ModelConfigTag": f"hay={HAY_MODEL}|nicolay={NICOLAY_MODEL}|reranker={COHERE_RERANK_MODEL}|k={RERANK_K}|colbert=disabled",
                                "HayModel": HAY_MODEL, "NicolayModel": NICOLAY_MODEL,
                                "HayTypeExpected": "", "HayTypeGot": "", "HayTypeCorrect": "",
                                "HayKeywordCount": 0, "HaySpuriousFields": "[]",
                                "HayTrainingBleed": "False", "InitialAnswer": "", "QueryAssessment": "",
                                "RetrievedDocIDs": "[]", "RetrievalSearchTypes": "[]",
                                "RetrievalPathTop5": "", "RetrievalKeywordCountTop5": 0, "RetrievalSemanticCountTop5": 0,
                                "RerankerScores": "[]", "RerankerScoreMaxTop5": 0.0, "RerankerScoreMinTop5": 0.0, "RerankerScoreMeanTop5": 0.0,
                                "PrecisionAt5": 0.0, "RecallAt5": 0.0, "CeilingAdjustedPrecision": 0.0,
                                "IdealDocsHit": "[]", "IdealDocsMissed": "[]",
                                "NicolayTypeExpected": "", "NicolayTypeGot": "", "NicolayTypeCorrect": "",
                                "NicolaySynthesisAssessmentRaw": "",
                                "SchemaComplete": "", "FinalAnswerWordCount": 0,
                                "QuotesVerified": 0, "QuotesApprox": 0, "QuotesDisplaced": 0, "QuotesApproxDisplaced": 0, "QuotesFabricated": 0,
                                "BleuMaxRetrieved": 0.0, "BleuAvgRetrieved": 0.0,
                                "Rouge1MaxRetrieved": 0.0, "Rouge1AvgRetrieved": 0.0,
                                "Rouge1MaxIdeal": 0.0, "Rouge1AvgIdeal": 0.0,
                                "Rouge1RetrievedVsIdealRatio": 0.0,
                                "RougeL_MaxRetrieved": 0.0, "RougeL_MaxIdeal": 0.0,
                                "CriticalMissingEvidence": "",
                                "RubricFactualAccuracy": "", "RubricCitationAccuracy": "",
                                "RubricHistoriographicalDepth": "", "RubricEpistemicCalibration": "",
                                "RubricTotal": "", "EvaluatorNotes": "DataLogger path test",
                            }
                            rows_before = len(ws.get_all_records())
                            benchmark_logger.record_api_outputs(test_record)
                            rows_after = len(ws.get_all_records())
                            if rows_after > rows_before:
                                st.success(f"✅ DataLogger write succeeded — rows went from {rows_before} to {rows_after}.")
                            else:
                                st.error(f"❌ DataLogger write appeared to succeed but row count unchanged ({rows_before} → {rows_after}). Data may be going to a different sheet tab.")
                                st.write(f"DataLogger's internal sheet object tab: `{benchmark_logger.sheet.title}`")
                        except Exception as e:
                            st.error(f"❌ Test failed at: {type(e).__name__}: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    st.warning("⚠️ No gcp_service_account in secrets — Sheets logging disabled.")
            except ImportError:
                st.warning("⚠️ pygsheets not installed — Sheets logging disabled.")
            except Exception as e:
                st.warning(f"⚠️ Sheets init failed: {e}")

        st.markdown("---")
        st.markdown("### 📋 Run Mode")
        run_mode = st.radio("Mode", ["Single Query", "Full Benchmark", "Resume from Checkpoint", "Compare Pair"],
                            label_visibility="collapsed")

        st.markdown("---")
        _pair_keys = list(MODEL_PAIRS.keys())
        import sys as _sys_mp
        _mod_mp = _sys_mp.modules[__name__]

        if run_mode == "Compare Pair":
            # ── Compare Pair: independent left / right selectors ──────────────
            st.markdown("### 🔬 Compare Pair")
            _cp_left_key = st.selectbox(
                "Left pair",
                _pair_keys,
                index=1,  # default: H4N3
                key="cp_left_key",
            )
            _cp_right_key = st.selectbox(
                "Right pair",
                _pair_keys,
                index=2,  # default: H4N4
                key="cp_right_key",
            )
            # Use left pair as the "active" pair for module globals (restores after compare)
            selected_pair_key = _cp_left_key
            _pair = MODEL_PAIRS[selected_pair_key]
            _mod_mp.HAY_MODEL     = _pair["hay"]
            _mod_mp.NICOLAY_MODEL = _pair["nicolay"]
            # Validation
            if _cp_left_key == _cp_right_key:
                st.warning("⚠️ Left and Right pairs must be different.")
            else:
                _cpl_lbl = MODEL_PAIRS[_cp_left_key]["label"]
                _cpr_lbl = MODEL_PAIRS[_cp_right_key]["label"]
                st.caption(
                    f"**L:** `{MODEL_PAIRS[_cp_left_key]['hay'].split(':')[-2]}` / "
                    f"`{MODEL_PAIRS[_cp_left_key]['nicolay'].split(':')[-2]}`  \n"
                    f"**R:** `{MODEL_PAIRS[_cp_right_key]['hay'].split(':')[-2]}` / "
                    f"`{MODEL_PAIRS[_cp_right_key]['nicolay'].split(':')[-2]}`"
                )
        else:
            # ── All other modes: single active-pair radio ─────────────────────
            st.markdown("### 🤖 Model Pair")
            selected_pair_key = st.radio(
                "Active model pair",
                _pair_keys,
                index=0,
                label_visibility="collapsed",
            )
            _pair = MODEL_PAIRS[selected_pair_key]
            # Update module-level variables so run_pipeline_for_query() picks up the selection.
            _mod_mp.HAY_MODEL     = _pair["hay"]
            _mod_mp.NICOLAY_MODEL = _pair["nicolay"]
            st.caption(
                f"🤖 **Hay:** `{_pair['hay'].split(':')[-2]}`  \n"
                f"📚 **Nicolay:** `{_pair['nicolay'].split(':')[-2]}`"
            )
            # Provide fallback values so compare-mode variables are always defined
            _cp_left_key  = _pair_keys[0]
            _cp_right_key = _pair_keys[1] if len(_pair_keys) > 1 else _pair_keys[0]

        st.markdown("---")
        show_group = st.multiselect("Query groups", ["core", "revised", "new"],
                                    default=["core", "revised", "new"])
        query_options = [q["id"] for q in BENCHMARK_QUERIES if q["group"] in show_group]
        selected_query_id = st.selectbox(
            "Select Query",
            query_options,
            format_func=lambda qid: f"{qid} — {QUERY_BY_ID[qid]['query'][:55]}..."
        )

        st.markdown("---")
        if st.button("📁 Export CSV"):
            results_data = st.session_state.get("results", empty_results())
            csv_bytes = export_csv(results_data)
            if csv_bytes:
                st.download_button("⬇️ Download CSV", csv_bytes,
                                   file_name="nicolay_benchmark_summary.csv", mime="text/csv")

    # ── LOAD SAVED RESULTS ────────────────────────────────────────────────────
    # session_state is the live working store. On Cloud, this resets on sleep/rerun.
    # The durable record is in Google Sheets. For local dev, also try loading from file.
    if "results" not in st.session_state:
        st.session_state.results = load_results()  # returns empty_results() on Cloud

    results = st.session_state.results

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab_run, tab_metrics, tab_rubric, tab_summary, tab_bleu = st.tabs([
        "▶️ Run", "📊 Metrics", "📝 Rubric", "📋 Summary", "🔬 BLEU/ROUGE"
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1: RUN
    # ═══════════════════════════════════════════════════════════════════════
    with tab_run:
        qdef = QUERY_BY_ID[selected_query_id]
        already_run = selected_query_id in results.get("queries", {})

        col_info, col_status = st.columns([3, 1])
        with col_info:
            st.markdown(f'<span class="query-badge">{qdef["id"]}</span> &nbsp; <strong>{qdef["query"]}</strong>', unsafe_allow_html=True)
            st.caption(f"Category: {qdef['category']} · Expected Hay: {qdef['expected_hay_type']} · Expected Nicolay: {qdef['expected_nicolay_type']} · Ideal docs: {qdef['ideal_docs_new']}")
        with col_status:
            if already_run:
                st.success("✅ Already run")
            else:
                st.info("⬜ Not yet run")

        if qdef.get("critical_missing_evidence"):
            st.warning(f"⚠️ **Critical Missing Evidence:** {qdef['critical_missing_evidence']}")
        if qdef.get("watchlist"):
            with st.expander("👁️ Watchlist items"):
                for w in qdef["watchlist"]:
                    st.markdown(f"- {w}")

        # Run button
        do_run = False
        if run_mode == "Single Query":
            do_run = st.button(f"▶️ Run {selected_query_id}", type="primary")
        elif run_mode == "Full Benchmark":
            if st.button("▶️ Run All 25 Queries", type="primary"):
                do_run = "all"
        elif run_mode == "Resume from Checkpoint":
            pending = [q["id"] for q in BENCHMARK_QUERIES if q["id"] not in results.get("queries", {})]
            st.info(f"Pending: {len(pending)} queries — {', '.join(pending) if pending else 'none'}")
            if pending and st.button("▶️ Resume", type="primary"):
                do_run = "resume"
        elif run_mode == "Compare Pair":
            _cp_lbl_l = MODEL_PAIRS[_cp_left_key]["label"]
            _cp_lbl_r = MODEL_PAIRS[_cp_right_key]["label"]
            if _cp_left_key == _cp_right_key:
                st.warning("⚠️ Left and Right pairs must be different to compare.")
            else:
                st.info(
                    f"Runs **{selected_query_id}** through **{_cp_lbl_l}** (left) "
                    f"and **{_cp_lbl_r}** (right) back-to-back, "
                    f"then shows a side-by-side comparison."
                )
            if st.button(f"▶️ Compare {selected_query_id}", type="primary"):
                do_run = "compare"

        # Validate prerequisites — run_rag_pipeline loads its own data, so we
        # only need to verify API keys are present before dispatching.
        def can_run():
            if not openai_key:
                st.error("OpenAI API key not found. Set 'openai_api_key' in Streamlit secrets (or OPENAI_API_KEY env var for local dev).")
                return False
            if not cohere_key:
                st.error("Cohere API key not found. Set 'cohere_api_key' in Streamlit secrets (or COHERE_API_KEY env var for local dev).")
                return False
            return True

        if do_run:
            if can_run():
                queries_to_run = []
                if do_run == "all":
                    queries_to_run = BENCHMARK_QUERIES
                elif do_run == "resume":
                    queries_to_run = [q for q in BENCHMARK_QUERIES if q["id"] not in results.get("queries", {})]
                elif do_run == "compare":
                    # Compare Pair: run selected query through the two user-chosen pairs
                    # back-to-back, then display side-by-side. Does not write to the
                    # main results store — stored separately under session_state.
                    import sys as _sys_cp
                    _mod_cp = _sys_cp.modules[__name__]
                    _compare_results  = {}
                    _compare_progress = st.progress(0)
                    _compare_status   = st.empty()

                    for _cp_step, _cp_pk in enumerate([_cp_left_key, _cp_right_key]):
                        _cp_cfg = MODEL_PAIRS[_cp_pk]
                        _cp_side = "left" if _cp_step == 0 else "right"
                        _compare_status.info(
                            f"🔄 Running {selected_query_id} with "
                            f"{_cp_cfg['label']} ({_cp_side}, {_cp_step + 1}/2)…"
                        )
                        _mod_cp.HAY_MODEL     = _cp_cfg["hay"]
                        _mod_cp.NICOLAY_MODEL = _cp_cfg["nicolay"]
                        with st.spinner(f"Processing {selected_query_id} [{_cp_cfg['label']}]…"):
                            _cp_qr = run_pipeline_for_query(
                                qdef,
                                openai_api_key=openai_key,
                                cohere_api_key=cohere_key,
                                corpus_file=corpus_file,
                                status_cb=lambda msg: _compare_status.info(msg),
                                hay_model=_cp_cfg["hay"],
                                nicolay_model=_cp_cfg["nicolay"],
                            )
                        _cp_qr["_pair_label"] = _cp_cfg["label"]
                        _compare_results[_cp_cfg["label"]] = _cp_qr
                        _compare_progress.progress((_cp_step + 1) / 2)

                    # Restore module globals to whichever pair is "active" (left)
                    _mod_cp.HAY_MODEL     = MODEL_PAIRS[selected_pair_key]["hay"]
                    _mod_cp.NICOLAY_MODEL = MODEL_PAIRS[selected_pair_key]["nicolay"]
                    st.session_state[f"_compare_{selected_query_id}"] = _compare_results
                    _compare_status.success(f"✅ Compare complete — {selected_query_id}")

                    # ── Side-by-side comparison display ─────────────────────────
                    st.markdown("---")
                    _cp_lbl_l2 = MODEL_PAIRS[_cp_left_key]["label"]
                    _cp_lbl_r2 = MODEL_PAIRS[_cp_right_key]["label"]
                    st.markdown(f"### 🔬 {_cp_lbl_l2} vs {_cp_lbl_r2}")
                    _cp_labels = list(_compare_results.keys())
                    _cp_cols   = st.columns(len(_cp_labels))

                    for _cpc, _cpl in zip(_cp_cols, _cp_labels):
                        _cr = _compare_results[_cpl]
                        with _cpc:
                            st.markdown(f"#### {_cpl}")

                            # ── Metrics summary row ───────────────────────────
                            _cr_ci  = _cr.get("confidence_icon", "?")
                            _cr_cl  = (_cr.get("confidence_rating") or "unknown").capitalize()
                            _cr_t   = _cr.get("nicolay_synthesis_type_raw", "?")
                            _cr_p5  = _cr.get("precision_at_5", 0) or 0
                            _cr_r5  = _cr.get("recall_at_5", 0) or 0
                            _cr_fab = _cr.get("quotes_fabricated_count", 0) or 0
                            _cr_dis = _cr.get("quotes_displaced_count", 0) or 0
                            _cr_r1  = _cr.get("confidence_rouge1")
                            _cr_cw  = _cr.get("confidence_calib_warning", False)
                            _cr_spr = _cr.get("confidence_spread")
                            _cr_nsrc = _count_distinct_sources_from_qv(_cr.get("quote_verification", []))

                            st.markdown(
                                f"{_cr_ci} **{_cr_cl}** confidence &nbsp;·&nbsp; Type `{_cr_t}`",
                                unsafe_allow_html=True
                            )
                            st.markdown(
                                f"**P@5:** {_cr_p5:.2f} &nbsp;·&nbsp; **R@5:** {_cr_r5:.2f} &nbsp;·&nbsp; "
                                f"**Speeches:** {_cr_nsrc}",
                                unsafe_allow_html=True
                            )
                            _fab_icon = "🚨" if _cr_fab > 0 else "✅"
                            st.markdown(
                                f"{_fab_icon} **Fab:** {_cr_fab} &nbsp;·&nbsp; **Displaced:** {_cr_dis}"
                                + (f" &nbsp;·&nbsp; **ROUGE-1:** {_cr_r1:.3f}" if _cr_r1 is not None else ""),
                                unsafe_allow_html=True
                            )
                            if _cr_spr is not None:
                                _cr_spr_lbl = "✅ Differentiated" if _cr_spr >= 0.20 else ("⚠️ Flat" if _cr_spr < 0.10 else "Moderate")
                                st.caption(f"Spread: {_cr_spr:.3f} — {_cr_spr_lbl}")
                            if _cr_cw:
                                st.warning("⚠️ Calibration warning: flat spread")

                            st.divider()

                            # ── HAY OUTPUT ───────────────────────────────────
                            st.markdown("##### 🔍 Hay")
                            _cr_hay = _cr.get("hay_output", {}) or {}
                            st.markdown(f"**Type:** `{_cr.get('hay_task_type_raw','?')}` "
                                        + ("✅" if _cr.get("hay_task_type_correct") else "❌"))
                            with st.expander("Initial Answer & Assessment", expanded=False):
                                st.markdown("**Initial Answer:**")
                                st.markdown(_cr_hay.get("initial_answer", "_none_"))
                                st.markdown("**Query Assessment:**")
                                st.markdown(_cr_hay.get("query_assessment", "_none_"))
                            with st.expander("Keywords", expanded=False):
                                st.markdown("**Weighted Keywords:**")
                                st.json(_cr_hay.get("weighted_keywords", {}))
                                st.markdown(f"**Year:** {_cr_hay.get('year_keywords', [])}")
                                st.markdown(f"**Text:** {_cr_hay.get('text_keywords', [])}")
                            if _cr.get("hay_spurious_fields"):
                                st.warning(f"⚠️ Spurious fields: {_cr['hay_spurious_fields']}")
                            if _cr.get("hay_training_bleed"):
                                st.warning("⚠️ Training bleed in query_assessment")

                            st.divider()

                            # ── RETRIEVAL ────────────────────────────────────
                            st.markdown("##### 📚 Retrieval")
                            _cr_ids    = _cr.get("retrieved_doc_ids", [])
                            _cr_scores = _cr.get("reranker_scores", [])
                            _cr_types  = _cr.get("retrieval_search_types", [])
                            _cr_ideal  = set(qdef.get("ideal_docs_new", []))
                            for _ri, (_rt, _rs_val, _rtype) in enumerate(
                                    zip(_cr_ids, _cr_scores, _cr_types)):
                                _r_hit = "✅" if _rt in _cr_ideal else "❌"
                                _r_sc  = f"{_rs_val:.4f}" if _rs_val is not None else "N/A"
                                st.markdown(
                                    f"{_r_hit} **{_ri+1}** — #{_rt} &nbsp;·&nbsp; "
                                    f"{_r_sc} &nbsp;·&nbsp; _{_rtype}_",
                                    unsafe_allow_html=True
                                )

                            st.divider()

                            # ── NICOLAY OUTPUT ───────────────────────────────
                            st.markdown("##### ✍️ Nicolay")
                            _cr_fa    = _cr.get("nicolay_final_answer_text", "") or ""
                            _cr_qvl   = _cr.get("quote_verification", [])

                            # Build annotation lists for inline verification markers
                            _cr_iv_v  = [(q.get("cited_passage",""), q.get("cited_chunk_text_id",""), "")
                                          for q in _cr_qvl if q.get("outcome") in ("verified", "approximate_quote")]
                            _cr_iv_d  = [q.get("cited_passage","") for q in _cr_qvl
                                          if q.get("outcome") in ("displacement","approximate_displacement")]
                            _cr_iv_u  = [q.get("cited_passage","") for q in _cr_qvl
                                          if q.get("outcome") == "fabrication"]
                            _cr_iv_ml = [q.get("cited_passage","") for q in _cr_qvl
                                          if q.get("outcome") == "source_mislabeled"]
                            _QPAIRS   = [('“','”'),('"','"'),('‘','’'),("'","'")]
                            _cr_annot = _cr_fa
                            for _iq, _, __ in _cr_iv_v:
                                if not _iq: continue
                                for _oq, _cq in _QPAIRS:
                                    _lit = _oq + _iq + _cq
                                    if _lit in _cr_annot:
                                        _cr_annot = _cr_annot.replace(_lit, _lit + " ✅", 1); break
                            for _iq in _cr_iv_d:
                                if not _iq: continue
                                for _oq, _cq in _QPAIRS:
                                    _lit = _oq + _iq + _cq
                                    if _lit in _cr_annot:
                                        _cr_annot = _cr_annot.replace(_lit, _lit + " 🔀", 1); break
                            for _iq in _cr_iv_u:
                                if not _iq: continue
                                for _oq, _cq in _QPAIRS:
                                    _lit = _oq + _iq + _cq
                                    if _lit in _cr_annot:
                                        _cr_annot = _cr_annot.replace(_lit, _lit + " ⚠️", 1); break
                            for _iq in _cr_iv_ml:
                                if not _iq: continue
                                for _oq, _cq in _QPAIRS:
                                    _lit = _oq + _iq + _cq
                                    if _lit in _cr_annot:
                                        _cr_annot = _cr_annot.replace(_lit, _lit + " 🏷️", 1); break

                            # FinalAnswer with inline verification markers
                            with st.expander("✍️ FinalAnswer", expanded=True):
                                st.markdown(_cr_annot)
                                _cr_iv_leg = []
                                if _cr_iv_v:  _cr_iv_leg.append("✅ verified against corpus")
                                if _cr_iv_d:  _cr_iv_leg.append("🔀 found in document, displaced chunk")
                                if _cr_iv_u:  _cr_iv_leg.append("⚠️ not found — possible fabrication")
                                if _cr_iv_ml: _cr_iv_leg.append("🏷️ text verified but source attribution wrong")
                                if _cr_iv_leg:
                                    st.caption(" · ".join(_cr_iv_leg))
                                st.caption(f"Word count: {_cr.get('nicolay_final_answer_wordcount', 0)}")

                            # ── QUOTE VERIFICATION debug table ────────────────
                            with st.expander("🔎 Quote Verification", expanded=False):
                                if _cr_qvl:
                                    _cp_qv_icons = {
                                        "verified": "✅", "approximate_quote": "🟡",
                                        "displacement": "⚠️", "approximate_displacement": "🟠",
                                        "fabrication": "🚨", "source_mislabeled": "🏷️",
                                        "too_short": "—",
                                    }
                                    for _cp_qv in _cr_qvl:
                                        _cp_icon = _cp_qv_icons.get(_cp_qv.get("outcome",""), "?")
                                        st.markdown(
                                            f"{_cp_icon} **{_cp_qv.get('match','')}** "
                                            f"({_cp_qv.get('text_id','')}) — *{_cp_qv.get('outcome','')}*"
                                        )
                                        if _cp_qv.get("cited_passage"):
                                            st.caption(f'"{str(_cp_qv["cited_passage"])[:150]}..."')
                                        st.caption(
                                            f"cited_num={_cp_qv.get('cited_num')} | "
                                            f"cited_chunk_present={_cp_qv.get('cited_chunk_present')} | "
                                            f"cited_source={(_cp_qv.get('cited_chunk_source') or '')[:80]}"
                                        )
                                        if _cp_qv.get("cited_chunk_preview"):
                                            st.caption(f"cited_chunk_preview: {_cp_qv.get('cited_chunk_preview')}")
                                        if _cp_qv.get("match_method") or _cp_qv.get("approx_score") is not None:
                                            _cp_ms = _cp_qv.get("match_method","")
                                            _cp_sc = _cp_qv.get("approx_score", None)
                                            st.caption(f"match_method={_cp_ms}" + (
                                                f" | approx_score={_cp_sc:.2f}" if isinstance(_cp_sc, (int,float)) else ""))
                                        if _cp_qv.get("match_chunk_num") is not None:
                                            st.caption(
                                                f"found_at_chunk={_cp_qv.get('match_chunk_num')} | "
                                                f"found_source={(_cp_qv.get('match_chunk_source') or '')[:80]}"
                                            )
                                        if _cp_qv.get("note"):
                                            st.caption(_cp_qv["note"])
                                else:
                                    st.info("No quote verification data for this result.")

                            # ── MATCH ANALYSIS CARDS (full, with corpus highlight) ──
                            # Note: NO sub-columns inside each match card.
                            # We are already inside a top-level pair column (_cpc),
                            # so one more column level is allowed but creating ANOTHER
                            # level inside st.container() exceeds Streamlit's 2-deep limit.
                            # Cards stack vertically; relevance + verification badges are HTML.
                            _cr_ma = (_cr.get("nicolay_output") or {}).get("Match Analysis", {})
                            if _cr_ma and isinstance(_cr_ma, dict):
                                st.markdown("---")
                                st.markdown("##### 🎯 Match Analysis")
                                _cr_sc_map = {str(t): s for t, s in
                                              zip(_cr.get("retrieved_doc_ids",[]),
                                                  _cr.get("reranker_scores",[]))}
                                import re as _cp_re2
                                for _cmi, (_cmk, _cmv) in enumerate(_cr_ma.items()):
                                    if not isinstance(_cmv, dict): continue
                                    _cm_tid  = str(_cmv.get("Text ID",""))
                                    _cm_src  = (_cmv.get("Source","") or "").strip()
                                    if _cm_src.lower().startswith("source:"):
                                        _cm_src = _cm_src[len("source:"):].strip()
                                    _cm_kq   = _cmv.get("Key Quote","")
                                    _cm_rel  = _cmv.get("Relevance Assessment","")
                                    _cm_sum  = _cmv.get("Summary","")
                                    _cm_hist = _cmv.get("Historical Context","")
                                    _cm_sc   = _cr_sc_map.get(_cm_tid)
                                    # Join QV record by integer ID (cited_num is stored as int;
                                    # _cm_tid may be "Text #: 311" or "311" — extract int for
                                    # reliable comparison) or fall back to match key equality.
                                    _cm_tid_int = _extract_int_from_text_id(_cm_tid)
                                    _cm_qv   = next((q for q in _cr_qvl
                                                     if (_cm_tid_int is not None and q.get("cited_num") == _cm_tid_int)
                                                     or q.get("match","") == _cmk), None)
                                    _cm_out  = (_cm_qv or {}).get("outcome","")

                                    # Relevance badge (inline HTML, no sub-column needed)
                                    _cm_rt = (_cm_rel or "").lower()
                                    if "high" in _cm_rt:   _cm_bg,_cm_fg = "#d4edda","#155724"
                                    elif "medium" in _cm_rt or "moderate" in _cm_rt: _cm_bg,_cm_fg = "#fff3cd","#856404"
                                    elif "low" in _cm_rt:  _cm_bg,_cm_fg = "#f8d7da","#721c24"
                                    else:                   _cm_bg,_cm_fg = "#e2e3e5","#383d41"
                                    _cm_rel_lbl = _cp_re2.split(r"[—,]", _cm_rel or "N/A")[0].strip()[:30]
                                    _cm_rel_badge = (
                                        f'<span style="background:{_cm_bg};color:{_cm_fg};padding:2px 8px;'                                        f'border-radius:8px;font-size:0.8em;font-weight:700;margin-left:8px;'                                        f'display:inline-block;">{_cm_rel_lbl}</span>'
                                    )

                                    # Verification badge (full descriptive text)
                                    if _cm_out == "verified":
                                        _cm_vbadge = '<span style="color:#155724;font-size:0.85em;font-weight:600;">✅ Quote verified in corpus</span>'
                                    elif _cm_out in ("displacement","approximate_displacement"):
                                        _cm_vbadge = '<span style="color:#664d03;background:#fff3cd;padding:1px 6px;border-radius:4px;font-size:0.85em;font-weight:600;">🔀 Found in document, displaced chunk</span>'
                                    elif _cm_out == "approximate_quote":
                                        _cm_vbadge = '<span style="color:#155724;font-size:0.85em;font-weight:500;">🟡 Approximate match confirmed</span>'
                                    elif _cm_out == "fabrication":
                                        _cm_vbadge = '<span style="color:#721c24;font-size:0.85em;font-weight:600;">⚠️ Not found in corpus</span>'
                                    elif _cm_out == "source_mislabeled":
                                        _cm_vbadge = '<span style="color:#495057;background:#e2e3e5;padding:1px 6px;border-radius:4px;font-size:0.85em;font-weight:600;">🏷️ Text found — source attribution wrong</span>'
                                    else:
                                        _cm_vbadge = '<span style="color:#6c757d;font-size:0.85em;">⬜ Not verified</span>'

                                    with st.container(border=True):
                                        # Title + relevance badge on same line via HTML
                                        st.markdown(
                                            f'**{_cmk}** {_cm_rel_badge}',
                                            unsafe_allow_html=True
                                        )
                                        # Full untruncated source
                                        st.markdown(f"**ID:** {_cm_tid}  ·  **Source:** {_cm_src}")
                                        if _cm_sc is not None:
                                            st.caption(f"Reranker score: {_cm_sc:.3f}")
                                        if _cm_kq:
                                            _cm_kq_disp = _cm_kq[:320] + ("…" if len(_cm_kq) > 320 else "")
                                            st.markdown(f'> *"{_cm_kq_disp}"*')
                                        if _cm_sum:
                                            _cm_sum_disp = _cm_sum[:300] + ("…" if len(_cm_sum) > 300 else "")
                                            st.markdown(f"**Nicolay's Analysis:** {_cm_sum_disp}")
                                        st.markdown(_cm_vbadge, unsafe_allow_html=True)

                                        # Full-text expander with corpus highlight
                                        with st.expander(f"Full text & highlight — {_cmk}", expanded=False):
                                            if _cm_hist:
                                                st.markdown(f"**Historical Context:** {_cm_hist}")
                                            # _get_or_load_corpus() tries session_state first,
                                            # then loads from disk on demand — no prior pipeline
                                            # run required.  Returns {} if file not found.
                                            _cp_corpus_ss = _get_or_load_corpus(corpus_file)
                                            # Use _extract_int_from_text_id() rather than bare int()
                                            # so both "311" and "Text #: 311" formats resolve correctly.
                                            _cp_int_id = _extract_int_from_text_id(_cm_tid)
                                            _cp_chunk = _cp_corpus_ss.get(_cp_int_id) if _cp_int_id is not None else None
                                            if _cp_chunk:
                                                _cp_ft = _cp_chunk.get("full_text","")
                                                if _cp_ft:
                                                    _cp_html = _cp_ft.replace("\n","<br>")
                                                    if _cm_kq and _cm_kq in _cp_ft:
                                                        _cp_html = _cp_html.replace(_cm_kq, f"<mark>{_cm_kq}</mark>", 1)
                                                    elif _cm_kq:
                                                        _cp_segs = [p.strip() for p in _cm_kq.split("...") if p.strip()]
                                                        for _cp_seg in _cp_segs:
                                                            if len(_cp_seg) > 20 and _cp_seg in _cp_ft:
                                                                _cp_html = _cp_html.replace(_cp_seg, f"<mark>{_cp_seg}</mark>", 1)
                                                                break
                                                    st.markdown("**Full Text with Highlighted Quote:**")
                                                    st.markdown(
                                                        f'<div style="font-size:0.95em;line-height:1.65;">{_cp_html}</div>',
                                                        unsafe_allow_html=True
                                                    )
                                                else:
                                                    st.info("Full text not available for this chunk.")
                                            elif not _cp_corpus_ss:
                                                st.warning(
                                                    f"⚠️ Corpus not loaded — "
                                                    f"`{corpus_file}` not found on disk. "
                                                    f"Check path in sidebar debug expander."
                                                )
                                            else:
                                                st.info(f"Chunk {_cm_tid} not found in corpus ({len(_cp_corpus_ss)} chunks loaded).")

                            # ── RAW JSON OUTPUT ───────────────────────────────
                            with st.expander("🔬 Raw JSON — Nicolay output", expanded=False):
                                _cr_nico_raw = _cr.get("nicolay_output") or {}
                                if _cr_nico_raw:
                                    st.json(_cr_nico_raw)
                                else:
                                    st.info("nicolay_output not stored in this result.")
                            with st.expander("🔬 Raw JSON — Hay output", expanded=False):
                                _cr_hay_raw = _cr.get("hay_output") or {}
                                if _cr_hay_raw:
                                    st.json(_cr_hay_raw)
                                else:
                                    st.info("hay_output not stored in this result.")

                            # ── CONFIDENCE PANEL ────────────────────────────────
                            st.divider()
                            _cr_expl   = _cr.get("confidence_explanation", "")
                            _cr_r2     = _cr.get("confidence_rouge2")
                            _cr_nsrc2  = _cr.get("confidence_n_sources", 0) or 0
                            _cr_max_sc = _cr.get("confidence_max_score")
                            _cr_st_num = _cr.get("confidence_synth_type") or 0
                            _cr_nv     = len(_cr_iv_v)
                            _cr_nd     = len(_cr_iv_d)
                            _cr_nu     = len(_cr_iv_u)
                            _cr_total_q = _cr_nv + _cr_nd + _cr_nu

                            _cr_conf_hdr = f"📊 Confidence: {_cr_ci} {_cr_cl}"
                            with st.expander(_cr_conf_hdr, expanded=False):
                                # Overall rating banner
                                st.markdown(f"**Overall: {_cr_ci} {_cr_cl}**")
                                if _cr_expl:
                                    st.markdown(_cr_expl)
                                st.divider()

                                # Signal 1 — Quote verification
                                _cr_nml = _cr.get("quotes_mislabeled_count", 0) or 0
                                _cr_total_q = _cr_nv + _cr_nd + _cr_nu + _cr_nml
                                _cs1, _cs2 = st.columns([1, 2])
                                with _cs1: st.caption("**Quote verification**")
                                with _cs2:
                                    if _cr_total_q == 0:
                                        st.markdown("⬜ No direct quotes")
                                        st.caption("No quoted text in FinalAnswer.")
                                    elif _cr_nu == 0 and _cr_nd == 0 and _cr_nml == 0:
                                        st.markdown(f"✅ {_cr_total_q}/{_cr_total_q} verified")
                                        st.caption("All quoted passages confirmed in corpus.")
                                    elif _cr_nu > 0:
                                        st.markdown(f"🔴 {_cr_nv}/{_cr_total_q} verified — **{_cr_nu} not found**")
                                        st.caption("One or more quotes not found anywhere in corpus.")
                                    elif _cr_nml > 0:
                                        st.markdown(f"🏷️ {_cr_nv}/{_cr_total_q} correctly attributed — **{_cr_nml} mislabeled**")
                                        st.caption("Quote text found in corpus but source attribution wrong.")
                                    else:
                                        st.markdown(f"🔀 {_cr_nv}/{_cr_total_q} verified — **{_cr_nd} displaced**")
                                        st.caption("Quote in right document but wrong chunk.")
                                st.divider()

                                # Signal 2 — ROUGE corpus grounding
                                _cs1, _cs2 = st.columns([1, 2])
                                with _cs1: st.caption("**Corpus grounding**")
                                with _cs2:
                                    if _cr_r1 is not None:
                                        _r1_icon = "✅" if _cr_r1 >= 0.40 else ("⚠️" if _cr_r1 >= 0.25 else "🔴")
                                        st.markdown(f"{_r1_icon} ROUGE-1: `{_cr_r1:.3f}`" +
                                                    (f"  ·  ROUGE-2: `{_cr_r2:.3f}`" if _cr_r2 is not None else ""))
                                        _r_thr = 0.45 if _cr_st_num in (1,2) else (None if _cr_st_num == 3 else 0.30)
                                        if _r_thr is None:
                                            st.caption("Type 3 (absence) — ROUGE threshold not applied.")
                                        else:
                                            st.caption(f"Threshold for this synthesis type: {_r_thr:.2f}")
                                    else:
                                        st.markdown("⬜ Not computed")
                                st.divider()

                                # Signal 3 — Reranker score ceiling
                                _cs1, _cs2 = st.columns([1, 2])
                                with _cs1: st.caption("**Retrieval ceiling**")
                                with _cs2:
                                    if _cr_max_sc is not None:
                                        _sc_icon = "✅" if _cr_max_sc >= 0.55 else ("⚠️" if _cr_max_sc >= 0.35 else "🔴")
                                        st.markdown(f"{_sc_icon} Max reranker score: `{_cr_max_sc:.3f}`")
                                        st.caption("Score ≥ 0.55 indicates strong retrieval signal.")
                                    else:
                                        st.markdown("⬜ Not available")
                                st.divider()

                                # Signal 4 — Spread / calibration
                                _cs1, _cs2 = st.columns([1, 2])
                                with _cs1: st.caption("**Score spread**")
                                with _cs2:
                                    if _cr_spr is not None:
                                        _spr_icon = "✅" if _cr_spr >= 0.20 else ("⚠️" if _cr_spr < 0.10 else "🟡")
                                        st.markdown(f"{_spr_icon} Spread: `{_cr_spr:.3f}`")
                                        if _cr_cw:
                                            st.caption("⚠️ Calibration warning: scores are flat — retriever "
                                                       "may be off-target for this query.")
                                        else:
                                            st.caption("Spread ≥ 0.20 = discriminating retrieval; "
                                                       "< 0.10 = flat (suspect).")
                                    else:
                                        st.markdown("⬜ Not available")
                                st.divider()

                                # Signal 5 — Source diversity
                                _cs1, _cs2 = st.columns([1, 2])
                                with _cs1: st.caption("**Source diversity**")
                                with _cs2:
                                    _div_icon = "✅" if _cr_nsrc2 >= 3 else ("⚠️" if _cr_nsrc2 == 2 else "🔴")
                                    st.markdown(f"{_div_icon} **{_cr_nsrc2}** distinct Lincoln speech(es) in top-5")
                                    st.caption("≥ 3 distinct speeches = well-diversified retrieval.")

                    queries_to_run = []   # skip normal loop
                else:
                    queries_to_run = [qdef]

                progress_bar = st.progress(0)
                status_box = st.empty()

                for i, q in enumerate(queries_to_run):
                    status_box.info(f"Running {q['id']} ({i+1}/{len(queries_to_run)})...")
                    with st.spinner(f"Processing {q['id']}..."):
                        qresult = run_pipeline_for_query(
                            q,
                            openai_api_key=openai_key,
                            cohere_api_key=cohere_key,
                            corpus_file=corpus_file,
                            status_cb=lambda msg: status_box.info(msg),
                            hay_model=MODEL_PAIRS[selected_pair_key]["hay"],
                            nicolay_model=MODEL_PAIRS[selected_pair_key]["nicolay"],
                        )
                    results["queries"][q["id"]] = qresult
                    # Persist: Sheets first (durable on Cloud), then local file (dev fallback)
                    log_query_to_sheets(benchmark_logger, qresult)
                    save_results(results)
                    st.session_state.results = results
                    progress_bar.progress((i + 1) / len(queries_to_run))

                status_box.success(f"✅ Done — {len(queries_to_run)} queries completed.")

        # Display results if available
        if selected_query_id in results.get("queries", {}):
            qr = results["queries"][selected_query_id]
            st.markdown("---")

            # Hay output
            with st.expander("🔍 Hay Output", expanded=False):
                hay_out = qr.get("hay_output", {})
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Initial Answer:**")
                    st.markdown(hay_out.get("initial_answer", "_none_"))
                    st.markdown(f"**Query Assessment:**")
                    st.markdown(hay_out.get("query_assessment", "_none_"))
                with col2:
                    st.markdown("**Weighted Keywords:**")
                    st.json(hay_out.get("weighted_keywords", {}))
                    st.markdown(f"**Year Keywords:** {hay_out.get('year_keywords', [])}")
                    st.markdown(f"**Text Keywords:** {hay_out.get('text_keywords', [])}")
                if qr.get("hay_spurious_fields"):
                    st.warning(f"⚠️ Spurious fields detected: {qr['hay_spurious_fields']}")
                if qr.get("hay_training_bleed"):
                    st.warning("⚠️ Training signal bleed detected in query_assessment")

            # Retrieval summary
            with st.expander("📚 Retrieved Documents", expanded=False):
                retrieved_ids = qr.get("retrieved_doc_ids", [])
                ideal = set(qdef["ideal_docs_new"])
                scores = qr.get("reranker_scores", [])
                types = qr.get("retrieval_search_types", [])
                for i, (tid, score, stype) in enumerate(zip(retrieved_ids, scores, types)):
                    hit = "✅" if tid in ideal else "❌"
                    score_str = f"{score:.4f}" if score is not None else "N/A"
                    st.markdown(f"{hit} **Rank {i+1}** — Text #: {tid} &nbsp;·&nbsp; Score: {score_str} &nbsp;·&nbsp; via: _{stype}_")

            # ── DEBUG EXPANDER ────────────────────────────────────────────────
            # Surfaces raw pipeline internals so we can diagnose ID/search-type
            # mapping failures and Sheets logging issues without guessing.
            with st.expander("🔬 Debug: Raw Pipeline Internals", expanded=False):
                st.caption("These values show exactly what the pipeline returned, before any transformation. Use these to diagnose ID remapping and search type issues.")

                st.markdown("**A. `pipeline_out` top-level keys**")
                # We can't store pipeline_out itself (too large), but we store
                # the raw reranked_df snapshot in session_state for inspection.
                raw_df = st.session_state.get(f"_debug_reranked_df_{selected_query_id}")
                if raw_df is not None:
                    st.markdown(f"Raw `reranked_df` shape: `{raw_df.shape}` — columns: `{list(raw_df.columns)}`")
                    st.dataframe(raw_df, use_container_width=True)
                else:
                    st.info("Raw reranked_df not captured yet — re-run this query to populate debug data. (Debug capture was added in v6.)")

                st.markdown("**B. `retrieved_doc_ids` (after reranked_df_to_list)**")
                st.json(qr.get("retrieved_doc_ids", []))

                st.markdown("**C. `retrieval_search_types` (after reranked_df_to_list)**")
                st.json(qr.get("retrieval_search_types", []))

                st.markdown("**D. `reranker_scores`**")
                st.json(qr.get("reranker_scores", []))

                st.markdown("**E. `ideal_docs_set_used`** (new vs original index auto-detection)")
                st.write(qr.get("ideal_docs_set_used", "not set"))

                st.markdown("**F. `corpus_for_verify` key count** (populated on re-run)")
                st.write(st.session_state.get(f"_debug_corpus_size_{selected_query_id}", "not captured yet"))

                st.markdown("**G. Last Sheets logging error**")
                last_sheets_err = st.session_state.get(f"_debug_sheets_err_{selected_query_id}", "none recorded")
                if last_sheets_err and last_sheets_err != "none recorded":
                    st.error(last_sheets_err)
                else:
                    st.write(last_sheets_err)

                st.markdown("**H. Corpus file fingerprint**")
                fp = st.session_state.get(f"_debug_corpus_fingerprint_{selected_query_id}", None)
                if fp:
                    st.json(fp)
                else:
                    st.write("not captured yet")

                st.markdown("**I. Missing IDs (retrieved vs corpus)**")
                st.write("Missing retrieved IDs:", st.session_state.get(f"_debug_missing_retrieved_{selected_query_id}", []))

                st.markdown("**J. Nicolay cited IDs (and missing vs corpus)**")
                st.write("Cited IDs:", st.session_state.get(f"_debug_cited_ids_{selected_query_id}", []))
                st.write("Missing cited IDs:", st.session_state.get(f"_debug_missing_cited_{selected_query_id}", []))

                st.markdown("**K. Corpus spot-check (retrieved + cited IDs)**")
                spot = st.session_state.get(f"_debug_corpus_spotcheck_{selected_query_id}", [])
                if spot:
                    st.dataframe(pd.DataFrame(spot), use_container_width=True)
                else:
                    st.write("not captured yet")

                st.markdown("**L. ID remap audit (raw → parsed → final)**")
                audit = st.session_state.get(f"_debug_id_audit_{selected_query_id}", [])
                if audit:
                    st.dataframe(pd.DataFrame(audit), use_container_width=True)
                else:
                    st.write("not captured yet")

                st.markdown("**M. Pipeline retry log**")
                attempts = qr.get("pipeline_attempts", 1)
                retry_log = qr.get("pipeline_retry_log", "")
                if attempts and int(attempts) > 1:
                    st.warning(f"⚠️ {attempts} attempt(s) required for this query.")
                else:
                    st.write(f"Attempts: {attempts or 1} (no retries needed)")
                if retry_log:
                    try:
                        st.json(json.loads(retry_log))
                    except Exception:
                        st.code(retry_log)


            # ── Response Confidence Summary (U12) ──────────────────────────────────────────────────
            # Same five-signal panel as the main app.  ROUGE is recomputed here
            # using Key Quote text recovered from Match Analysis (richer than the
            # proxy stored in qresult, which is based on reranker scores alone).
            _conf_rating   = qr.get("confidence_rating", "unknown")
            _conf_icon     = qr.get("confidence_icon", "?")
            _conf_explain  = qr.get("confidence_explanation", "")
            _conf_synth    = qr.get("confidence_synth_type", 3) or 3
            _conf_calib    = qr.get("confidence_calib_warning", False)
            _conf_spread   = qr.get("confidence_spread")
            _conf_nsrc     = qr.get("confidence_n_sources")
            _conf_maxscore = qr.get("confidence_max_score")
            _conf_r1_stored = qr.get("confidence_rouge1")
            _conf_r2_stored = qr.get("confidence_rouge2")

            _qv_list      = qr.get("quote_verification", [])

            # Re-verify if stored results were computed with an empty corpus
            # (common for checkpoint-loaded results from a prior session).
            # If corpus is now available in shared session state and all stored
            # outcomes are "fabrication" or missing, re-run verification.
            # _get_or_load_corpus() handles session_state + on-demand disk load.
            _corpus_for_rev = _get_or_load_corpus(corpus_file)
            _all_fabricated = bool(_qv_list) and all(
                q.get("outcome") in ("fabrication", "too_short") for q in _qv_list
            )
            if _all_fabricated and _corpus_for_rev:
                try:
                    _nicolay_out_rev = qr.get("nicolay_output") or {}
                    # Reconstruct minimal reranked list from stored IDs + scores
                    # (only needed for cited_in_reranked_topk metadata, not outcomes)
                    _stored_ids_rev   = qr.get("retrieved_doc_ids", [])
                    _stored_scores_rev = qr.get("reranker_scores", [])
                    _reranked_rev = [
                        {"text_id_num": _tid, "rank": _rank + 1,
                         "reranker_score": _score}
                        for _rank, (_tid, _score) in enumerate(
                            zip(_stored_ids_rev, _stored_scores_rev))
                        if _tid is not None
                    ]
                    if _nicolay_out_rev:
                        _qv_list = verify_all_quotes(
                            _nicolay_out_rev, _reranked_rev, _corpus_for_rev)
                        qr["quote_verification"] = _qv_list
                        _outcomes_rev = [q.get("outcome") for q in _qv_list]
                        qr["quotes_verified_count"]          = _outcomes_rev.count("verified")
                        qr["quotes_approx_count"]            = _outcomes_rev.count("approximate_quote")
                        qr["quotes_displaced_count"]         = _outcomes_rev.count("displacement")
                        qr["quotes_approx_displaced_count"]  = _outcomes_rev.count("approximate_displacement")
                        qr["quotes_fabricated_count"]        = _outcomes_rev.count("fabrication")
                        qr["quotes_mislabeled_count"]        = _outcomes_rev.count("source_mislabeled")
                        st.caption("ℹ️ Quote verification re-run with corpus now in memory.")
                except Exception:
                    pass

            _verified_q   = [q for q in _qv_list if q.get("outcome") == "verified"]
            _displaced_q  = [q for q in _qv_list if q.get("outcome") in
                              ("displacement", "approximate_displacement")]
            _unverified_q = [q for q in _qv_list if q.get("outcome") == "fabrication"]
            # source_mislabeled: quote text verified but Nicolay's claimed source label
            # doesn't match the actual corpus source (parametric source fabrication).
            # Counts as an integrity failure for overall rating but is distinct from
            # a fabricated quote — the text IS in the corpus.
            _mislabeled_q = [q for q in _qv_list if q.get("outcome") == "source_mislabeled"]

            # Attempt to recompute ROUGE with richer chunk text from Match Analysis
            _final_for_rouge = qr.get("nicolay_final_answer_text", "") or ""
            _stored_scores   = qr.get("reranker_scores", [])[:5]
            _stored_ids      = qr.get("retrieved_doc_ids", [])[:5]
            _ma_for_rouge    = (qr.get("nicolay_output") or {}).get("Match Analysis", {})
            _rp_for_rouge    = []
            for _rs, _rid in zip(_stored_scores, _stored_ids):
                _kq = ""
                if isinstance(_ma_for_rouge, dict):
                    for _mv in _ma_for_rouge.values():
                        if isinstance(_mv, dict) and _extract_int_from_text_id(
                                _mv.get("Text ID", "")) == _rid:
                            _kq = _mv.get("Key Quote", "") or ""
                            break
                # Carry doc_id so _u12_analyze_reranker_scores can count
                # distinct documents when Source strings are unavailable.
                _rp_for_rouge.append({
                    "Relevance Score": _rs,
                    "Source": "",
                    "Key Quote": _kq,
                    "doc_id": _rid,
                })

            _rouge_live = _u12_compute_corpus_grounding(_final_for_rouge, _rp_for_rouge)
            _conf_r1 = (_rouge_live or {}).get("rouge1", _conf_r1_stored)
            _conf_r2 = (_rouge_live or {}).get("rouge2", _conf_r2_stored)

            _conf_label = _conf_rating.capitalize() if _conf_rating else "Unknown"
            with st.expander(
                f"📊 Response Confidence: {_conf_icon} {_conf_label}",
                expanded=False
            ):
                st.markdown(f"**Overall confidence: {_conf_icon} {_conf_label}**")
                if _conf_explain:
                    st.markdown(_conf_explain)
                st.divider()

                # Signal 1: Quote verification
                # _total_q includes all classifiable outcomes: verified, displaced,
                # fabricated, and source-mislabeled. too_short entries are excluded
                # (they have no usable passage and don't affect the integrity score).
                _total_q = len(_verified_q) + len(_displaced_q) + len(_unverified_q) + len(_mislabeled_q)
                _col_qa, _col_qb = st.columns([1, 2])
                with _col_qa:
                    st.caption("**Quote verification**")
                with _col_qb:
                    if _total_q == 0:
                        st.markdown("⬜ No direct quotes in this response")
                        st.caption("Nicolay did not include any directly quoted text.")
                    elif not _unverified_q and not _displaced_q and not _mislabeled_q:
                        st.markdown(f"✅ {_total_q}/{_total_q} quotes verified")
                        st.caption("Every quoted passage confirmed present in the Lincoln corpus.")
                    elif _unverified_q:
                        st.markdown(
                            f"🔴 {len(_verified_q)}/{_total_q} verified — "
                            f"**{len(_unverified_q)} not found in corpus**"
                        )
                        st.caption(
                            "One or more quoted passages could not be located in the Lincoln corpus. "
                            "These may be fabricated, misremembered, or from a source outside the collection."
                        )
                    elif _mislabeled_q:
                        st.markdown(
                            f"🏷️ {len(_verified_q)}/{_total_q} correctly attributed — "
                            f"**{len(_mislabeled_q)} source attribution wrong**"
                        )
                        st.caption(
                            "Quote text confirmed present in the corpus but Nicolay's claimed source "
                            "label does not match the actual document. The text is real; the attribution is wrong."
                        )
                    else:
                        st.markdown(
                            f"🔀 {len(_verified_q)}/{_total_q} verified — "
                            f"**{len(_displaced_q)} displaced**"
                        )
                        st.caption(
                            "Displaced quote appears in the correct document but was drawn "
                            "from a different passage than cited."
                        )
                st.divider()

                # Signal 2: Corpus grounding (ROUGE)
                _col_ra, _col_rb = st.columns([1, 2])
                with _col_ra:
                    st.caption("**Corpus grounding (lexical overlap)**")
                with _col_rb:
                    if _conf_r1 is not None:
                        _conf_r2_str = f"{_conf_r2:.3f}" if _conf_r2 is not None else "n/a"
                        st.markdown(
                            f"ROUGE-1: **{_conf_r1:.3f}** &nbsp;&nbsp; "
                            f"ROUGE-2: **{_conf_r2_str}**",
                            unsafe_allow_html=True,
                        )
                        if _conf_synth in (1, 2):
                            _rouge_lbl = ("✅ Closely tracks retrieved sources" if _conf_r1 >= 0.45
                                          else "⚠️ Low grounding for direct retrieval type" if _conf_r1 < 0.25
                                          else "Moderate grounding — some inference beyond sources")
                        elif _conf_synth == 3:
                            _rouge_lbl = "Partial grounding expected for absence/partial response"
                        else:
                            _rouge_lbl = ("✅ Good synthesis grounding" if _conf_r1 >= 0.30
                                          else "⚠️ Low grounding — verify independently")
                        st.caption(_rouge_lbl)
                        st.caption(
                            "Scores (0.0–1.0) measure word/phrase overlap between FinalAnswer "
                            "and Key Quotes from retrieved passages. Above ~0.45 = closely tracks sources; "
                            "0.25–0.45 = moderate grounding; below 0.25 = significant divergence "
                            "(expected for absence responses; caution flag for direct-retrieval types)."
                        )
                    else:
                        st.markdown("⬜ Not computed (no Key Quote text available)")
                st.divider()

                # Signals 3+4: Retrieval quality
                _col_sa, _col_sb = st.columns([1, 2])
                with _col_sa:
                    st.caption("**Retrieval quality**")
                with _col_sb:
                    # Recompute speech-level source diversity from quote verification results.
                    # cited_chunk_source is stored per qv item; we count distinct speeches.
                    _qv_sources = _count_distinct_sources_from_qv(_qv_list)
                    _display_nsrc = _qv_sources if _qv_sources > 0 else (_conf_nsrc or 0)
                    st.markdown(f"**{_display_nsrc}** distinct Lincoln speech(es) in top-5")
                    if _conf_spread is not None:
                        _spread_lbl = (
                            "Differentiated retrieval ✅" if _conf_spread >= 0.20
                            else "Moderate differentiation" if _conf_spread >= 0.10
                            else "Flat distribution ⚠️ — no passage clearly dominant"
                        )
                        st.markdown(f"Score spread: **{_conf_spread:.3f}** — {_spread_lbl}")
                    if _conf_maxscore is not None:
                        st.caption(f"Max reranker score: {_conf_maxscore:.3f}")
                    if _conf_calib:
                        st.warning(
                            "⚠️ **High confidence, flat spread.** Retrieval returned "
                            "high scores with little discrimination. Corpus may lack ideal documents "
                            "for this query. The response may sound plausible but rest on off-target sources."
                        )
                st.divider()

                # Signal 5: Complexity–type match
                _col_ca, _col_cb = st.columns([1, 2])
                with _col_ca:
                    st.caption("**Response depth**")
                with _col_cb:
                    if _conf_synth in (4, 5):
                        st.markdown("Multi-passage synthesis — type consistent with complex query ✅")
                    else:
                        import re as _re_cp
                        _cpat = _re_cp.compile(
                            r"(compare|contrast|how did .+ change|evolution of|shift|differ|"
                            r"difference|between|throughout|across|over time|consistent|"
                            r"inconsistent|develop|arc|trajectory)",
                            _re_cp.IGNORECASE
                        )
                        _q_text = qr.get("query", "")
                        _est_c  = "high" if _cpat.search(_q_text) else "moderate"
                        if _conf_synth in (1, 2) and _est_c == "high":
                            st.warning(
                                "⚠️ Query appears complex but response is Type 1/2 — "
                                "possible under-synthesis (heuristic estimate)."
                            )
                        else:
                            st.markdown("Query complexity consistent with synthesis type ✅")
                        st.caption(
                            "Heuristic: queries with comparative/synthesis language flagged as "
                            "high-complexity. A mismatch may mean the corpus lacks enough relevant "
                            "material, or rephrasing could surface more evidence."
                        )

                st.divider()
                st.caption(
                    "Signals are automatically computed — not a guarantee of accuracy. "
                    "'Distinct speeches' counts unique Lincoln documents at the speech level "
                    "(multiple retrieved chunks from the same speech count as one). "
                    "Quote verification checks whether direct quotations appear in the Lincoln corpus. "
                    "ROUGE measures lexical overlap with Key Quotes from retrieved passages. "
                    "Always read source passages before relying on specific claims."
                )

            # ── Nicolay output ───────────────────────────────────────────────────────────────────────────
            with st.expander("✍️ Nicolay Output", expanded=True):
                final_text = get_final_answer_text(qr.get("nicolay_output", {}))
                _qv_for_inline = qr.get("quote_verification", [])
                # Reconstruct verified/displaced/unverified/mislabeled lists from stored qv results
                # so we can annotate the FinalAnswer text without re-running verification.
                _iv_verified   = [(q.get("cited_passage",""), q.get("cited_chunk_text_id",""), q.get("cited_chunk_source",""))
                                   for q in _qv_for_inline if q.get("outcome") == "verified"]
                _iv_displaced  = [q.get("cited_passage","")
                                   for q in _qv_for_inline
                                   if q.get("outcome") in ("displacement", "approximate_displacement", "approximate_quote")
                                   and q.get("outcome") != "verified"]
                _iv_unverified = [q.get("cited_passage","")
                                   for q in _qv_for_inline if q.get("outcome") == "fabrication"]
                _iv_mislabeled = [q.get("cited_passage","")
                                   for q in _qv_for_inline if q.get("outcome") == "source_mislabeled"]

                # Annotate FinalAnswer text with inline emoji markers
                _QUOTE_PAIRS_IV = [
                    ('\u201c', '\u201d'),
                    ('"',      '"'     ),
                    ('\u2018', '\u2019'),
                    ("'",      "'"     ),
                ]
                _annotated_fa = final_text
                for _iq, _itid, _isrc in _iv_verified:
                    if not _iq: continue
                    for _oq, _cq in _QUOTE_PAIRS_IV:
                        _lit = _oq + _iq + _cq
                        if _lit in _annotated_fa:
                            _annotated_fa = _annotated_fa.replace(_lit, _lit + " ✅", 1)
                            break
                for _iq in _iv_displaced:
                    if not _iq: continue
                    for _oq, _cq in _QUOTE_PAIRS_IV:
                        _lit = _oq + _iq + _cq
                        if _lit in _annotated_fa:
                            _annotated_fa = _annotated_fa.replace(_lit, _lit + " 🔀", 1)
                            break
                for _iq in _iv_unverified:
                    if not _iq: continue
                    for _oq, _cq in _QUOTE_PAIRS_IV:
                        _lit = _oq + _iq + _cq
                        if _lit in _annotated_fa:
                            _annotated_fa = _annotated_fa.replace(_lit, _lit + " ⚠️", 1)
                            break
                for _iq in _iv_mislabeled:
                    if not _iq: continue
                    for _oq, _cq in _QUOTE_PAIRS_IV:
                        _lit = _oq + _iq + _cq
                        if _lit in _annotated_fa:
                            _annotated_fa = _annotated_fa.replace(_lit, _lit + " 🏷️", 1)
                            break

                st.markdown("**Final Answer:**")
                st.markdown(_annotated_fa)
                # Legend
                _iv_has_v = bool(_iv_verified)
                _iv_has_d = bool(_iv_displaced)
                _iv_has_u = bool(_iv_unverified)
                _iv_has_m = bool(_iv_mislabeled)
                if _iv_has_v or _iv_has_d or _iv_has_u or _iv_has_m:
                    _iv_legend = []
                    if _iv_has_v: _iv_legend.append("✅ verified against corpus")
                    if _iv_has_d: _iv_legend.append("🔀 found in document, displaced chunk")
                    if _iv_has_u: _iv_legend.append("⚠️ not found — possible fabrication")
                    if _iv_has_m: _iv_legend.append("🏷️ text verified but source attribution wrong")
                    st.caption(" · ".join(_iv_legend))
                st.caption(f"Word count: {qr.get('nicolay_final_answer_wordcount', 0)}")

            # Quote verification
            with st.expander("🔎 Quote Verification", expanded=False):
                for qv_item in qr.get("quote_verification", []):
                    icon = {"verified": "✅", "approximate_quote": "🟡", "displacement": "⚠️", "approximate_displacement": "🟠", "fabrication": "🚨", "source_mislabeled": "🏷️", "too_short": "—"}.get(qv_item.get("outcome", ""), "?")
                    st.markdown(f"{icon} **{qv_item.get('match', '')}** ({qv_item.get('text_id', '')}) — *{qv_item.get('outcome', '')}*")
                    if qv_item.get("cited_passage"):
                        st.caption(f"\"{str(qv_item['cited_passage'])[:150]}...\"")
                    # Debug line: did we even find the cited chunk in the verifier corpus?
                    st.caption(
                        f"cited_num={qv_item.get('cited_num')} | cited_chunk_present={qv_item.get('cited_chunk_present')} | "
                        f"cited_source={qv_item.get('cited_chunk_source','')[:80]}"
                    )
                    if qv_item.get("cited_chunk_preview"):
                        st.caption(f"cited_chunk_preview: {qv_item.get('cited_chunk_preview')}")
                    if qv_item.get("match_method") or qv_item.get("approx_score") is not None:
                        _ms = qv_item.get("match_method", "")
                        _sc = qv_item.get("approx_score", None)
                        st.caption(f"match_method={_ms}" + (f" | approx_score={_sc:.2f}" if isinstance(_sc, (int, float)) else ""))
                    if qv_item.get("match_chunk_num") is not None:
                        st.caption(f"found_at_chunk={qv_item.get('match_chunk_num')} | found_source={qv_item.get('match_chunk_source','')[:80]}")
                    if qv_item.get("note"):
                        st.caption(qv_item["note"])


            # ── Match Analysis cards ─────────────────────────────────────────────────────────────────────
            # Displays reranker score badge + key quote blockquote + Nicolay's analysis +
            # verification badge + collapsible full-text with highlight.
            # Preserves the existing Quote Verification debug expander above.
            _ma_cards = (qr.get("nicolay_output") or {}).get("Match Analysis", {})
            if _ma_cards and isinstance(_ma_cards, dict):
                st.markdown("---")
                st.markdown("#### 🎯 Match Analysis")
                _ma_scores_map = {}
                _ma_retrieved  = qr.get("retrieved_doc_ids", [])
                _ma_rscores    = qr.get("reranker_scores", [])
                for _ma_tid, _ma_sc in zip(_ma_retrieved, _ma_rscores):
                    _ma_scores_map[str(_ma_tid)] = _ma_sc

                _ma_col_l, _ma_col_r = st.columns(2)
                _ma_col_map = {0: _ma_col_l, 1: _ma_col_r}

                for _ma_i, (_ma_key, _ma_info) in enumerate(_ma_cards.items()):
                    if not isinstance(_ma_info, dict):
                        continue
                    _ma_text_id   = str(_ma_info.get("Text ID", ""))
                    _ma_src       = (_ma_info.get("Source", "") or "").strip()
                    if _ma_src.lower().startswith("source:"):
                        _ma_src = _ma_src[len("source:"):].strip()
                    _ma_kq        = _ma_info.get("Key Quote", "")
                    _ma_summary   = _ma_info.get("Summary", "")
                    _ma_relevance = _ma_info.get("Relevance Assessment", "")
                    _ma_hist      = _ma_info.get("Historical Context", "")
                    _ma_score     = _ma_scores_map.get(_ma_text_id)

                    # Relevance badge
                    _ma_rel_t = (_ma_relevance or "").lower()
                    if "high" in _ma_rel_t:   _ma_rbg, _ma_rfg = "#d4edda", "#155724"
                    elif "medium" in _ma_rel_t or "moderate" in _ma_rel_t: _ma_rbg, _ma_rfg = "#fff3cd", "#856404"
                    elif "low" in _ma_rel_t:  _ma_rbg, _ma_rfg = "#f8d7da", "#721c24"
                    else:                      _ma_rbg, _ma_rfg = "#e2e3e5", "#383d41"
                    import re as _ma_re
                    _ma_rel_label = _ma_re.split(r"[—,]", _ma_relevance or "N/A")[0].strip()[:30]
                    _ma_rel_badge = (
                        f'<span style="background:{_ma_rbg};color:{_ma_rfg};padding:4px 12px;'
                        f'border-radius:12px;font-size:0.85em;font-weight:700;'
                        f'display:inline-block;">{_ma_rel_label}</span>'
                    )

                    # Verification badge from stored qv results
                    # Use integer comparison for cited_num (stored as int); _ma_text_id
                    # may be "Text #: 311" or "311" — extract int for reliable matching.
                    _ma_tid_int = _extract_int_from_text_id(_ma_text_id)
                    _ma_qv_item = next(
                        (q for q in qr.get("quote_verification", [])
                         if (_ma_tid_int is not None and q.get("cited_num") == _ma_tid_int)
                         or q.get("match", "") == _ma_key),
                        None
                    )
                    _ma_outcome = (_ma_qv_item or {}).get("outcome", "")
                    if _ma_outcome == "verified":
                        _ma_vbadge = '<span style="color:#155724;font-size:0.9em;font-weight:600;">✅ Quote verified in corpus</span>'
                    elif _ma_outcome in ("displacement", "approximate_displacement"):
                        _ma_vbadge = '<span style="color:#664d03;background:#fff3cd;padding:1px 6px;border-radius:4px;font-size:0.9em;font-weight:600;">🔀 Found in document, displaced chunk</span>'
                    elif _ma_outcome == "approximate_quote":
                        _ma_vbadge = '<span style="color:#155724;font-size:0.9em;font-weight:500;">🟡 Approximate match confirmed</span>'
                    elif _ma_outcome == "fabrication":
                        _ma_vbadge = '<span style="color:#721c24;font-size:0.9em;font-weight:600;">⚠️ Not found in corpus</span>'
                    else:
                        _ma_vbadge = '<span style="color:#6c757d;font-size:0.9em;">⬜ Not verified</span>'

                    with _ma_col_map[_ma_i % 2]:
                        with st.container(border=True):
                            # Header: title left, relevance badge right
                            _ma_h1, _ma_h2 = st.columns([5, 1])
                            with _ma_h1:
                                st.markdown(f"**{_ma_key}**")
                            with _ma_h2:
                                st.markdown(f'<div style="text-align:right;">{_ma_rel_badge}</div>',
                                            unsafe_allow_html=True)

                            # Metadata + reranker score
                            st.markdown(f"**ID:** {_ma_text_id}  ·  **Source:** {_ma_src}")
                            if _ma_score is not None:
                                st.caption(f"Reranker score: {_ma_score:.3f}")

                            # Key quote blockquote
                            if _ma_kq:
                                _ma_kq_disp = _ma_kq[:320] + ("…" if len(_ma_kq) > 320 else "")
                                st.markdown(f'> *“{_ma_kq_disp}”*')

                            # Nicolay's analysis
                            if _ma_summary:
                                _ma_sum_disp = _ma_summary[:300] + ("…" if len(_ma_summary) > 300 else "")
                                st.markdown(f"**Nicolay's Analysis:** {_ma_sum_disp}")

                            # Verification badge
                            st.markdown(_ma_vbadge, unsafe_allow_html=True)

                            # Full text expander with highlight
                            with st.expander(f"Full text & highlight — {_ma_key}", expanded=False):
                                if _ma_hist:
                                    st.markdown(f"**Historical Context:** {_ma_hist}")
                                # _get_or_load_corpus() loads on demand if not already cached —
                                # no prior pipeline run required.
                                _ma_corpus_ss = _get_or_load_corpus(corpus_file)
                                # Use _extract_int_from_text_id() rather than bare int()
                                # so both "311" and "Text #: 311" formats resolve correctly.
                                _ma_int_id = _extract_int_from_text_id(_ma_text_id)
                                _ma_chunk = _ma_corpus_ss.get(_ma_int_id) if _ma_int_id is not None else None
                                if _ma_chunk:
                                    _ma_ft = _ma_chunk.get("full_text", "")
                                    if _ma_ft:
                                        _ma_html = _ma_ft.replace("\n", "<br>")
                                        if _ma_kq and _ma_kq in _ma_ft:
                                            _ma_html = _ma_html.replace(_ma_kq, f"<mark>{_ma_kq}</mark>")
                                        elif _ma_kq:
                                            # Partial highlight: find first segment
                                            _ma_segs = [p.strip() for p in _ma_kq.split("...") if p.strip()]
                                            for _seg in _ma_segs:
                                                if len(_seg) > 20 and _seg in _ma_ft:
                                                    _ma_html = _ma_html.replace(_seg, f"<mark>{_seg}</mark>", 1)
                                                    break
                                        st.markdown("**Full Text with Highlighted Quote:**")
                                        st.markdown(
                                            f'<div style="font-size:0.95em;line-height:1.65;">{_ma_html}</div>',
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        st.info("Full text not available for this chunk.")
                                elif not _ma_corpus_ss:
                                    st.warning(
                                        f"⚠️ Corpus not loaded — "
                                        f"`{corpus_file}` not found on disk. "
                                        f"Check path in sidebar debug expander."
                                    )
                                else:
                                    st.info(f"Chunk {_ma_text_id} not found in corpus ({len(_ma_corpus_ss)} chunks loaded).")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2: METRICS
    # ═══════════════════════════════════════════════════════════════════════
    with tab_metrics:
        if selected_query_id not in results.get("queries", {}):
            st.info("Run this query first to see metrics.")
        else:
            qr = results["queries"][selected_query_id]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                p = qr.get("precision_at_5", 0)
                color = "pass" if p >= 0.4 else "warn" if p > 0 else "fail"
                st.markdown(f'<div class="metric-card"><b>Precision@5</b><br><span class="{color}">{p:.4f}</span></div>', unsafe_allow_html=True)
            with col2:
                r = qr.get("recall_at_5", 0)
                color = "pass" if r >= 0.5 else "warn" if r > 0 else "fail"
                st.markdown(f'<div class="metric-card"><b>Recall@5</b><br><span class="{color}">{r:.4f}</span></div>', unsafe_allow_html=True)
            with col3:
                cap = qr.get("ceiling_adjusted_precision", 0)
                color = "pass" if cap >= 0.7 else "warn" if cap > 0 else "fail"
                st.markdown(f'<div class="metric-card"><b>Ceiling-Adj P</b><br><span class="{color}">{cap:.4f}</span></div>', unsafe_allow_html=True)
            with col4:
                wc = qr.get("nicolay_final_answer_wordcount", 0)
                expected_t = qr.get("expected_nicolay_type", "")
                min_words = {"T1": 50, "T2": 100, "T3": 150, "T4": 200, "T5": 300}.get(expected_t, 0)
                color = "pass" if wc >= min_words else "warn"
                st.markdown(f'<div class="metric-card"><b>FinalAnswer Words</b><br><span class="{color}">{wc}</span> <small>(≥{min_words} for {expected_t})</small></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header">Task Type Classification</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                hay_raw = qr.get("hay_task_type_raw", "—")
                hay_correct = qr.get("hay_task_type_correct")
                expected_hay = qr.get("expected_hay_type", "")
                icon = "✅" if hay_correct else ("❌" if hay_correct is False else "?")
                st.markdown(f"**Hay type:** {icon} Got `{hay_raw}`, expected `{expected_hay}`")
            with col2:
                nic_raw = qr.get("nicolay_synthesis_type_raw", "—")
                nic_correct = qr.get("nicolay_synthesis_type_correct")
                expected_nic = qr.get("expected_nicolay_type", "")
                icon = "✅" if nic_correct else ("❌" if nic_correct is False else "?")
                st.markdown(f"**Nicolay type:** {icon} Got `{nic_raw}`, expected `{expected_nic}`")

            st.markdown('<div class="section-header">Schema & Quote Verification</div>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                sc = qr.get("nicolay_schema_complete", False)
                st.metric("Schema Complete", "✅" if sc else "❌")
            with col2:
                st.metric("Quotes Verified", qr.get("quotes_verified_count", 0))
            with col3:
                disp = qr.get("quotes_displaced_count", 0)
                st.metric("Displaced", disp, delta=f"{disp}" if disp > 0 else None, delta_color="inverse")
            with col4:
                fab = qr.get("quotes_fabricated_count", 0)
                st.metric("Fabricated", fab, delta=f"{fab}" if fab > 0 else None, delta_color="inverse")

            # Watchlist
            if qdef.get("watchlist"):
                st.markdown('<div class="section-header">Watchlist Observations</div>', unsafe_allow_html=True)
                for w in qdef["watchlist"]:
                    st.markdown(f"- {w}")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3: RUBRIC (manual scoring)
    # ═══════════════════════════════════════════════════════════════════════
    with tab_rubric:
        if selected_query_id not in results.get("queries", {}):
            st.info("Run this query first to score it.")
        else:
            qr = results["queries"][selected_query_id]
            st.markdown(f"**Manual Qualitative Rubric — {selected_query_id}**")
            st.caption("Score each criterion 0–1 (0 = absent/poor, 0.5 = partial, 1 = fully present/strong)")

            col1, col2 = st.columns(2)
            with col1:
                fa = st.select_slider(
                    "Factual Accuracy",
                    options=[0.0, 0.25, 0.5, 0.75, 1.0],
                    value=float(qr.get("rubric_factual_accuracy") or 0.0),
                    help="Are the factual claims in the FinalAnswer verifiable against the corpus?"
                )
                ca = st.select_slider(
                    "Citation Accuracy",
                    options=[0.0, 0.25, 0.5, 0.75, 1.0],
                    value=float(qr.get("rubric_citation_accuracy") or 0.0),
                    help="Do inline citations correctly match the passages being cited?"
                )
            with col2:
                hd = st.select_slider(
                    "Historiographical Depth",
                    options=[0.0, 0.25, 0.5, 0.75, 1.0],
                    value=float(qr.get("rubric_historiographical_depth") or 0.0),
                    help="Does the response demonstrate appropriate historical reasoning and contextualization?"
                )
                ec = st.select_slider(
                    "Epistemic Calibration",
                    options=[0.0, 0.25, 0.5, 0.75, 1.0],
                    value=float(qr.get("rubric_epistemic_calibration") or 0.0),
                    help="Does Nicolay flag gaps, limitations, or missing evidence appropriately?"
                )

            total = round(fa + ca + hd + ec, 2)
            color = "pass" if total >= 3.0 else "warn" if total >= 2.0 else "fail"
            st.markdown(f'<div class="metric-card" style="margin-top:1rem;"><b>Rubric Total</b>: <span class="{color}">{total} / 4.0</span></div>', unsafe_allow_html=True)

            notes = st.text_area(
                "Evaluator notes (watchlist observations, notable failures, CoT quality):",
                value=qr.get("evaluator_notes", ""),
                height=150
            )

            if st.button("💾 Save Rubric Scores"):
                qr["rubric_factual_accuracy"] = fa
                qr["rubric_citation_accuracy"] = ca
                qr["rubric_historiographical_depth"] = hd
                qr["rubric_epistemic_calibration"] = ec
                qr["rubric_total"] = total
                qr["evaluator_notes"] = notes
                results["queries"][selected_query_id] = qr
                save_results(results)
                st.session_state.results = results
                # Update Sheets row in place
                update_rubric_in_sheets(benchmark_logger, gc_client, selected_query_id, {
                    "RubricFactualAccuracy": fa, "RubricCitationAccuracy": ca,
                    "RubricHistoriographicalDepth": hd, "RubricEpistemicCalibration": ec,
                    "RubricTotal": total, "EvaluatorNotes": notes
                })
                st.success("✅ Rubric scores saved.")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 4: SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    with tab_summary:
        completed = results.get("queries", {})
        st.markdown(f"**{len(completed)} / 25 queries completed**")

        if completed:
            rows = []
            for qid, qd in completed.items():
                rows.append({
                    "ID": qid,
                    "Category": qd.get("category", "")[:20],
                    "Hay✓": "✅" if qd.get("hay_task_type_correct") else ("❌" if qd.get("hay_task_type_correct") is False else "?"),
                    "P@5": f"{qd.get('precision_at_5', 0):.2f}",
                    "R@5": f"{qd.get('recall_at_5', 0):.2f}",
                    "Cap-P": f"{qd.get('ceiling_adjusted_precision', 0):.2f}",
                    "Nic✓": "✅" if qd.get("nicolay_synthesis_type_correct") else ("❌" if qd.get("nicolay_synthesis_type_correct") is False else "?"),
                    "Schema": "✅" if qd.get("nicolay_schema_complete") else "❌",
                    "Words": qd.get("nicolay_final_answer_wordcount", ""),
                    "Verified": qd.get("quotes_verified_count", ""),
                    "Approx": qd.get("quotes_approx_count", ""),
                    "Displaced": qd.get("quotes_displaced_count", ""),
                    "ApproxDisp": qd.get("quotes_approx_displaced_count", ""),
                    "Fabr.": qd.get("quotes_fabricated_count", ""),
                    "Rubric": qd.get("rubric_total", "—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Category performance
            st.markdown('<div class="section-header">Category Performance</div>', unsafe_allow_html=True)
            cats = {}
            for qid, qd in completed.items():
                cat = qd.get("category", "unknown")
                if cat not in cats:
                    cats[cat] = {"p5": [], "r5": [], "rubric": []}
                cats[cat]["p5"].append(qd.get("precision_at_5", 0) or 0)
                cats[cat]["r5"].append(qd.get("recall_at_5", 0) or 0)
                if qd.get("rubric_total") is not None:
                    cats[cat]["rubric"].append(qd["rubric_total"])

            cat_rows = []
            for cat, vals in cats.items():
                cat_rows.append({
                    "Category": cat,
                    "N": len(vals["p5"]),
                    "Avg P@5": f"{sum(vals['p5'])/len(vals['p5']):.2f}",
                    "Avg R@5": f"{sum(vals['r5'])/len(vals['r5']):.2f}",
                    "Avg Rubric": f"{sum(vals['rubric'])/len(vals['rubric']):.2f}" if vals['rubric'] else "—",
                })
            st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No queries run yet.")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 5: BLEU/ROUGE
    # ═══════════════════════════════════════════════════════════════════════
    with tab_bleu:
        st.markdown("### BLEU / ROUGE Scores")
        st.caption(
            "Two reference sets: **retrieved** (top-5 reranked chunks — measures synthesis faithfulness) "
            "and **ideal** (ground-truth chunks — measures alignment with what should have been used). "
            "These are lexical overlap metrics; they complement but do not replace the qualitative rubric."
        )

        if selected_query_id not in results.get("queries", {}):
            st.info("Run this query first.")
        else:
            qr = results["queries"][selected_query_id]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**BLEU (vs retrieved)**")
                st.metric("Max", f"{qr.get('bleu_max_retrieved', 0):.4f}")
                st.metric("Avg", f"{qr.get('bleu_avg_retrieved', 0):.4f}")
            with col2:
                st.markdown("**ROUGE-1**")
                st.metric("Max (retrieved)", f"{qr.get('rouge1_max_retrieved', 0):.4f}")
                st.metric("Max (ideal)", f"{qr.get('rouge1_max_ideal', 0):.4f}")
                ratio = qr.get("rouge1_retrieved_vs_ideal_ratio")
                if ratio is not None:
                    color = "normal" if 0.7 <= ratio <= 1.3 else "inverse"
                    st.metric("Retrieved/Ideal Ratio", f"{ratio:.3f}", delta_color=color)
            with col3:
                st.markdown("**ROUGE-L**")
                st.metric("Max (retrieved)", f"{qr.get('rougeL_max_retrieved', 0):.4f}")
                st.metric("Max (ideal)", f"{qr.get('rougeL_max_ideal', 0):.4f}")

            st.caption(qr.get("bleu_rouge_note", ""))

        # Aggregate BLEU/ROUGE table
        if results.get("queries"):
            st.markdown('<div class="section-header">Aggregate BLEU/ROUGE</div>', unsafe_allow_html=True)
            br_rows = []
            for qid, qd in results["queries"].items():
                br_rows.append({
                    "ID": qid, "Category": qd.get("category", "")[:18],
                    "BLEU-max-retr": f"{qd.get('bleu_max_retrieved', 0) or 0:.4f}",
                    "R1-max-retr": f"{qd.get('rouge1_max_retrieved', 0) or 0:.4f}",
                    "R1-max-ideal": f"{qd.get('rouge1_max_ideal', 0) or 0:.4f}",
                    "R1 ratio": f"{qd.get('rouge1_retrieved_vs_ideal_ratio') or '—'}",
                    "RL-max-retr": f"{qd.get('rougeL_max_retrieved', 0) or 0:.4f}",
                    "RL-max-ideal": f"{qd.get('rougeL_max_ideal', 0) or 0:.4f}",
                })
            st.dataframe(pd.DataFrame(br_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
