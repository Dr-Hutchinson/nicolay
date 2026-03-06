"""
Nicolay Formal Benchmark — Streamlit App
=========================================
Runs the 13-query (+3 replacement) formal benchmark against the Nicolay RAG pipeline.
Captures all defined metrics: Hay-layer, retrieval-layer, Nicolay-layer, quote verification,
BLEU/ROUGE NLP scores, and a manual qualitative rubric.

System state: Hay v3 + Nicolay v3 + Cohere rerank-v4.0-pro + full chunk text + k=5
Corpus: lincoln_speech_corpus_repaired_1.json (772 chunks)
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
# MODEL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

HAY_MODEL    = "ft:gpt-4.1-mini-2025-04-14:personal:hays-v3:DEcb9s4u"
NICOLAY_MODEL = "ft:gpt-4.1-mini-2025-04-14:personal:nicolay-v3:DEccNnWt"
COHERE_RERANK_MODEL = "rerank-v4.0-pro"
RERANK_K = 5

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK QUERY LIST
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_QUERIES = [
    # ── Core 13 ──────────────────────────────────────────────────────────────
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
        "id": "Q4", "group": "core",
        "query": "How did Lincoln incorporate allusions in his Second Inaugural Address?",
        "category": "analysis",
        "expected_hay_type": "A", "expected_nicolay_type": "T2",
        "ideal_docs_new": [419, 420, 421, 422], "ideal_docs_original": [77, 78], "ideal_docs_count": 4,
        "critical_missing_evidence": None,
        "watchlist": ["Biblical allusion depth in chunks 421-422"],
    },
    {
        "id": "Q5", "group": "core",
        "query": "How did Lincoln characterize the implications of major Supreme Court decisions before the Civil War?",
        "category": "analysis",
        "expected_hay_type": "D", "expected_nicolay_type": "T2",
        "ideal_docs_new": [88, 95, 101], "ideal_docs_original": [15, 16, 17], "ideal_docs_count": 3,
        "critical_missing_evidence": None,
        "watchlist": [],
    },
    {
        "id": "Q6", "group": "core",
        "query": "How did Lincoln explain his administration's approach to the Fugitive Slave Law?",
        "category": "analysis",
        "expected_hay_type": "A", "expected_nicolay_type": "T1",
        "ideal_docs_new": [185, 191, 197, 202], "ideal_docs_original": [33, 34, 35, 36], "ideal_docs_count": 4,
        "critical_missing_evidence": None,
        "watchlist": [],
    },
    {
        "id": "Q7", "group": "core",
        "query": "How did Lincoln's discussion of slavery evolve between his House Divided speech and his Second Inaugural Address?",
        "category": "comparative_analysis",
        "expected_hay_type": "E", "expected_nicolay_type": "T4",
        "ideal_docs_new": [88, 95, 101, 419, 420, 421, 422], "ideal_docs_original": [15, 16, 17, 77, 78], "ideal_docs_count": 7,
        "critical_missing_evidence": None,
        "watchlist": ["Lincoln-Douglas Debate chunk retrieval", "Hay Contrastive over-classification"],
    },
    {
        "id": "Q8", "group": "core",
        "query": "How did Lincoln's justification for the Civil War evolve between his First Inaugural and Second Inaugural?",
        "category": "comparative_analysis",
        "expected_hay_type": "E", "expected_nicolay_type": "T4",
        "ideal_docs_new": [185, 191, 197, 202, 419, 420, 421, 422], "ideal_docs_original": [33, 34, 35, 36, 77, 78], "ideal_docs_count": 8,
        "critical_missing_evidence": None,
        "watchlist": ["Hay Contrastive over-classification"],
    },
    {
        "id": "Q9", "group": "core",
        "query": "How did Lincoln's views of African American soldiers change or remain the same over time?",
        "category": "comparative_analysis",
        "expected_hay_type": "D", "expected_nicolay_type": "T3",
        "ideal_docs_new": [288, 295, 367, 374], "ideal_docs_original": [51, 52, 65, 66], "ideal_docs_count": 4,
        "critical_missing_evidence": None,
        "watchlist": ["295 soldiers passage", "374 100,000 in service"],
    },
    {
        "id": "Q10", "group": "core",
        "query": "How did Lincoln develop the theme of divine providence throughout his wartime speeches?",
        "category": "synthesis",
        "expected_hay_type": "D", "expected_nicolay_type": "T4",
        "ideal_docs_new": [298, 418, 419, 420, 421, 422], "ideal_docs_original": [53, 76, 77, 78], "ideal_docs_count": 6,
        "critical_missing_evidence": None,
        "watchlist": ["Chunk 298 Second Annual opening providence language"],
    },
    {
        "id": "Q11", "group": "core",
        "query": "How did Lincoln consistently frame the relationship between liberty and law?",
        "category": "synthesis",
        "expected_hay_type": "D", "expected_nicolay_type": "T4",
        "ideal_docs_new": [153, 159, 185, 191, 418, 419], "ideal_docs_original": [27, 28, 33, 34, 76, 77], "ideal_docs_count": 6,
        "critical_missing_evidence": None,
        "watchlist": [],
    },
    {
        "id": "Q12", "group": "core",
        "query": "What themes did Lincoln consistently employ when discussing the Constitution's relationship to slavery?",
        "category": "synthesis",
        "expected_hay_type": "D", "expected_nicolay_type": "T4",
        "ideal_docs_new": [153, 159, 185, 191], "ideal_docs_original": [27, 28, 33, 34], "ideal_docs_count": 4,
        "critical_missing_evidence": None,
        "watchlist": ["Lincoln-Douglas Debate chunk retrieval", "Cooper Union additional chunks"],
    },
    {
        "id": "Q13", "group": "core",
        "query": "How did Lincoln's views on African American citizenship and racial equality evolve across his speeches?",
        "category": "race_citizenship",
        "expected_hay_type": "E", "expected_nicolay_type": "T5",
        "ideal_docs_new": [288, 295, 367, 374, 413, 414, 419], "ideal_docs_original": [51, 52, 65, 66, 77, 78], "ideal_docs_count": 7,
        "critical_missing_evidence": "Last Public Address (Apr 11, 1865) — conditional suffrage statement NOT IN CORPUS",
        "watchlist": [
            "Jonesboro chunks 517-518 retrieval (racial hierarchy statement)",
            "Charleston debate chunks 572-614 retrieval",
            "Historiographical nuance: does Nicolay handle limiting statements appropriately?",
        ],
    },
    # ── Replacements ──────────────────────────────────────────────────────────
    {
        "id": "R1", "group": "replacement",
        "query": "How did Lincoln justify the naval blockade of Confederate ports?",
        "category": "analysis",
        "expected_hay_type": "D", "expected_nicolay_type": "T3",
        "ideal_docs_new": [218, 272, 300, 345, 359], "ideal_docs_original": [39, 48, 53, 61, 63], "ideal_docs_count": 5,
        "critical_missing_evidence": "Trent Affair entirely absent from corpus — Mason, Slidell, San Jacinto references NOT IN CORPUS",
        "watchlist": ["Trent Affair gap recognition by Nicolay"],
    },
    {
        "id": "R2", "group": "replacement",
        "query": "How did Lincoln describe U.S. relations with Great Britain during the Civil War?",
        "category": "comparative_analysis",
        "expected_hay_type": "D", "expected_nicolay_type": "T3",
        "ideal_docs_new": [242, 243, 247, 300, 301, 345, 346, 388], "ideal_docs_original": [43, 44, 53, 54, 61, 62, 69], "ideal_docs_count": 8,
        "critical_missing_evidence": "Trent Affair absent (most diplomatically significant U.S.-British episode of the war)",
        "watchlist": ["Trent Affair gap recognition by Nicolay"],
    },
    {
        "id": "R3", "group": "replacement",
        "query": "How did Lincoln report on the financial condition of the Post Office Department during the war?",
        "category": "factual_retrieval",
        "expected_hay_type": "D", "expected_nicolay_type": "T2",
        "ideal_docs_new": [311, 312, 364, 365, 401], "ideal_docs_original": [55, 56, 64, 65, 71], "ideal_docs_count": 5,
        "critical_missing_evidence": None,
        "watchlist": ["Numerical progression: $8.3M → near-self-sustaining → $12.4M"],
    },
]

QUERY_IDS = [q["id"] for q in BENCHMARK_QUERIES]
QUERY_BY_ID = {q["id"]: q for q in BENCHMARK_QUERIES}

# ─────────────────────────────────────────────────────────────────────────────
# CORPUS LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_corpus(path: str) -> dict:
    """Load lincoln_speech_corpus_repaired_1.json → {int(text_id): chunk_dict}."""
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
    - Otherwise use ideal_docs (772-chunk new IDs)
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
# QUOTE VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────


def normalize_for_quote_matching(text: str) -> str:
    """
    Normalize text for robust quote matching.

    Key goals:
    - Make matching resilient to unicode punctuation and dash variants
    - Ignore editorial brackets
    - Collapse whitespace
    - **Do NOT** keep ellipsis as literal characters (models often include "..." or "…" to indicate truncation)
    """
    if not text:
        return ""
    # Unicode normalization
    text = unicodedata.normalize("NFKD", str(text))

    # Normalize common quote glyphs
    text = text.replace("`", "'")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # Ellipsis variants -> sentinel (we will strip in segmenting, but also remove here for plain substring matches)
    text = text.replace("…", "...")

    # Dashes: collapse multi-dash runs and single dashes to spaces
    text = re.sub(r'[-\u2013\u2014]{2,}', ' ', text)
    text = re.sub(r'[-\u2013\u2014]', ' ', text)

    # Editorial brackets
    text = re.sub(r'\[.*?\]', '', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text.lower()

# Loose normalization for "approximate_quote" detection (tolerant to punctuation/quote styling)
_STOPWORDS = {
    "the","a","an","and","or","but","if","then","than","to","of","in","on","for","with","by","at","as",
    "is","are","was","were","be","been","being","it","its","this","that","these","those","we","you","i",
    "he","she","they","them","his","her","their","our","us","my","your","not","no","so","do","does","did",
    "from","into","over","under","up","down","out","about","because","which","who","whom","what","when","where","why","how"
}

def normalize_for_quote_matching_loose(text: str) -> str:
    """More forgiving normalization: remove most punctuation so minor quote styling doesn't cause false 'fabrication'."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))

    # Normalize quote glyphs and ellipsis
    text = text.replace("`", "'")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("…", "...")

    # Dashes -> spaces (same idea as strict)
    text = re.sub(r'[-\u2013\u2014]{2,}', ' ', text)
    text = re.sub(r'[-\u2013\u2014]', ' ', text)

    # Editorial brackets
    text = re.sub(r'\[.*?\]', '', text)

    # Drop punctuation entirely (keep letters/numbers/spaces)
    text = re.sub(r"[^0-9A-Za-z\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text.lower()

def _quote_segments_for_matching_loose(passage: str) -> list[str]:
    """Segments-in-order matching, but under loose normalization."""
    if not passage:
        return []
    p = unicodedata.normalize("NFKD", str(passage)).strip()
    # Strip common surrounding quotes (ascii)
    p = p.strip().strip('"').strip("'").strip()
    parts = re.split(r'(?:\.\.\.|…)', p)
    segs = []
    for part in parts:
        s = normalize_for_quote_matching_loose(part)
        if len(s) >= 18:
            segs.append(s)
    if not segs:
        p2 = re.sub(r'(?:\.\.\.|…)+', ' ', p)
        s2 = normalize_for_quote_matching_loose(p2)
        if s2:
            segs = [s2]
    return segs

def _content_tokens(norm_text: str) -> list[str]:
    """Tokenize and drop common stopwords; used for approximate matching heuristics."""
    if not norm_text:
        return []
    toks = [t for t in norm_text.split() if len(t) >= 3 and t not in _STOPWORDS]
    return toks

def _token_coverage(quote_norm_loose: str, chunk_norm_loose: str) -> float:
    """Fraction of (content) quote tokens present in chunk tokens."""
    q = set(_content_tokens(quote_norm_loose))
    if len(q) < 6:
        return 0.0
    c = set(_content_tokens(chunk_norm_loose))
    if not c:
        return 0.0
    return len(q & c) / max(1, len(q))


def _quote_segments_for_matching(passage: str) -> list[str]:
    """
    Convert a possibly-truncated "quote" (often containing .../…) into
    a list of normalized segments that should appear **in order**.

    This solves the exact bug you hit in Q8:
    Nicolay's Key Quote often ends with "..." which is not present in the corpus,
    causing false "fabrication" even when the quote is present verbatim.
    """
    if not passage:
        return []

    # Normalize first (includes converting … -> ...)
    p = unicodedata.normalize("NFKD", str(passage)).strip()

    # Strip common surrounding quotation marks
    p = p.strip().strip('"').strip("'").strip()

    # Split on ellipsis markers (internal or trailing)
    parts = re.split(r'(?:\.\.\.|…)', p)

    # Normalize each part and keep non-trivial segments
    segs = []
    for part in parts:
        s = normalize_for_quote_matching(part)
        # Skip tiny fragments (e.g., a single cutoff word)
        if len(s) >= 10:
            segs.append(s)

    # If everything was tiny (rare), fall back to the whole normalized string with ellipses removed
    if not segs:
        p2 = re.sub(r'(?:\.\.\.|…)+', ' ', p)
        s2 = normalize_for_quote_matching(p2)
        if s2:
            segs = [s2]

    return segs


def _contains_segments_in_order(haystack_norm: str, segs_norm: list[str]) -> bool:
    """Return True iff all segments occur in haystack in order."""
    if not haystack_norm or not segs_norm:
        return False
    i = 0
    for s in segs_norm:
        pos = haystack_norm.find(s, i)
        if pos == -1:
            return False
        i = pos + len(s)
    return True



def verify_quote(cited_passage: str, cited_chunk: dict, corpus: dict) -> dict:
    """
    Quote verification with 5 outcomes:

    - verified:               match in cited chunk (strict OR punctuation-insensitive normalization)
    - approximate_quote:      likely paraphrase: high token overlap without an exact segments-in-order match
    - displacement:           match found in a different chunk (strict OR punctuation-insensitive normalization)
    - approximate_displacement: loose match or high token coverage found in a different chunk
    - fabrication:            no match found

    Notes:
    - "..." / "…" are treated as truncation markers, not literal text.
    - Loose matching is designed to prevent false failures due to quote glyphs, backticks, and punctuation.
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
            # debug
            "cited_chunk_present": cited_chunk_present,
            "cited_chunk_source": cited_source,
            "cited_chunk_text_id": cited_text_id,
            "cited_chunk_text_len": len(cited_text) if cited_text else 0,
        }

    # ── Strict match (punctuation-sensitive)
    segs_strict = _quote_segments_for_matching(cited_passage)
    cited_norm_strict = normalize_for_quote_matching(cited_text) if cited_text else ""

    if cited_norm_strict and _contains_segments_in_order(cited_norm_strict, segs_strict):
        return {
            "outcome": "verified",
            "in_cited_chunk": True,
            "in_any_chunk": True,
            "fabricated": False,
            "match_source_id": cited_text_id or "unknown",
            "match_method": "strict_segments",
            # debug
            "cited_chunk_present": cited_chunk_present,
            "cited_chunk_source": cited_source,
            "cited_chunk_text_id": cited_text_id,
            "cited_chunk_text_len": len(cited_text),
            "cited_chunk_preview": _safe_preview(cited_text, 140),
            "norm_segments": segs_strict[:3],
            "num_segments": len(segs_strict),
        }

    # ── Punctuation-insensitive match inside cited chunk -> VERIFIED (still verbatim content; ignores punctuation/quote glyphs)
    segs_loose = _quote_segments_for_matching_loose(cited_passage)
    cited_norm_loose = normalize_for_quote_matching_loose(cited_text) if cited_text else ""

    if cited_norm_loose and _contains_segments_in_order(cited_norm_loose, segs_loose):
        return {
            "outcome": "verified",
            "in_cited_chunk": True,
            "in_any_chunk": True,
            "fabricated": False,
            "match_source_id": cited_text_id or "unknown",
            "match_method": "loose_segments",
            "note": "VERIFIED — punctuation-insensitive match in cited chunk",
            # debug
            "cited_chunk_present": cited_chunk_present,
            "cited_chunk_source": cited_source,
            "cited_chunk_text_id": cited_text_id,
            "cited_chunk_text_len": len(cited_text),
            "cited_chunk_preview": _safe_preview(cited_text, 140),
            "loose_segments": segs_loose[:3],
            "num_loose_segments": len(segs_loose),
        }

    # ── Strict displacement search (elsewhere in corpus)
    if corpus and segs_strict:
        for cid, chunk in corpus.items():
            chunk_text = chunk.get("full_text", chunk.get("text", "")) or ""
            if not chunk_text:
                continue
            chunk_norm = normalize_for_quote_matching(chunk_text)
            if _contains_segments_in_order(chunk_norm, segs_strict):
                return {
                    "outcome": "displacement",
                    "in_cited_chunk": False,
                    "in_any_chunk": True,
                    "fabricated": False,
                    "match_source_id": str(chunk.get("text_id", "")),
                    "match_chunk_num": cid,
                    "match_chunk_source": str(chunk.get("source", "")),
                    "match_method": "strict_segments",
                    "note": "DISPLACEMENT — strict match found in different chunk",
                    # debug
                    "cited_chunk_present": cited_chunk_present,
                    "cited_chunk_source": cited_source,
                    "cited_chunk_text_id": cited_text_id,
                    "cited_chunk_text_len": len(cited_text) if cited_text else 0,
                    "cited_chunk_preview": _safe_preview(cited_text, 140) if cited_text else "",
                    "norm_segments": segs_strict[:3],
                    "num_segments": len(segs_strict),
                }

    # ── Loose displacement search (punctuation-tolerant) -> approximate_displacement
    if corpus and segs_loose:
        for cid, chunk in corpus.items():
            chunk_text = chunk.get("full_text", chunk.get("text", "")) or ""
            if not chunk_text:
                continue
            chunk_norm_loose = normalize_for_quote_matching_loose(chunk_text)
            if _contains_segments_in_order(chunk_norm_loose, segs_loose):
                return {
                    "outcome": "displacement",
                    "in_cited_chunk": False,
                    "in_any_chunk": True,
                    "fabricated": False,
                    "match_source_id": str(chunk.get("text_id", "")),
                    "match_chunk_num": cid,
                    "match_chunk_source": str(chunk.get("source", "")),
                    "match_method": "loose_segments",
                    "note": "DISPLACEMENT — punctuation-insensitive match found in different chunk",
                    # debug
                    "cited_chunk_present": cited_chunk_present,
                    "cited_chunk_source": cited_source,
                    "cited_chunk_text_id": cited_text_id,
                    "cited_chunk_text_len": len(cited_text) if cited_text else 0,
                    "cited_chunk_preview": _safe_preview(cited_text, 140) if cited_text else "",
                    "loose_segments": segs_loose[:3],
                    "num_loose_segments": len(segs_loose),
                }

    # ── Token-coverage heuristic (last chance): treat as approximate if most content tokens appear
    # This catches lightly paraphrased "quotes" that aren't substring-identical.
    if cited_norm_loose:
        quote_loose = " ".join(segs_loose) if segs_loose else normalize_for_quote_matching_loose(cited_passage)
        cov = _token_coverage(quote_loose, cited_norm_loose)
        if cov >= 0.95:
            return {
                "outcome": "verified",
                "in_cited_chunk": True,
                "in_any_chunk": True,
                "fabricated": False,
                "match_source_id": cited_text_id or "unknown",
                "match_method": "token_coverage",
                "approx_score": float(cov),
                "note": "VERIFIED — near-exact token coverage in cited chunk",
                # debug
                "cited_chunk_present": cited_chunk_present,
                "cited_chunk_source": cited_source,
                "cited_chunk_text_id": cited_text_id,
                "cited_chunk_text_len": len(cited_text) if cited_text else 0,
                "cited_chunk_preview": _safe_preview(cited_text, 140) if cited_text else "",
            }

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
                # debug
                "cited_chunk_present": cited_chunk_present,
                "cited_chunk_source": cited_source,
                "cited_chunk_text_id": cited_text_id,
                "cited_chunk_text_len": len(cited_text) if cited_text else 0,
                "cited_chunk_preview": _safe_preview(cited_text, 140) if cited_text else "",
            }

    # Optional: token-coverage search elsewhere (bounded by high threshold)
    if corpus and segs_loose:
        quote_loose = " ".join(segs_loose)
        best = (0.0, None, None, None)
        for cid, chunk in corpus.items():
            chunk_text = chunk.get("full_text", chunk.get("text", "")) or ""
            if not chunk_text:
                continue
            cn = normalize_for_quote_matching_loose(chunk_text)
            cov = _token_coverage(quote_loose, cn)
            if cov > best[0]:
                best = (cov, cid, str(chunk.get("text_id", "")), str(chunk.get("source", "")))
        if best[1] is not None and best[0] >= 0.95:
            return {
                "outcome": "displacement",
                "in_cited_chunk": False,
                "in_any_chunk": True,
                "fabricated": False,
                "match_source_id": best[2],
                "match_chunk_num": best[1],
                "match_chunk_source": best[3],
                "match_method": "token_coverage",
                "approx_score": float(best[0]),
                "note": "DISPLACEMENT — near-exact token coverage in different chunk",
                # debug
                "cited_chunk_present": cited_chunk_present,
                "cited_chunk_source": cited_source,
                "cited_chunk_text_id": cited_text_id,
                "cited_chunk_text_len": len(cited_text) if cited_text else 0,
                "cited_chunk_preview": _safe_preview(cited_text, 140) if cited_text else "",
            }

        if best[1] is not None and best[0] >= 0.90:
            return {
                "outcome": "approximate_displacement",
                "in_cited_chunk": False,
                "in_any_chunk": True,
                "fabricated": False,
                "match_source_id": best[2],
                "match_chunk_num": best[1],
                "match_chunk_source": best[3],
                "match_method": "token_coverage",
                "approx_score": float(best[0]),
                "note": "APPROXIMATE DISPLACEMENT — high token coverage in different chunk",
                # debug
                "cited_chunk_present": cited_chunk_present,
                "cited_chunk_source": cited_source,
                "cited_chunk_text_id": cited_text_id,
                "cited_chunk_text_len": len(cited_text) if cited_text else 0,
                "cited_chunk_preview": _safe_preview(cited_text, 140) if cited_text else "",
            }

    return {
        "outcome": "fabrication",
        "in_cited_chunk": False,
        "in_any_chunk": False,
        "fabricated": True,
        "note": "FABRICATION — passage not found in corpus",
        # debug
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

        results.append({
            "match": match_key,
            "text_id": text_id_str,
            "cited_num": cited_num,
            "cited_chunk_present": cited_present,
            "cited_chunk_source": str(cited_chunk.get("source", "")) if cited_chunk else "",
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
            "corpus": "lincoln_speech_corpus_repaired_1.json",
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
    SchemaComplete, FinalAnswerWordCount,
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

    # Build record — do NOT include 'Timestamp' here; DataLogger adds it automatically.
    # All values must be JSON-serializable primitives (str, int, float, bool) because
    # pygsheets cannot write Python objects directly to cells.
    record = {
        "QueryID": str(qresult.get("id", "")),
        "Query": str(qresult.get("query", "")),
        "Category": str(qresult.get("category", "")),
        # HV-5: Model configuration tag
        "ModelConfigTag": str(qresult.get("model_config_tag", "")),
        "HayModel": str(HAY_MODEL),
        "NicolayModel": str(NICOLAY_MODEL),
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
        "SchemaComplete", "FinalAnswerWordCount",
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
    status_cb=None
) -> dict:
    """
    Run the full Nicolay pipeline for one benchmark query and return a complete
    result record. Delegates to run_rag_pipeline() — no retrieval reimplementation.

    Parameters
    ----------
    qdef          : benchmark query definition dict
    openai_api_key: passed through to run_rag_pipeline
    cohere_api_key: passed through to run_rag_pipeline
    corpus_file   : path to lincoln_speech_corpus_repaired_1.(json|msgpack) (772-chunk corpus).
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
        "model_config_tag": f"hay={HAY_MODEL}|nicolay={NICOLAY_MODEL}|reranker={COHERE_RERANK_MODEL}|k={RERANK_K}|colbert=disabled",
    }

    # ── 1. RUN FULL PIPELINE ──────────────────────────────────────────────────
    status("🔍 Running pipeline (Hay → retrieval → rerank → Nicolay)...")
    try:
        pipeline_out = run_rag_pipeline(
            user_query=query,
            perform_keyword_search=True,
            perform_semantic_search=True,
            perform_colbert_search=False,   # ColBERT skipped — no Astra in benchmark
            perform_reranking=True,
            colbert_searcher=_NoOpColBERT(),  # Stub prevents Astra init in rag_pipeline.py
            openai_api_key=openai_api_key,
            cohere_api_key=cohere_api_key,
            top_n_results=10,               # Retrieve more candidates before reranking to k=5
        )
    except Exception as e:
        result["pipeline_error"] = str(e)
        status(f"❌ Pipeline error: {e}")
        return result

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
    # Load directly from corpus_file (the 772-chunk repaired JSON) rather than
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
    result["nicolay_final_answer_wordcount"] = len(final_answer_text.split()) if final_answer_text else 0

    match_analysis = nicolay_output.get("Match Analysis", {})
    relevance_map = {}
    if isinstance(match_analysis, dict):
        for mk, mv in match_analysis.items():
            if isinstance(mv, dict):
                relevance_map[mk] = mv.get("Relevance Assessment", "")
    result["nicolay_relevance_assessments"] = relevance_map

    # ── 6. QUOTE VERIFICATION ─────────────────────────────────────────────────
    # corpus_for_verify was built at step 2b above (new 772-chunk corpus).
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
    result["quotes_verified_count"] = outcomes.count("verified")
    result["quotes_approx_count"] = outcomes.count("approximate_quote")
    result["quotes_displaced_count"] = outcomes.count("displacement")
    result["quotes_approx_displaced_count"] = outcomes.count("approximate_displacement")
    result["quotes_fabricated_count"] = outcomes.count("fabrication")

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

    status(f"✓ {qid} complete")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown('<h1 style="font-family:Playfair Display,serif;color:#3d2b1f;">Nicolay Benchmark</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#7a6a5a;font-size:0.95rem;">Hay v3 + Nicolay v3 · Cohere rerank-v4.0-pro · 772-chunk corpus · 16 queries</p>', unsafe_allow_html=True)

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
        _base = "lincoln_speech_corpus_repaired_1"
        _candidates = [
            f"Data/{_base}.json", f"data/{_base}.json", f"{_base}.json",
            f"Data/{_base}.msgpack", f"data/{_base}.msgpack", f"{_base}.msgpack",
        ]
        _corpus_default = next((p for p in _candidates if Path(p).exists()), f"Data/{_base}.msgpack")
        corpus_file = st.text_input("Corpus path (JSON or MSGPACK)", value=_corpus_default)
        if corpus_file and not Path(corpus_file).exists():
            st.error(f"⚠️ Corpus file not found: {corpus_file}")
        elif corpus_file and Path(corpus_file).exists():
            st.caption(f"✅ Corpus file found")
        st.warning(
            "**Pipeline corpus note:** The benchmark script loads the corpus above "
            "for metrics and quote verification. But `run_rag_pipeline()` loads its "
            "own corpus internally via `modules/data_utils.py → load_lincoln_speech_corpus()`. "
            "If retrieval returns IDs only in the 0–79 range, that function is pointing "
            "at the old 80-doc corpus — fix the path in `data_utils.py` to point at "
            "`lincoln_speech_corpus_repaired_1.json`.",
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
        run_mode = st.radio("Mode", ["Single Query", "Full Benchmark", "Resume from Checkpoint"],
                            label_visibility="collapsed")

        st.markdown("---")
        show_group = st.multiselect("Query groups", ["core", "replacement"],
                                    default=["core", "replacement"])
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
            if st.button("▶️ Run All 16 Queries", type="primary"):
                do_run = "all"
        elif run_mode == "Resume from Checkpoint":
            pending = [q["id"] for q in BENCHMARK_QUERIES if q["id"] not in results.get("queries", {})]
            st.info(f"Pending: {len(pending)} queries — {', '.join(pending) if pending else 'none'}")
            if pending and st.button("▶️ Resume", type="primary"):
                do_run = "resume"

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
                            status_cb=lambda msg: status_box.info(msg)
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


            # Nicolay output
            with st.expander("✍️ Nicolay Output", expanded=True):
                final_text = get_final_answer_text(qr.get("nicolay_output", {}))
                st.markdown("**Final Answer:**")
                st.markdown(final_text)
                st.caption(f"Word count: {qr.get('nicolay_final_answer_wordcount', 0)}")

            # Quote verification
            with st.expander("🔎 Quote Verification", expanded=False):
                for qv_item in qr.get("quote_verification", []):
                    icon = {"verified": "✅", "approximate_quote": "🟡", "displacement": "⚠️", "approximate_displacement": "🟠", "fabrication": "🚨", "too_short": "—"}.get(qv_item.get("outcome", ""), "?")
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
        st.markdown(f"**{len(completed)} / 16 queries completed**")

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
