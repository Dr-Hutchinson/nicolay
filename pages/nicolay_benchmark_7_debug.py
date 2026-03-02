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
    """Extract Type A/B/C/D/E from Hay's query_assessment prose."""
    m = re.search(r'Type\s+([ABCDE])', query_assessment, re.IGNORECASE)
    return m.group(1).upper() if m else None


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
    Handles backtick quotes, em-dash variants, editorial brackets,
    whitespace, and unicode normalization.
    """
    if not text:
        return ""
    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)
    # Backtick quotes → standard
    text = text.replace("`", "'")
    # Curly quotes → straight
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Double/triple hyphens and em-dashes → space
    text = re.sub(r'[-\u2013\u2014]{2,}', ' ', text)
    text = re.sub(r'[-\u2013\u2014]', ' ', text)
    # Editorial brackets
    text = re.sub(r'\[.*?\]', '', text)
    # Whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def verify_quote(cited_passage: str, cited_chunk: dict, corpus: dict) -> dict:
    """
    Three-outcome quote verification:
    - in_cited_chunk: passage found in the chunk Nicolay cited
    - in_any_chunk but different chunk: displacement
    - in neither: fabrication
    NOTE: The prior pipeline's quote_match TRUE/FALSE logging has a known logic bug —
    this normalization-based approach is the first accurate quote check the system has had.
    """
    if not cited_passage or len(cited_passage.strip()) < 5:
        return {"outcome": "too_short", "in_cited_chunk": None, "in_any_chunk": None,
                "fabricated": None, "note": "Passage too short to verify"}

    norm_passage = normalize_for_quote_matching(cited_passage)

    # Check cited chunk first
    cited_text = cited_chunk.get("full_text", cited_chunk.get("text", "")) if cited_chunk else ""
    if cited_text and norm_passage in normalize_for_quote_matching(cited_text):
        return {"outcome": "verified", "in_cited_chunk": True, "in_any_chunk": True,
                "fabricated": False, "match_source_id": cited_chunk.get("text_id", "unknown")}

    # Search full corpus
    for chunk in corpus.values():
        norm_chunk = normalize_for_quote_matching(chunk.get("full_text", chunk.get("text", "")))
        if norm_passage in norm_chunk:
            return {"outcome": "displacement", "in_cited_chunk": False, "in_any_chunk": True,
                    "fabricated": False, "match_source_id": chunk.get("text_id", ""),
                    "note": "DISPLACEMENT — passage found in different chunk"}

    return {"outcome": "fabrication", "in_cited_chunk": False, "in_any_chunk": False,
            "fabricated": True, "note": "FABRICATION — passage not found in corpus"}


def verify_all_quotes(nicolay_output: dict, reranked: list[dict], corpus: dict) -> list[dict]:
    """Run quote verification for all Match Analysis entries."""
    match_analysis = nicolay_output.get("Match Analysis", {})
    if not isinstance(match_analysis, dict):
        return []

    reranked_by_num = {r["text_id_num"]: r for r in reranked}
    results = []

    for match_key, match_val in match_analysis.items():
        if not isinstance(match_val, dict):
            continue

        key_passage = match_val.get("Key Quote", match_val.get("Key Passage", ""))  # v3 uses "Key Quote"
        text_id_str = match_val.get("Text ID", "")

        # Parse integer from either "Text #: 77" or bare "77"
        cited_chunk = None
        try:
            tid = str(text_id_str).strip()
            if "Text #:" in tid:
                num = int(tid.split("Text #:")[1].strip())
            else:
                num = int(tid)
            cited_chunk = corpus.get(num)
        except (ValueError, IndexError):
            pass

        verification = verify_quote(key_passage, cited_chunk, corpus)
        results.append({
            "match": match_key,
            "text_id": text_id_str,
            "cited_passage": key_passage,
            **verification
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
    Append one benchmark query result row to the 'benchmark_results' Google Sheet.
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

    Sheet columns (must be present in row 1 of benchmark_results sheet, in this order):
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
        "HayModel": str(HAY_MODEL),
        "NicolayModel": str(NICOLAY_MODEL),
        # Hay layer
        "HayTypeExpected": str(qresult.get("expected_hay_type", "")),
        "HayTypeGot": str(qresult.get("hay_task_type_raw", "") or ""),
        "HayTypeCorrect": str(qresult.get("hay_task_type_correct", "")),
        "HayKeywordCount": int(qresult.get("hay_keyword_count", 0) or 0),
        "HaySpuriousFields": json.dumps(qresult.get("hay_spurious_fields", [])),
        "HayTrainingBleed": str(qresult.get("hay_training_bleed", False)),
        "InitialAnswer": str(hay_out.get("initial_answer", "")),
        "QueryAssessment": str(hay_out.get("query_assessment", "")),
        # Retrieval layer
        "RetrievedDocIDs": json.dumps(qresult.get("retrieved_doc_ids", [])),
        "RetrievalSearchTypes": json.dumps(qresult.get("retrieval_search_types", [])),
        "RerankerScores": json.dumps([
            round(s, 6) if isinstance(s, float) else s
            for s in qresult.get("reranker_scores", [])
        ]),
        "PrecisionAt5": float(qresult.get("precision_at_5", 0) or 0),
        "RecallAt5": float(qresult.get("recall_at_5", 0) or 0),
        "CeilingAdjustedPrecision": float(qresult.get("ceiling_adjusted_precision", 0) or 0),
        "IdealDocsHit": json.dumps(qresult.get("ideal_docs_hit", [])),
        "IdealDocsMissed": json.dumps(qresult.get("ideal_docs_missed", [])),
        # Nicolay layer
        "NicolayTypeExpected": str(qresult.get("expected_nicolay_type", "")),
        "NicolayTypeGot": str(qresult.get("nicolay_synthesis_type_raw", "") or ""),
        "NicolayTypeCorrect": str(qresult.get("nicolay_synthesis_type_correct", "")),
        "SchemaComplete": str(qresult.get("nicolay_schema_complete", "")),
        "FinalAnswerWordCount": int(qresult.get("nicolay_final_answer_wordcount", 0) or 0),
        # Quote verification
        "QuotesVerified": int(qresult.get("quotes_verified_count", 0) or 0),
        "QuotesDisplaced": int(qresult.get("quotes_displaced_count", 0) or 0),
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
    Write the required column headers to row 1 of the 'benchmark_results' sheet.
    Call this once before the first benchmark run to ensure positional alignment
    with DataLogger.record_api_outputs() which uses set_dataframe(copy_head=False).

    Safe to call again — checks if headers are already present before writing.
    """
    if gc_client is None:
        return
    headers = [
        "Timestamp", "QueryID", "Query", "Category", "HayModel", "NicolayModel",
        "HayTypeExpected", "HayTypeGot", "HayTypeCorrect", "HayKeywordCount",
        "HaySpuriousFields", "HayTrainingBleed", "InitialAnswer", "QueryAssessment",
        "RetrievedDocIDs", "RetrievalSearchTypes", "RerankerScores",
        "PrecisionAt5", "RecallAt5", "CeilingAdjustedPrecision",
        "IdealDocsHit", "IdealDocsMissed",
        "NicolayTypeExpected", "NicolayTypeGot", "NicolayTypeCorrect",
        "SchemaComplete", "FinalAnswerWordCount",
        "QuotesVerified", "QuotesDisplaced", "QuotesFabricated",
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
        sh = gc_client.open("benchmark_results").sheet1
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
        sh = gc_client.open("benchmark_results").sheet1
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
        st.warning(f"Could not find {qid} in benchmark_results sheet to update rubric.")
    except Exception as e:
        st.warning(f"⚠️ Rubric sheet update failed: {e}")


def export_csv(results: dict, csv_file: str = CSV_FILE) -> bytes:
    """Serialize results to CSV bytes for Streamlit download. No file write needed."""
    rows = []
    for qid, qdata in results.get("queries", {}).items():
        rows.append({
            "id": qid,
            "category": qdata.get("category", ""),
            "expected_hay_type": qdata.get("expected_hay_type", ""),
            "hay_type_raw": qdata.get("hay_task_type_raw", ""),
            "hay_type_correct": qdata.get("hay_task_type_correct", ""),
            "hay_keyword_count": qdata.get("hay_keyword_count", ""),
            "hay_spurious_fields": str(qdata.get("hay_spurious_fields", [])),
            "precision_at_5": qdata.get("precision_at_5", ""),
            "recall_at_5": qdata.get("recall_at_5", ""),
            "ceiling_adjusted_precision": qdata.get("ceiling_adjusted_precision", ""),
            "expected_nicolay_type": qdata.get("expected_nicolay_type", ""),
            "nicolay_type_raw": qdata.get("nicolay_synthesis_type_raw", ""),
            "nicolay_type_correct": qdata.get("nicolay_synthesis_type_correct", ""),
            "schema_complete": qdata.get("nicolay_schema_complete", ""),
            "final_answer_wordcount": qdata.get("nicolay_final_answer_wordcount", ""),
            "quotes_verified": qdata.get("quotes_verified_count", ""),
            "quotes_displaced": qdata.get("quotes_displaced_count", ""),
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

    ID remapping: The pipeline's reranked DataFrame may carry old corpus text IDs
    (parent document IDs from the original index). If a `corpus` dict is provided
    (keyed by new int IDs), we verify whether the raw ID exists in the new corpus.
    If not, we attempt to find the chunk by matching the 'Source' field against
    corpus entries to resolve to the new ID. This ensures displayed IDs and
    precision/recall metrics are aligned to the 772-chunk new corpus index.

    Search type: `Search Type` column may be absent or empty in some pipeline
    versions — falls back to "Unknown" for display purposes.
    """
    # Build a source→new_id lookup from the corpus for ID remapping
    source_to_new_id = {}
    if corpus:
        for new_id, chunk in corpus.items():
            src = chunk.get("source", "")
            if src:
                source_to_new_id.setdefault(src, new_id)

    records = []
    for _, row in reranked_df.iterrows():
        raw_id = str(row.get("Text ID", "")).strip()

        # Parse raw ID to integer
        try:
            num_id = int(raw_id)
        except ValueError:
            m = re.search(r"(\d+)", raw_id)
            num_id = int(m.group(1)) if m else None

        # Remap to new corpus ID if the raw ID is not present in the new corpus
        if corpus and num_id is not None and num_id not in corpus:
            # Try to resolve via Source field
            row_source = str(row.get("Source", "")).strip()
            remapped = source_to_new_id.get(row_source)
            if remapped is not None:
                num_id = remapped

        # Normalize search type label — pipeline column is "Search Type"
        raw_search_type = str(row.get("Search Type", "")).strip()
        search_type = raw_search_type if raw_search_type else "Unknown"

        records.append({
            "text_id_num": num_id,
            "text_id_str": f"Text #: {num_id}" if num_id is not None else raw_id,
            "source": row.get("Source", ""),
            "full_text": row.get("Key Quote", ""),  # pipeline stores full text here
            "reranker_score": row.get("Relevance Score"),
            "_search_type": search_type,
        })
    return records


def run_pipeline_for_query(
    qdef: dict,
    openai_api_key: str,
    cohere_api_key: str,
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
    status_cb     : optional callable(str) for progress UI updates
    """
    from modules.rag_pipeline import run_rag_pipeline
    from modules.data_utils import load_lincoln_speech_corpus

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

    # ── 2b. LOAD CORPUS (needed for ID remapping + quote verification) ────────
    # Build int-keyed dict {413: chunk_dict, ...} from the 772-chunk new corpus.
    # Loading here (before retrieval metrics) ensures the corpus is available for
    # reranked_df_to_list to remap old parent text IDs to new chunk IDs.
    try:
        lincoln_data_df = load_lincoln_speech_corpus()
        lincoln_data = lincoln_data_df.to_dict("records")
        corpus_for_verify = {}
        for item in lincoln_data:
            tid = item.get("text_id", "")
            m = re.search(r"(\d+)", str(tid))
            if m:
                corpus_for_verify[int(m.group(1))] = item
    except Exception:
        corpus_for_verify = {}

    # DEBUG: log corpus key count
    try:
        st.session_state[f"_debug_corpus_size_{qid}"] = f"{len(corpus_for_verify)} keys (range: {min(corpus_for_verify) if corpus_for_verify else 'N/A'} – {max(corpus_for_verify) if corpus_for_verify else 'N/A'})"
    except Exception:
        pass
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

    metrics = compute_retrieval_metrics(reranked, qdef["ideal_docs_new"], qdef.get("ideal_docs_original"))
    result.update(metrics)

    # ── 5. NICOLAY METRICS ────────────────────────────────────────────────────
    status("✍️ Computing Nicolay metrics...")
    result["nicolay_output"] = nicolay_output

    synth_str = get_synthesis_assessment(nicolay_output)
    nicolay_type = extract_nicolay_type(synth_str)
    result["nicolay_synthesis_type_raw"] = nicolay_type
    result["nicolay_synthesis_type_correct"] = (
        (nicolay_type == qdef["expected_nicolay_type"]) if nicolay_type else None
    )

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

    qv = verify_all_quotes(nicolay_output, reranked, corpus_for_verify)
    result["quote_verification"] = qv
    outcomes = [q.get("outcome") for q in qv]
    result["quotes_verified_count"] = outcomes.count("verified")
    result["quotes_displaced_count"] = outcomes.count("displacement")
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

        openai_key = st.text_input("OpenAI API Key", type="password",
                                   value=os.environ.get("OPENAI_API_KEY", ""))
        cohere_key = st.text_input("Cohere API Key", type="password",
                                   value=os.environ.get("COHERE_API_KEY", ""))

        # Auto-detect corpus across common repo layouts (root, Data/, data/)
        _cf = "lincoln_speech_corpus_repaired_1.json"
        _corpus_default = next(
            (p for p in [f"Data/{_cf}", f"data/{_cf}", _cf] if Path(p).exists()),
            f"Data/{_cf}"
        )
        corpus_file = st.text_input("Corpus JSON path", value=_corpus_default)
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
                    benchmark_logger = DataLogger(gc=gc_client, sheet_name="benchmark_results")
                    st.success("✅ Sheets connected")
                    # Button to initialize/verify sheet headers (run once before first benchmark)
                    if st.button("📋 Init Sheet Headers", help="Write required column headers to row 1. Safe to run again."):
                        init_benchmark_sheet_headers(gc_client)
                    # Standalone connection test — writes one test row to verify the
                    # full write path works independently of a pipeline run.
                    if st.button("🧪 Test Sheets Write", help="Write a dummy row to benchmark_results to verify the full logging path."):
                        test_record = {
                            "QueryID": "TEST",
                            "Query": "Sheets connectivity test",
                            "Category": "debug",
                            "HayModel": HAY_MODEL, "NicolayModel": NICOLAY_MODEL,
                            "HayTypeExpected": "", "HayTypeGot": "", "HayTypeCorrect": "",
                            "HayKeywordCount": 0, "HaySpuriousFields": "[]",
                            "HayTrainingBleed": "False", "InitialAnswer": "", "QueryAssessment": "",
                            "RetrievedDocIDs": "[]", "RetrievalSearchTypes": "[]", "RerankerScores": "[]",
                            "PrecisionAt5": 0.0, "RecallAt5": 0.0, "CeilingAdjustedPrecision": 0.0,
                            "IdealDocsHit": "[]", "IdealDocsMissed": "[]",
                            "NicolayTypeExpected": "", "NicolayTypeGot": "", "NicolayTypeCorrect": "",
                            "SchemaComplete": "", "FinalAnswerWordCount": 0,
                            "QuotesVerified": 0, "QuotesDisplaced": 0, "QuotesFabricated": 0,
                            "BleuMaxRetrieved": 0.0, "BleuAvgRetrieved": 0.0,
                            "Rouge1MaxRetrieved": 0.0, "Rouge1AvgRetrieved": 0.0,
                            "Rouge1MaxIdeal": 0.0, "Rouge1AvgIdeal": 0.0,
                            "Rouge1RetrievedVsIdealRatio": 0.0,
                            "RougeL_MaxRetrieved": 0.0, "RougeL_MaxIdeal": 0.0,
                            "CriticalMissingEvidence": "",
                            "RubricFactualAccuracy": "", "RubricCitationAccuracy": "",
                            "RubricHistoriographicalDepth": "", "RubricEpistemicCalibration": "",
                            "RubricTotal": "", "EvaluatorNotes": "TEST ROW — delete after verification",
                        }
                        try:
                            benchmark_logger.record_api_outputs(test_record)
                            st.success("✅ Test row written successfully — check benchmark_results sheet.")
                        except Exception as e:
                            st.error(f"❌ Test write failed: {type(e).__name__}: {e}")
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
                st.error("OpenAI API key required.")
                return False
            if not cohere_key:
                st.error("Cohere API key required.")
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

            # Nicolay output
            with st.expander("✍️ Nicolay Output", expanded=True):
                final_text = get_final_answer_text(qr.get("nicolay_output", {}))
                st.markdown("**Final Answer:**")
                st.markdown(final_text)
                st.caption(f"Word count: {qr.get('nicolay_final_answer_wordcount', 0)}")

            # Quote verification
            with st.expander("🔎 Quote Verification", expanded=False):
                for qv_item in qr.get("quote_verification", []):
                    icon = {"verified": "✅", "displacement": "⚠️", "fabrication": "🚨", "too_short": "—"}.get(qv_item.get("outcome", ""), "?")
                    st.markdown(f"{icon} **{qv_item.get('match', '')}** ({qv_item.get('text_id', '')}) — *{qv_item.get('outcome', '')}*")
                    if qv_item.get("cited_passage"):
                        st.caption(f'"{qv_item["cited_passage"][:150]}..."')
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
                    "Displaced": qd.get("quotes_displaced_count", ""),
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
