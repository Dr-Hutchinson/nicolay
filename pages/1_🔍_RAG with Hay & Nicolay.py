import streamlit as st
import json
import pygsheets
from google.oauth2 import service_account
import re
from openai import OpenAI
import cohere
import os
import pandas as pd
import numpy as np
from datetime import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor
import yaml
try:
    from rouge_score import rouge_scorer as _rouge_scorer_mod
    _ROUGE_AVAILABLE = True
except ImportError:
    _rouge_scorer_mod = None
    _ROUGE_AVAILABLE = False

# ── version 1.7 ──────────────────────────────────────────────────────────────
# UI Enhancements over v1.6:
#   [U1]  Hay keyword transparency strip (retained)
#   [U2]  Synthesis type badge (retained)
#   [U3]  Calibration decoupling warning → replaced by [U9] multi-signal panel
#   [U4]  Session query history in sidebar (retained; dedup guard added)
#   [U5]  Corpus coverage notice (retained)
#   [U6]  Retrieval diagnostics panel (retained; "Used by Nicolay" ID fix)
#   [U7]  Card text size (retained)
#   [U8]  Quote verification labels (retained)
#   [U9]  Multi-signal diagnostic dashboard: four independent heuristic signals
#         (calibration gap, low retrieval ceiling, Type 3/4 synthesis,
#         FinalAnswer brevity) each shown only when triggered.
#   [U10] FinalAnswer quote verification: quoted strings in Nicolay's synthesis
#         are matched against retrieved corpus chunks; unverified quotes flagged
#         inline. Out-of-corpus source references in References list flagged.
#   [U11] Three-tab result layout (Answer / Sources / Pipeline) for mobile
#         friendliness and showcase readability. All existing content preserved.
#   [U12] Response Confidence Summary panel: five independently computed signals
#         (quote verification rate, ROUGE-1/2 corpus grounding, reranker score
#         spread/ceiling, source diversity, complexity-type match heuristic)
#         aggregated into a scannable epistemological situation report. Displayed
#         between the synthesis type badge and FinalAnswer text in Tab 1.
#         Refactors S1 calibration gap detection into analyze_reranker_scores().
#   [FIX] render_sidebar_history() called only once at top; post-query re-call
#         removed to prevent duplicate history entries.
#   [FIX] Hay pill query-type regex extended to cover Absence Recognition,
#         Multi-passage, Temporal.
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nicolay: Exploring the Speeches of Abraham Lincoln with AI",
    layout='wide',
    page_icon='🔍'
)

# ── Tab styling — make tabs visually prominent ────────────────────────────────
st.markdown("""
<style>
/* Tab bar background */
div[data-baseweb="tab-list"] {
    background-color: #1a1a2e;
    border-radius: 8px 8px 0 0;
    padding: 4px 8px 0 8px;
    gap: 4px;
}
/* Individual tab buttons */
button[data-baseweb="tab"] {
    background-color: #2a2a4a !important;
    color: #a0a8c0 !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 10px 24px !important;
    font-size: 0.95em !important;
    font-weight: 600 !important;
    border: none !important;
    transition: background 0.2s, color 0.2s;
}
/* Hovered tab */
button[data-baseweb="tab"]:hover {
    background-color: #3a3a6a !important;
    color: #d0d8f0 !important;
}
/* Active / selected tab */
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #4a6fa5 !important;
    color: #ffffff !important;
    border-bottom: 3px solid #7eb3ff !important;
}
/* Tab panel top border */
div[data-baseweb="tab-panel"] {
    border-top: 3px solid #4a6fa5;
    padding-top: 1.2em;
}
</style>
""", unsafe_allow_html=True)

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
client = OpenAI()

os.environ["CO_API_KEY"] = st.secrets["cohere_api_key"]
co = cohere.Client()

scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"], scopes=scope)
gc = pygsheets.authorize(custom_credentials=credentials)

from modules.data_utils import (
    load_lincoln_speech_corpus,
    load_voyant_word_counts,
    load_lincoln_index_embedded,
)
from modules.keyword_search import search_with_dynamic_weights_expanded


# ── Session state initialisation ──────────────────────────────────────────────
if "query_history" not in st.session_state:
    st.session_state["query_history"] = []   # list of dicts for sidebar [U4]


# ── DataLogger ────────────────────────────────────────────────────────────────
class DataLogger:
    def __init__(self, gc, sheet_name):
        self.gc = gc
        self.sheet = self.gc.open(sheet_name).sheet1

    def record_api_outputs(self, data_dict):
        data_dict['Timestamp'] = dt.now()
        df = pd.DataFrame([data_dict])
        end_row = len(self.sheet.get_all_records()) + 2
        self.sheet.set_dataframe(df, (end_row, 1), copy_head=False, extend=True)


hays_data_logger         = DataLogger(gc, 'hays_data')
keyword_results_logger   = DataLogger(gc, 'keyword_search_results')
semantic_results_logger  = DataLogger(gc, 'semantic_search_results')
reranking_results_logger = DataLogger(gc, 'reranking_results')
nicolay_data_logger      = DataLogger(gc, 'nicolay_data')


# ── Logging helpers ───────────────────────────────────────────────────────────
def log_keyword_search_results(logger, df, user_query, initial_answer,
                                weighted_kw, year_kw, text_kw):
    now = dt.now()
    for _, row in df.iterrows():
        logger.record_api_outputs({
            'Timestamp': now,
            'UserQuery': user_query,
            'initial_Answer': initial_answer,
            'Weighted_Keywords': str(weighted_kw),
            'Year_Keywords': str(year_kw),
            'text_keywords': str(text_kw),
            'TextID': row.get('text_id', ''),
            'KeyQuote': row.get('key_quote', row.get('quote', '')),
            'WeightedScore': row.get('weighted_score', ''),
            'KeywordCounts': json.dumps(row.get('keyword_counts', {}))
        })


def log_semantic_search_results(logger, df, initial_answer):
    now = dt.now()
    for _, row in df.iterrows():
        logger.record_api_outputs({
            'Timestamp': now,
            'UserQuery': row.get('UserQuery', ''),
            'HyDE_Query': initial_answer,
            'TextID': row.get('text_id', ''),
            'SimilarityScore': row.get('similarities', ''),
            'TopSegment': row.get('TopSegment', '')
        })


def log_reranking_results(logger, df, user_query):
    now = dt.now()
    for _, row in df.iterrows():
        logger.record_api_outputs({
            'Timestamp': now,
            'UserQuery': user_query,
            'Rank': row.get('Rank', ''),
            'SearchType': row.get('Search Type', ''),
            'TextID': row.get('Text ID', ''),
            'KeyQuote': row.get('Key Quote', ''),
            'Relevance_Score': row.get('Relevance Score', '')
        })


def log_nicolay_model_output(logger, model_output, user_query,
                              highlight_success_dict, initial_answer):
    fa         = model_output.get("FinalAnswer", {})
    refs_raw   = fa.get("References", [])
    qa         = model_output.get("User Query Analysis", {})
    mf         = model_output.get("Model Feedback", {})
    match_data = {}
    for mk, md in model_output.get("Match Analysis", {}).items():
        fields = ['Text ID', 'Source', 'Summary', 'Key Quote',
                  'Historical Context', 'Relevance Assessment']
        match_data[mk] = "; ".join(f"{f}: {md.get(f,'')}" for f in fields)
    record = {
        'Timestamp': dt.now(),
        'UserQuery': user_query,
        'initial_Answer': initial_answer,
        'FinalAnswer': fa.get("Text", ""),
        'References': ", ".join(refs_raw) if isinstance(refs_raw, list) else str(refs_raw),
        'SynthesisAssessment': qa.get("synthesis_assessment", ""),
        'QueryIntent': qa.get("Query Intent", ""),
        'HistoricalContext': qa.get("Historical Context", ""),
        'AnswerEvaluation': model_output.get("Initial Answer Review", {}).get("Answer Evaluation", ""),
        'QuoteIntegration': model_output.get("Initial Answer Review", {}).get("Quote Integration Points", ""),
        **match_data,
        'MetaStrategy': str(model_output.get("Meta Analysis", {}).get("Strategy for Response Composition", {})),
        'MetaSynthesis': model_output.get("Meta Analysis", {}).get("Synthesis", ""),
        'RetrievalQualityNotes': mf.get("Retrieval Quality Notes", mf.get("Response Effectiveness", "")),
        'CriticalMissingEvidence': mf.get("Critical Missing Evidence Flag", ""),
        'SuggestedImprovements': mf.get("Suggested Improvements", mf.get("Suggestions for Improvement", ""))
    }
    for mk, success in highlight_success_dict.items():
        record[f'{mk}_HighlightSuccess'] = success
    logger.record_api_outputs(record)


# ── Prompt loading ────────────────────────────────────────────────────────────
def load_prompt(file_name):
    with open(file_name, 'r') as f:
        return f.read()


def load_prompts():
    defaults = {
        'keyword_model_system_prompt': 'prompts/keyword_model_system_prompt.txt',
        'response_model_system_prompt': 'prompts/response_model_system_prompt.txt',
        'app_intro': 'prompts/app_intro.txt',
        'keyword_search_explainer': 'prompts/keyword_search_explainer.txt',
        'semantic_search_explainer': 'prompts/semantic_search_explainer.txt',
        'relevance_ranking_explainer': 'prompts/relevance_ranking_explainer.txt',
        'nicolay_model_explainer': 'prompts/nicolay_model_explainer.txt',
    }
    for key, path in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = load_prompt(path)


load_prompts()

keyword_prompt              = st.session_state['keyword_model_system_prompt']
response_prompt             = st.session_state['response_model_system_prompt']
app_intro                   = st.session_state['app_intro']
keyword_search_explainer    = st.session_state['keyword_search_explainer']
semantic_search_explainer   = st.session_state['semantic_search_explainer']
relevance_ranking_explainer = st.session_state['relevance_ranking_explainer']
nicolay_model_explainer     = st.session_state['nicolay_model_explainer']


# ── Utility functions ─────────────────────────────────────────────────────────
def segment_text(text, segment_size=500, overlap=100):
    words = text.split()
    return [' '.join(words[i:i + segment_size])
            for i in range(0, len(words), segment_size - overlap)]


def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=[text.replace("\n", " ")], model=model)
    return np.array(response.data[0].embedding)


def cosine_similarity(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return float(np.dot(v1, v2) / (n1 * n2)) if n1 and n2 else 0.0


def search_text_local(df, query, n=5):
    """Semantic search over pre-loaded parquet embeddings — no per-row API calls."""
    emb = get_embedding(query)
    df  = df.copy()
    df["similarities"] = df['embedding'].apply(
        lambda x: cosine_similarity(np.array(x), emb)
        if isinstance(x, (list, np.ndarray)) else 0.0
    )
    top = df.sort_values("similarities", ascending=False).head(n).copy()
    top["UserQuery"] = query
    return top, emb


def compare_segments_parallel(segments, query_emb):
    with ThreadPoolExecutor(max_workers=5) as ex:
        embs = [f.result() for f in [ex.submit(get_embedding, s) for s in segments]]
    return [(segments[i], cosine_similarity(embs[i], query_emb)) for i in range(len(segments))]


def highlight_key_quote(text, key_quote):
    parts = key_quote.split("...")
    pattern = (re.escape(parts[0]) + r"\s*.*?\s*" + re.escape(parts[-1]) + r"[.;,]?"
               if len(parts) >= 2 else re.escape(key_quote) + r"\s*[.;,]?")
    for m in re.compile(pattern, re.IGNORECASE | re.DOTALL).findall(text):
        text = text.replace(m, f"<mark>{m}</mark>")
    return text


def triple_lookup(lincoln_dict, text_id):
    """Return corpus entry dict using any of the three key formats."""
    m = re.search(r"(\d+)", str(text_id))
    num = int(m.group(1)) if m else None
    for key in (text_id,
                f"Text #: {text_id}",
                f"Text #: {num}" if num is not None else None,
                num,
                str(num) if num is not None else None):
        if key is None:
            continue
        entry = lincoln_dict.get(key)
        if isinstance(entry, dict):
            return entry
    return {}


def format_reranked_for_nicolay(reranked_results, lincoln_dict):
    """
    Format top-5 reranked results for Nicolay model input.
    Source is explicitly included so Nicolay has a ground-truth anchor
    and cannot confabulate a source name from parametric memory.
    """
    out = []
    for i, r in enumerate(reranked_results[:5], 1):
        tid    = str(r.get('Text ID', 'Unknown')).strip()
        source = r.get('Source', '')
        if not source or source == 'Source not available':
            entry  = triple_lookup(lincoln_dict, tid)
            source = entry.get('source', 'Source not available')
        entry     = triple_lookup(lincoln_dict, tid)
        full_text = entry.get('full_text') or r.get('Key Quote', 'No quote')
        out.append(
            f"Match {i}: "
            f"Search Type - {r.get('Search Type','Unknown')}, "
            f"Text ID - {tid}, "
            f"Source (use this exact string as the citation label) - {source}, "
            f"Summary (curatorial description only - not quotable corpus text) - {r.get('Summary','No summary')}, "
            f"Full Text (select the most relevant passage to quote directly) - {full_text}, "
            f"Relevance Score - {r.get('Relevance Score', 0.0):.2f}"
        )
    return "\n\n".join(out) if out else "No results to format"


# ── E1 / U8: Quote verification — benchmark-grade stack ──────────────────────
# Ported from nicolay_benchmark for consistency. Key improvements over original:
#   • Segment-based matching handles ellipsis truncation correctly
#   • Loose (punctuation-insensitive) normalization catches quote-glyph variants
#   • Stage 3.5: hyphen/punctuation-collapsed anchor catches corpus parsing
#     artifacts (hyphens → spaces, dropped commas) without loosening enough
#     to create false positives in 19th-century prose
#   • Token-coverage heuristic catches near-exact paraphrases (≥ 0.95 coverage
#     = verified; 0.86–0.95 = approximate_quote, not displaced)
#   • approximate_quote outcome is treated as verified for confidence scoring,
#     not as displaced — eliminates the main source of false 🔀 annotations
# Backward-compatible: verify_quote() still returns (bool, method_str)

import unicodedata as _ud

_STOPWORDS = {
    "the","a","an","and","or","but","if","then","than","to","of","in","on",
    "for","with","by","at","as","is","are","was","were","be","been","being",
    "it","its","this","that","these","those","we","you","i","he","she","they",
    "them","his","her","their","our","us","my","your","not","no","so","do",
    "does","did","from","into","over","under","up","down","out","about",
    "because","which","who","whom","what","when","where","why","how",
}


def _norm_chunk(text):
    """Legacy whitespace normalizer — kept for compatibility with callers."""
    text = text.replace('\\n', ' ').replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_for_quote_matching(text: str) -> str:
    """Strict normalization: unicode, quote glyphs, dash variants, editorial brackets."""
    if not text:
        return ""
    text = _ud.normalize("NFKD", str(text))
    text = text.replace("`", "'")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2026", "...")
    text = re.sub(r'[-\u2013\u2014]{2,}', ' ', text)
    text = re.sub(r'[-\u2013\u2014]', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def normalize_for_quote_matching_loose(text: str) -> str:
    """Loose normalization: also strips all remaining punctuation."""
    if not text:
        return ""
    text = _ud.normalize("NFKD", str(text))
    text = text.replace("`", "'")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2026", "...")
    text = re.sub(r'[-\u2013\u2014]{2,}', ' ', text)
    text = re.sub(r'[-\u2013\u2014]', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"[^0-9A-Za-z\s]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def _content_tokens(norm_text: str) -> list:
    if not norm_text:
        return []
    return [t for t in norm_text.split() if len(t) >= 3 and t not in _STOPWORDS]


def _token_coverage(quote_norm_loose: str, chunk_norm_loose: str) -> float:
    q = set(_content_tokens(quote_norm_loose))
    if len(q) < 6:
        return 0.0
    c = set(_content_tokens(chunk_norm_loose))
    return len(q & c) / max(1, len(q))


def _quote_segments_for_matching(passage: str) -> list:
    if not passage:
        return []
    p = _ud.normalize("NFKD", str(passage)).strip().strip('"').strip("'").strip()
    parts = re.split(r'(?:\.\.\.|\u2026)', p)
    segs = [normalize_for_quote_matching(pt) for pt in parts]
    segs = [s for s in segs if len(s) >= 10]
    if not segs:
        p2 = re.sub(r'(?:\.\.\.|\u2026)+', ' ', p)
        s2 = normalize_for_quote_matching(p2)
        if s2:
            segs = [s2]
    return segs


def _quote_segments_for_matching_loose(passage: str) -> list:
    if not passage:
        return []
    p = _ud.normalize("NFKD", str(passage)).strip().strip('"').strip("'").strip()
    parts = re.split(r'(?:\.\.\.|\u2026)', p)
    segs = [normalize_for_quote_matching_loose(pt) for pt in parts]
    segs = [s for s in segs if len(s) >= 18]
    if not segs:
        p2 = re.sub(r'(?:\.\.\.|\u2026)+', ' ', p)
        s2 = normalize_for_quote_matching_loose(p2)
        if s2:
            segs = [s2]
    return segs


def _contains_segments_in_order(haystack_norm: str, segs_norm: list) -> bool:
    if not haystack_norm or not segs_norm:
        return False
    i = 0
    for s in segs_norm:
        pos = haystack_norm.find(s, i)
        if pos == -1:
            return False
        i = pos + len(s)
    return True


def _edit_distance_short(a: str, b: str) -> int:
    """Levenshtein — O(mn), called only on ≤60-char strings."""
    if a == b: return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j]+1, curr[j-1]+1, prev[j-1]+(0 if ca==cb else 1))
        prev = curr
    return prev[lb]


def _sliding_window_verify(norm_quote, norm_chunk, min_window=30, step=10):
    """Legacy sliding-window — kept for verify_final_answer_quotes fallback."""
    q_len = len(norm_quote)
    for window in range(min(60, q_len), min_window - 1, -step):
        for start in range(0, q_len - window + 1, step):
            segment = norm_quote[start:start + window]
            if segment in norm_chunk:
                return True, segment
    return False, ''


def _verify_quote_rich(key_quote: str, cited_chunk: dict, corpus: dict) -> dict:
    """
    Multi-stage quote verifier returning a rich result dict.

    Outcomes (in priority order):
      verified            — confirmed in cited chunk (any stage 1-3.5 or token_coverage ≥ 0.95)
      approximate_quote   — high token coverage (0.86-0.95) in cited chunk
      displacement        — strict/loose segments found in a different corpus chunk
      approximate_displacement — token coverage ≥ 0.95 in a different chunk
      fabrication         — not found anywhere

    Stages:
      1. Strict segment match (normalize_for_quote_matching)
      2. Loose segment match (normalize_for_quote_matching_loose, punctuation-stripped)
      3. Corpus-wide strict displacement scan
      3a. Corpus-wide loose displacement scan
      3.5. Hyphen/punctuation-collapsed anchor on cited chunk with edit-distance ≤ 3
      4. Token-coverage heuristic on cited chunk
      5. Token-coverage displacement scan
    """
    cited_chunk_present = bool(cited_chunk)
    cited_text = cited_chunk.get("full_text", cited_chunk.get("text", "")) if cited_chunk else ""
    cited_source = str(cited_chunk.get("source", "")) if cited_chunk else ""
    cited_text_id = str(cited_chunk.get("text_id", "")) if cited_chunk else ""

    _base = {
        "cited_chunk_present": cited_chunk_present,
        "cited_chunk_source":  cited_source,
        "cited_chunk_text_id": cited_text_id,
        "cited_chunk_text_len": len(cited_text) if cited_text else 0,
    }

    if not key_quote or len(str(key_quote).strip()) < 5:
        return {"outcome": "too_short", **_base, "note": "Passage too short to verify"}

    segs_strict = _quote_segments_for_matching(key_quote)
    segs_loose  = _quote_segments_for_matching_loose(key_quote)
    cited_norm_strict = normalize_for_quote_matching(cited_text) if cited_text else ""
    cited_norm_loose  = normalize_for_quote_matching_loose(cited_text) if cited_text else ""

    # Stage 1 — strict segments in cited chunk
    if cited_norm_strict and _contains_segments_in_order(cited_norm_strict, segs_strict):
        return {"outcome": "verified", "in_cited_chunk": True, "fabricated": False,
                "match_method": "strict_segments", **_base}

    # Stage 2 — loose segments in cited chunk
    if cited_norm_loose and _contains_segments_in_order(cited_norm_loose, segs_loose):
        return {"outcome": "verified", "in_cited_chunk": True, "fabricated": False,
                "match_method": "loose_segments",
                "note": "VERIFIED — punctuation-insensitive match", **_base}

    # Stage 3 — strict displacement scan
    if corpus and segs_strict:
        for cid, chunk in corpus.items():
            chunk_text = chunk.get("full_text", chunk.get("text", "")) or ""
            if not chunk_text: continue
            if _contains_segments_in_order(normalize_for_quote_matching(chunk_text), segs_strict):
                return {"outcome": "displacement", "in_cited_chunk": False, "fabricated": False,
                        "match_chunk_num": cid, "match_chunk_source": str(chunk.get("source","")),
                        "match_method": "strict_segments",
                        "note": "DISPLACEMENT — strict match in different chunk", **_base}

    # Stage 3a — loose displacement scan
    if corpus and segs_loose:
        for cid, chunk in corpus.items():
            chunk_text = chunk.get("full_text", chunk.get("text", "")) or ""
            if not chunk_text: continue
            if _contains_segments_in_order(normalize_for_quote_matching_loose(chunk_text), segs_loose):
                return {"outcome": "displacement", "in_cited_chunk": False, "fabricated": False,
                        "match_chunk_num": cid, "match_chunk_source": str(chunk.get("source","")),
                        "match_method": "loose_segments",
                        "note": "DISPLACEMENT — loose match in different chunk", **_base}

    # Stage 3.5 — hyphen/punctuation-collapsed anchor on cited chunk (corpus parsing artifacts)
    if cited_norm_loose and segs_loose:
        def _hc(text):
            return re.sub(r"[^a-z0-9]+", " ", text).strip()
        _hc_chunk = _hc(cited_norm_loose)
        _hc_anchor = _hc(segs_loose[0])[:60]
        if len(_hc_anchor) >= 20:
            if _hc_anchor in _hc_chunk:
                return {"outcome": "verified", "in_cited_chunk": True, "fabricated": False,
                        "match_method": "fuzzy_punctuation",
                        "note": "VERIFIED — hyphen-collapsed anchor in cited chunk", **_base}
            _alen = len(_hc_anchor)
            for _s in range(0, max(1, len(_hc_chunk) - _alen + 1)):
                _win = _hc_chunk[_s:_s + _alen]
                if len(_win) >= _alen - 2 and _edit_distance_short(_hc_anchor, _win) <= 3:
                    return {"outcome": "verified", "in_cited_chunk": True, "fabricated": False,
                            "match_method": "fuzzy_punctuation",
                            "note": "VERIFIED — edit-distance ≤3 anchor (corpus parsing artifact)", **_base}

    # Stage 4 — token-coverage on cited chunk
    if cited_norm_loose:
        q_loose = " ".join(segs_loose) if segs_loose else normalize_for_quote_matching_loose(key_quote)
        cov = _token_coverage(q_loose, cited_norm_loose)
        if cov >= 0.95:
            return {"outcome": "verified", "in_cited_chunk": True, "fabricated": False,
                    "match_method": "token_coverage", "approx_score": float(cov),
                    "note": "VERIFIED — near-exact token coverage in cited chunk", **_base}
        if cov >= 0.86:
            return {"outcome": "approximate_quote", "in_cited_chunk": True, "fabricated": False,
                    "match_method": "token_coverage", "approx_score": float(cov),
                    "note": "APPROXIMATE — high token coverage in cited chunk", **_base}

    # Stage 5 — token-coverage displacement scan
    if corpus and segs_loose:
        q_loose = " ".join(segs_loose)
        best = (0.0, None, "", "")
        for cid, chunk in corpus.items():
            chunk_text = chunk.get("full_text", chunk.get("text", "")) or ""
            if not chunk_text: continue
            cn = normalize_for_quote_matching_loose(chunk_text)
            cov = _token_coverage(q_loose, cn)
            if cov > best[0]:
                best = (cov, cid, str(chunk.get("text_id","")), str(chunk.get("source","")))
        if best[1] is not None and best[0] >= 0.95:
            return {"outcome": "displacement", "in_cited_chunk": False, "fabricated": False,
                    "match_chunk_num": best[1], "match_chunk_source": best[3],
                    "match_method": "token_coverage", "approx_score": float(best[0]),
                    "note": "DISPLACEMENT — near-exact token coverage in different chunk", **_base}
        if best[1] is not None and best[0] >= 0.90:
            return {"outcome": "approximate_displacement", "in_cited_chunk": False, "fabricated": False,
                    "match_chunk_num": best[1], "match_chunk_source": best[3],
                    "match_method": "token_coverage", "approx_score": float(best[0]),
                    "note": "APPROXIMATE DISPLACEMENT", **_base}

    return {"outcome": "fabrication", "in_cited_chunk": False, "fabricated": True,
            "note": "FABRICATION — passage not found in corpus", **_base}


def _search_same_document(norm_quote, text_id, lincoln_dict):
    """Legacy same-document search — retained for verify_final_answer_quotes."""
    entry = triple_lookup(lincoln_dict, text_id)
    target_source = entry.get('source', '')
    if not target_source:
        return False, ''

    def _ns(s):
        s = re.sub(r'^source:\s*', '', s.strip(), flags=re.IGNORECASE)
        return s.lower().strip()

    norm_target = _ns(target_source)
    norm_q = _norm_chunk(norm_quote)
    seen_ids = set()
    for key, candidate in lincoln_dict.items():
        if not isinstance(key, int) or key in seen_ids:
            continue
        seen_ids.add(key)
        if not isinstance(candidate, dict):
            continue
        if _ns(candidate.get('source', '')) != norm_target:
            continue
        cand_ft = candidate.get('full_text', '')
        if not cand_ft:
            continue
        norm_cand = _norm_chunk(cand_ft)
        if norm_q in norm_cand or (len(norm_q[:60]) > 20 and norm_q[:60] in norm_cand):
            return True, str(key)
        sw_found, _ = _sliding_window_verify(norm_q, norm_cand)
        if sw_found:
            return True, str(key)
    return False, ''


def verify_quote(key_quote, text_id, lincoln_dict):
    """
    Backward-compatible wrapper around _verify_quote_rich().
    Returns (verified: bool, method: str) as before.

    method values: 'exact' | 'fuzzy' | 'sliding' | 'fuzzy_punctuation' |
                   'approximate_quote' | 'token_coverage' | 'displaced' |
                   'not_found' | 'chunk_missing' | 'no_quote'

    Callers that only check the bool are unaffected.
    quote_badge_html() handles all new method strings.
    """
    if not key_quote or not key_quote.strip():
        return False, "no_quote"
    entry = triple_lookup(lincoln_dict, text_id)
    if not entry.get('full_text', ''):
        return False, "chunk_missing"

    # Build corpus dict from lincoln_dict (int-keyed entries only)
    corpus = {k: v for k, v in lincoln_dict.items() if isinstance(k, int)}
    cited_chunk = entry

    result = _verify_quote_rich(key_quote, cited_chunk, corpus)
    outcome = result.get("outcome", "fabrication")

    _OUTCOME_TO_BOOL_METHOD = {
        "verified":               (True,  result.get("match_method", "exact")),
        "approximate_quote":      (True,  "approximate_quote"),
        "displacement":           (False, "displaced"),
        "approximate_displacement":(False,"displaced"),
        "fabrication":            (False, "not_found"),
        "too_short":              (False, "no_quote"),
    }
    return _OUTCOME_TO_BOOL_METHOD.get(outcome, (False, "not_found"))


def quote_badge_html(verified, method):
    """[U8] Fully descriptive verification badge. Handles all method strings
    including benchmark-grade outcomes: fuzzy_punctuation, approximate_quote,
    token_coverage."""
    if method == "chunk_missing":
        return ('<span style="color:#6c757d;font-size:0.95em;font-weight:500;">'
                '⬜ Source chunk unavailable — quote cannot be verified</span>')
    if method == "no_quote":
        return ('<span style="color:#6c757d;font-size:0.95em;font-weight:500;">'
                '⬜ No quote provided by model</span>')
    if verified and method == "exact":
        return ('<span style="color:#155724;font-size:0.95em;font-weight:500;">'
                '✅ Quote verified — text confirmed present in Lincoln corpus</span>')
    if verified and method in ("fuzzy", "sliding", "token_coverage"):
        return ('<span style="color:#155724;font-size:0.95em;font-weight:500;">'
                '✅ Quote verified — partial or condensed match confirmed in Lincoln corpus</span>')
    if verified and method == "fuzzy_punctuation":
        return ('<span style="color:#155724;font-size:0.95em;font-weight:500;">'
                '✅ Quote verified — minor punctuation variant (corpus parsing artifact)</span>')
    if verified and method == "approximate_quote":
        return ('<span style="color:#0f5132;background:#d1e7dd;padding:1px 6px;'
                'border-radius:4px;font-size:0.95em;font-weight:600;">'
                '🟡 Approximate match — high token overlap confirms corpus presence</span>')
    if method == "displaced":
        return ('<span style="color:#664d03;background:#fff3cd;padding:1px 6px;'
                'border-radius:4px;font-size:0.95em;font-weight:600;">'
                '🔀 Quote found in document but assigned to wrong chunk — '
                'correct source, displaced chunk ID</span>')
    return ('<span style="color:#721c24;font-size:0.95em;font-weight:600;">'
            '⚠️ Quote not found in corpus — possible fabrication or out-of-corpus citation</span>')


# ── U10: FinalAnswer quote verification ──────────────────────────────────────
def extract_quoted_strings(text, min_words=7):
    """
    Extract quoted substrings from FinalAnswer text.

    Handles four quote styles Nicolay commonly uses:
      • Curly double  \u201c...\u201d  — unambiguous, matched first
      • Straight double "..."          — common in JSON output
      • Curly single  \u2018...\u2019  — unambiguous, safe to match
      • Straight single '...'          — requires care: apostrophes in words
                                         like "Lincoln's" or "it's" must not
                                         be treated as quote delimiters.
                                         Guard: opening ' must be preceded by
                                         a non-word character (lookbehind
                                         (?<!\\w)) and closing ' followed by a
                                         non-word character (lookahead (?!\\w)),
                                         so possessives and contractions are
                                         excluded.  Combined with min_words=7
                                         this makes false positives negligible.

    Only returns quotes of min_words or more to avoid matching short phrases.
    Deduplicates across patterns (curly variants of the same text won't double-count).
    """
    patterns = [
        r'\u201c([^\u201d]{20,})\u201d',       # curly double quotes (unambiguous)
        r'"([^"]{20,})"',                        # straight double quotes
        r'\u2018([^\u2019]{20,})\u2019',        # curly single quotes (unambiguous)
        r"(?<!\w)'([^']{20,})'(?!\w)",           # straight single quotes (guarded)
    ]
    seen = set()
    found = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            q = m.group(1).strip()
            if len(q.split()) >= min_words and q not in seen:
                seen.add(q)
                found.append(q)
    return found


def verify_final_answer_quotes(final_answer_text, reranked_results, lincoln_dict):
    """
    [U10] For each quoted string in FinalAnswer, check whether it appears in
    any of the reranked result chunks (not just the selected matches).

    Uses _verify_quote_rich() for each reranked chunk — same multi-stage logic
    as the benchmark verifier.  Outcome mapping:
      verified / approximate_quote / token_coverage  → verified_quotes
      displacement / approximate_displacement         → displaced_quotes
      fabrication                                     → unverified_quotes

    approximate_quote is treated as verified (not displaced) because token-coverage
    ≥ 0.86 in the cited chunk means the quote content is genuinely there — the
    difference is minor condensation, not a wrong source.

    Returns:
        verified_quotes   — list of (quote_str, text_id, source) confirmed in corpus
        displaced_quotes  — list of quote_str found in document but wrong chunk
        unverified_quotes — list of quote_str not found anywhere in corpus
    """
    quotes = extract_quoted_strings(final_answer_text)
    if not quotes:
        return [], [], []

    # Build integer-keyed corpus from lincoln_dict for _verify_quote_rich
    corpus = {k: v for k, v in lincoln_dict.items() if isinstance(k, int)}

    # Build search set from reranked results
    reranked_chunks = []
    for r in reranked_results:
        tid = str(r.get('Text ID', ''))
        entry = triple_lookup(lincoln_dict, tid)
        source = r.get('Source', '') or entry.get('source', '')
        if entry.get('full_text'):
            reranked_chunks.append((tid, source, entry))

    _VERIFIED_OUTCOMES  = {"verified", "approximate_quote"}
    _DISPLACED_OUTCOMES = {"displacement", "approximate_displacement"}

    verified, displaced, unverified = [], [], []
    for quote in quotes:
        found = False
        for tid, source, cited_chunk in reranked_chunks:
            result = _verify_quote_rich(quote, cited_chunk, corpus)
            outcome = result.get("outcome", "fabrication")
            if outcome in _VERIFIED_OUTCOMES:
                verified.append((quote, tid, source))
                found = True
                break
            if outcome in _DISPLACED_OUTCOMES:
                displaced.append(quote)
                found = True
                break
        if not found:
            unverified.append(quote)

    return verified, displaced, unverified


def check_out_of_corpus_references(references, reranked_results):
    """
    [U10] Check whether sources named in FinalAnswer.References appear in
    the reranked result set. Returns list of reference strings not found.

    Normalization strips the 'Source:' prefix common in corpus source fields,
    removes punctuation differences (periods vs commas, trailing dots), and
    compares lowercased token sets so that e.g.
      'First Annual Message, December 3, 1861'  matches
      'Source:  First Annual Message. December 3, 1861.'
    """
    if not references or not reranked_results:
        return []

    def _norm_ref(s):
        s = re.sub(r'^source:\s*', '', s.strip(), flags=re.IGNORECASE)
        s = re.sub(r'[.,;:\-]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip().lower()
        return s

    normed_sources = [_norm_ref(r.get('Source', '')) for r in reranked_results]

    out_of_corpus = []
    for ref in references:
        norm_ref = _norm_ref(ref)
        ref_tokens = set(norm_ref.split())

        matched = False
        for ns in normed_sources:
            # Exact normalized match
            if norm_ref == ns:
                matched = True
                break
            # Substring in either direction
            if norm_ref in ns or ns in norm_ref:
                matched = True
                break
            # High token overlap (≥80% of reference tokens found in source)
            src_tokens = set(ns.split())
            if ref_tokens and len(ref_tokens & src_tokens) / len(ref_tokens) >= 0.8:
                matched = True
                break

        if not matched:
            out_of_corpus.append(ref)

    return out_of_corpus


def render_final_answer_with_verification(fa_block, reranked_results, lincoln_dict,
                                           precomputed_quotes=None):
    """
    [U10] Render Nicolay's FinalAnswer with inline quote verification.
    Verified quotes annotated ✅; displaced quotes 🔀; unverified quotes ⚠️.
    Out-of-corpus references flagged in References list.

    precomputed_quotes: optional tuple (verified, displaced, unverified) from
      verify_final_answer_quotes() — supplied by Tab 1 to avoid running
      verification twice when the confidence summary panel is also shown.
    """
    fa_text = fa_block.get('Text', 'No response available')
    refs    = fa_block.get('References', [])

    if precomputed_quotes is not None:
        verified_quotes, displaced_quotes, unverified_quotes = precomputed_quotes
    else:
        verified_quotes, displaced_quotes, unverified_quotes = verify_final_answer_quotes(
            fa_text, reranked_results, lincoln_dict
        )
    out_of_corpus_refs = check_out_of_corpus_references(refs, reranked_results)

    # Annotate the text: insert inline markers after closing quote.
    # Use literal str.replace — NOT re.escape — which adds backslashes that
    # prevent str.replace from matching the literal text.
    # Try all four quote-style wrappers in priority order.
    _QUOTE_PAIRS = [
        ('\u201c', '\u201d'),   # curly double
        ('"',      '"'     ),   # straight double
        ('\u2018', '\u2019'),   # curly single
        ("'",      "'"     ),   # straight single
    ]
    annotated = fa_text
    for q, tid, source in verified_quotes:
        for open_q, close_q in _QUOTE_PAIRS:
            literal = open_q + q + close_q
            if literal in annotated:
                annotated = annotated.replace(literal, literal + ' ✅', 1)
                break
    for q in displaced_quotes:
        for open_q, close_q in _QUOTE_PAIRS:
            literal = open_q + q + close_q
            if literal in annotated:
                annotated = annotated.replace(literal, literal + ' 🔀', 1)
                break
    for q in unverified_quotes:
        for open_q, close_q in _QUOTE_PAIRS:
            literal = open_q + q + close_q
            if literal in annotated:
                annotated = annotated.replace(literal, literal + ' ⚠️', 1)
                break

    st.markdown(f"**Response:**\n{annotated}")

    # References list with out-of-corpus flags
    if refs:
        st.markdown("**References:**")
        for ref in refs:
            if ref in out_of_corpus_refs:
                st.markdown(
                    f"- {ref} "
                    f'<span style="background:#fff3cd;color:#664d03;'
                    f'padding:1px 7px;border-radius:8px;font-size:0.82em;'
                    f'font-weight:600;" title="This source was not in the '
                    f'retrieved set — Nicolay may be drawing on parametric '
                    f'memory rather than corpus text.">🔍 not in retrieved set</span>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"- {ref}")

    # Summary legend — only shown when annotations fired
    has_verified   = bool(verified_quotes)
    has_displaced  = bool(displaced_quotes)
    has_unverified = bool(unverified_quotes)
    has_ooc        = bool(out_of_corpus_refs)

    if has_verified or has_displaced or has_unverified or has_ooc:
        st.markdown("---")
        legend_parts = []
        if has_verified:
            legend_parts.append("✅ Quote verified against retrieved corpus chunk")
        if has_displaced:
            legend_parts.append(
                "🔀 Quote confirmed in document but Nicolay cited the wrong chunk ID — "
                "source is correct, chunk reference is displaced"
            )
        if has_unverified:
            legend_parts.append(
                "⚠️ Quote not found anywhere in corpus — possible fabrication or out-of-corpus citation"
            )
        if has_ooc:
            legend_parts.append(
                "🔍 Reference not in retrieved set — may draw on model's general knowledge"
            )
        st.caption(" · ".join(legend_parts))


# ── U2: Synthesis type badge ──────────────────────────────────────────────────
_SYNTH_META = {
    "1": ("#cfe2ff", "#084298", "Type 1 — Direct Retrieval",
          "A single highly relevant passage anchors the response."),
    "2": ("#d1e7dd", "#0a3622", "Type 2 — Inferential Retrieval",
          "Relevant passages retrieved; response requires some inference."),
    "3": ("#fff3cd", "#664d03", "Type 3 — Partial / Absence",
          "Retrieval partially succeeded; corpus may lack full coverage."),
    "4": ("#f8d7da", "#58151c", "Type 4 — Multi-passage Synthesis",
          "Response synthesises several passages across the corpus."),
    "5": ("#e2d9f3", "#3d0a6e", "Type 5 — Contrastive / Historiographical",
          "Response weighs competing interpretations or historiographical positions."),
}

def synthesis_type_badge(synthesis_assessment_text):
    """[U2] Parse synthesis_assessment string and return styled badge HTML + tooltip."""
    if not synthesis_assessment_text:
        return "", ""
    m = re.search(r"Type\s*([1-5])", str(synthesis_assessment_text), re.IGNORECASE)
    if not m:
        return "", synthesis_assessment_text
    key = m.group(1)
    bg, fg, label, tooltip = _SYNTH_META.get(key, ("#e2e3e5","#383d41",f"Type {key}",""))
    badge = (f'<span style="background:{bg};color:{fg};padding:3px 10px;'
             f'border-radius:12px;font-size:0.85em;font-weight:700;'
             f'border:1px solid {fg}33;" title="{tooltip}">{label}</span>')
    return badge, tooltip


# ── U9: Multi-signal diagnostic dashboard ────────────────────────────────────
def compute_diagnostic_signals(reranked_results, match_analysis,
                                synth_raw, final_answer_text):
    """
    Returns a list of (level, icon, title, detail) tuples for signals that
    fired. level ∈ {'warning', 'info'}. Empty list = no signals, clean response.

    Signals:
      S1 — Calibration gap: high reranker score but all Nicolay ratings Low.
      S2 — Low retrieval ceiling: max reranker score below weak-match threshold.
      S3 — Type 3/4 synthesis: Nicolay explicitly flagged absence or partiality.
      S4 — FinalAnswer brevity: synthesis is unusually short.

    Design note: the type-downgrade pattern (68% of Run 1 queries) is NOT
    surfaced as a signal — too frequent to be meaningful. Only explicit
    Type 3/4 classification is flagged.
    """
    signals = []

    # S1 — Calibration gap (original U3 logic)
    if reranked_results and match_analysis:
        top_score = max(r.get('Relevance Score', 0) for r in reranked_results)
        ratings = [v.get("Relevance Assessment", "").lower()
                   for v in match_analysis.values()]
        if top_score >= 0.70 and ratings and all("low" in r for r in ratings if r):
            signals.append((
                "warning", "⚠️",
                "Calibration gap detected",
                "The reranker returned high-confidence scores, but Nicolay rated "
                "all retrieved matches as Low relevance. This pattern — seen in "
                "queries where the corpus lacks the required documents — suggests "
                "the system may have retrieved plausible but ultimately off-target "
                "material. Consider rephrasing your query or checking the corpus "
                "coverage panel."
            ))

    # S2 — Low retrieval ceiling
    if reranked_results:
        top_score = max(r.get('Relevance Score', 0) for r in reranked_results)
        if top_score < 0.35:
            signals.append((
                "warning", "📉",
                "Weak retrieval signal",
                f"The highest reranker relevance score was {top_score:.3f} — below "
                "the threshold where retrieval is considered reliable. The corpus "
                "may not contain documents closely matching this query. Nicolay's "
                "synthesis is working with limited evidence."
            ))

    # S3 — Type 3 or Type 4 synthesis (explicit absence/partiality flag)
    if synth_raw:
        m = re.search(r"Type\s*([34])", str(synth_raw), re.IGNORECASE)
        if m:
            type_num = m.group(1)
            label = ("Partial retrieval" if type_num == "3"
                     else "Multi-passage synthesis — partial coverage")
            signals.append((
                "info", "🔍",
                f"Nicolay classified this as Type {type_num}: {label}",
                "Nicolay's own chain-of-thought assessment flagged this response "
                "as partial or absence-bounded. The synthesis reflects the best "
                "available evidence, but the corpus may not fully address the query."
            ))

    # S4 — FinalAnswer brevity
    if final_answer_text:
        word_count = len(final_answer_text.split())
        if word_count < 60:
            signals.append((
                "info", "📏",
                "Unusually brief response",
                f"Nicolay's synthesis is {word_count} words — shorter than typical "
                "for queries with adequate retrieval. This may indicate retrieval "
                "failure or an overly narrow query scope."
            ))

    return signals


def render_diagnostic_signals(signals):
    """[U9] Render each fired signal as its own callout. No-op if list is empty."""
    if not signals:
        return
    st.markdown("#### 🩺 Response Diagnostics")
    st.caption(
        "These indicators are heuristic signals, not ground-truth assessments. "
        "They flag patterns associated with retrieval limitations — treat them "
        "as prompts for scrutiny, not automatic distrust of the response."
    )
    for level, icon, title, detail in signals:
        if level == "warning":
            st.warning(f"**{icon} {title}**\n\n{detail}")
        else:
            st.info(f"**{icon} {title}**\n\n{detail}")


# ── U12: Response Confidence Summary ─────────────────────────────────────────
# Five independently computed signals aggregated into a scannable
# epistemological situation report.  Displayed between the synthesis type
# badge and FinalAnswer text in Tab 1.
#
# Signal inventory:
#   1. Quote verification rate          — proxies Citation Accuracy (CA)
#   2. ROUGE-1/2 corpus grounding       — proxies Factual Grounding (FA)
#   3. Reranker score spread + ceiling  — proxies Retrieval Confidence
#   4. Source diversity                 — proxies Synthesis Coverage
#   5. Complexity-type match heuristic  — proxies Historiographical Depth risk
#
# LABELLING CONSTRAINT: never say "accuracy" or "factual verification".
# ROUGE label must always read "Corpus grounding (lexical overlap …)".
# ─────────────────────────────────────────────────────────────────────────────

def extract_synth_type_num(synthesis_assessment_str):
    """
    [U12] Extract integer synthesis type (1-5) from Nicolay's
    synthesis_assessment field.  Returns 3 (Partial/Absence) as the
    conservative default when the field is absent or unparseable.
    """
    m = re.search(r'Type\s+(\d)', str(synthesis_assessment_str), re.IGNORECASE)
    return int(m.group(1)) if m else 3


# ── Signal 2: ROUGE-1/2 corpus grounding ─────────────────────────────────────

def compute_corpus_grounding(final_answer_text, reranked_results, top_n=3):
    """
    [U12] Compute ROUGE-1/2 F-measure between FinalAnswer text and the
    concatenated full text of the top-N reranked chunks.

    High score  → response tracks retrieved text.
    Low score   → response diverged (legitimate inference OR hallucination;
                  always read alongside synthesis type).

    Returns dict {'rouge1': float, 'rouge2': float} or None if inputs missing.
    """
    if not final_answer_text or not reranked_results:
        return None
    if not _ROUGE_AVAILABLE:
        return None

    ref_chunks = reranked_results[:top_n]
    reference_text = " ".join(
        r.get('Key Quote', '') or r.get('full_text', '') or r.get('Full Text', '')
        for r in ref_chunks
    ).strip()
    if not reference_text:
        return None

    scorer = _rouge_scorer_mod.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    scores = scorer.score(reference_text, final_answer_text)
    return {
        'rouge1': round(scores['rouge1'].fmeasure, 2),
        'rouge2': round(scores['rouge2'].fmeasure, 2),
    }


def interpret_rouge(r1, synth_type_num):
    """
    [U12] Context-sensitive interpretation of ROUGE-1 score.

    A Type 3 (Absence) response legitimately has low ROUGE.
    A Type 1 (Direct Retrieval) response with low ROUGE is a red flag.
    Types 4/5 (Synthesis/Contrastive) warrant moderate thresholds.
    """
    if synth_type_num in (1, 2):
        if r1 >= 0.45:
            return "Closely tracks retrieved sources ✅"
        elif r1 >= 0.25:
            return "Moderate grounding — some divergence from retrieved text"
        else:
            return "Low grounding for direct retrieval type ⚠️"
    elif synth_type_num == 3:
        return "Partial grounding expected for absence-bounded response"
    else:  # Types 4-5
        if r1 >= 0.30:
            return "Good synthesis grounding ✅"
        else:
            return "Low grounding — verify response independently ⚠️"


# ── Signals 3 + 4: Reranker score analysis + source diversity ─────────────────

def analyze_reranker_scores(reranked_results):
    """
    [U12] Derive retrieval confidence signals from reranker output.

    Sub-signal A — Score ceiling (max score): already partially used in S2.
    Sub-signal B — Score spread (max - min): flat spread predicts synthesis
      difficulty and the type-downgrade pattern documented in Run 1 (68%).
    Sub-signal C — Calibration decoupling: high ceiling + flat spread is the
      Q11 failure signature.  Refactors and supersedes the S1 logic in
      compute_diagnostic_signals(); that function retains its own check so
      the U9 panel continues to work unchanged.

    Returns dict or None.
    """
    if not reranked_results:
        return None

    scores = [r.get('Relevance Score', r.get('relevance_score', 0))
              for r in reranked_results[:5]]
    scores = [s for s in scores if s is not None]
    if not scores:
        return None

    sources = [
        r.get('Source', '') or r.get('source', '')
        for r in reranked_results[:5]
    ]
    n_distinct = len(set(s for s in sources if s))

    max_s  = max(scores)
    min_s  = min(scores)
    spread = round(max_s - min_s, 3)

    # High ceiling + flat spread = reranker confident but undiscriminating
    calibration_warning = (max_s >= 0.70 and spread <= 0.08)

    return {
        'max_score':          round(max_s, 3),
        'min_score':          round(min_s, 3),
        'spread':             spread,
        'n_distinct_sources': n_distinct,
        'calibration_warning': calibration_warning,
    }


def interpret_spread(spread, synth_type_num=None):
    """
    [U12] Human-readable retrieval differentiation label.
    For Types 4/5 (multi-passage synthesis), distributed retrieval is expected
    and healthy — flat spread is not a warning in that context.
    """
    if synth_type_num in (4, 5):
        if spread >= 0.20:
            return "Wide spread — one passage dominates synthesis"
        else:
            return "Distributed retrieval — consistent with multi-passage synthesis ✅"
    # Types 1–3: differentiation matters
    if spread >= 0.20:
        return "Differentiated retrieval ✅"
    elif spread >= 0.10:
        return "Moderate differentiation"
    elif spread >= 0.05:
        return "Low differentiation — verify coverage"
    else:
        return "Flat distribution ⚠️ — no passage clearly dominant"


# ── Signal 5: Complexity-type match heuristic ─────────────────────────────────

_COMPLEXITY_PATTERNS = [
    # Multi-part / contrastive queries
    (re.compile(
        r'\b(compare|contrast|how did .+ change|evolution of|shift|'
        r'differ|difference|between|versus|vs\.?)\b',
        re.IGNORECASE
    ), "high"),
    # Synthesis / across-time queries
    (re.compile(
        r'\b(throughout|across|over time|consistent|inconsistent|'
        r'develop|develop\w+|arc|trajectory)\b',
        re.IGNORECASE
    ), "high"),
    # Simple single-fact lookups
    (re.compile(
        r'\b(what did|when did|who|where|which speech|did lincoln)\b',
        re.IGNORECASE
    ), "low"),
]


def estimate_query_complexity(query_text):
    """
    [U12] Heuristic complexity estimate for user query.
    Returns 'high', 'low', or 'moderate'.
    """
    if not query_text:
        return "moderate"
    for pattern, level in _COMPLEXITY_PATTERNS:
        if pattern.search(query_text):
            return level
    return "moderate"


def check_complexity_match(query_text, synth_type_num):
    """
    [U12] Compare estimated query complexity against Nicolay's synthesis type.

    Types 4/5 are by definition multi-passage responses; the synthesis
    classification itself is sufficient evidence of appropriate complexity
    handling, so we treat them as a match regardless of the query heuristic.
    The heuristic is only applied for Types 1–3.

    Returns (matched: bool, message: str).
    """
    # For synthesis/contrastive responses, trust the model's own classification
    if synth_type_num in (4, 5):
        return True, "Multi-passage or contrastive synthesis — type consistent with complex query ✅"

    complexity = estimate_query_complexity(query_text)
    if synth_type_num in (1, 2) and complexity == "high":
        return False, "⚠️ Query appears complex but response is Type 1/2 — possible under-synthesis (heuristic)"
    else:
        return True, "Query complexity consistent with synthesis type"


# ── Rendering ─────────────────────────────────────────────────────────────────

def _compute_overall_confidence(
    verified_quotes, displaced_quotes, unverified_quotes,
    rouge_data, reranker_data, synth_type_num
):
    """
    [U12] Derive an overall confidence rating (high / medium / low) and a
    plain-language explanation from the five signal values.

    Low  — any unverified quotes, OR calibration warning, OR ROUGE < 0.25
           for Type 1/2 (direct retrieval with weak grounding).
    High — type-specific:
           Types 1/2: no unverified quotes, no calib warning, ROUGE ≥ 0.45
           Types 4/5: no unverified quotes, no calib warning, ROUGE ≥ 0.30
           Type 3:    no unverified quotes, no calib warning
                      (ROUGE thresholds don't apply — low ROUGE is expected
                      when the response correctly reports corpus absence)
    Medium — everything else.

    Returns (rating: str, icon: str, color: str, explanation: str).
    """
    has_unverified    = len(unverified_quotes) > 0
    calib_warning     = reranker_data.get('calibration_warning', False) if reranker_data else False
    r1                = rouge_data.get('rouge1', 0.0) if rouge_data else 0.0
    direct_type       = synth_type_num in (1, 2)
    absence_type      = synth_type_num == 3
    synthesis_type    = synth_type_num in (4, 5)

    # ── Low conditions ────────────────────────────────────────────────────────
    if has_unverified:
        return ("low", "🔴", "#f8d7da",
                "One or more quotes in this response could not be verified "
                "against the Lincoln corpus. This may indicate fabricated or "
                "misremembered quotations. Check the sources before relying on "
                "any quoted text.")
    if calib_warning:
        return ("low", "🔴", "#f8d7da",
                "The retrieval system returned high confidence scores across all "
                "results with little discrimination between them. This pattern "
                "often appears when the corpus doesn't contain the right documents "
                "for the query. The response may sound plausible but rest on "
                "off-target sources.")
    if direct_type and r1 < 0.25:
        return ("low", "🔴", "#f8d7da",
                "This response is classified as a direct retrieval (Type 1/2) but "
                "shows weak alignment with the retrieved Lincoln texts. The answer "
                "may rely more on the model's general knowledge than on corpus "
                "evidence. Treat quoted material with particular caution.")

    # ── High conditions ───────────────────────────────────────────────────────

    # Type 3 — absence/partial: ROUGE thresholds do not apply.
    # A well-executed absence response correctly reports what the corpus
    # lacks; low ROUGE is expected and healthy, not a warning sign.
    if absence_type and not has_unverified and not calib_warning:
        return ("high", "✅", "#d1e7dd",
                "Nicolay correctly identified the limits of the corpus on this "
                "query. The response is transparent about what the Lincoln texts "
                "don't directly address, and any quoted passages are verified. "
                "This is the expected, well-calibrated behavior for a question "
                "that falls outside or at the edges of the corpus's scope.")

    # Types 1/2 and 4/5 — ROUGE threshold required
    rouge_threshold = 0.45 if direct_type else 0.30
    rouge_ok        = r1 >= rouge_threshold
    if not has_unverified and not calib_warning and rouge_ok:
        if synthesis_type:
            return ("high", "✅", "#d1e7dd",
                    "The response draws on multiple relevant Lincoln passages, "
                    "all quotes are verified, and the answer closely tracks the "
                    "retrieved source texts. This is a well-grounded synthesis.")
        else:
            return ("high", "✅", "#d1e7dd",
                    "All quotes verified against the corpus and the response closely "
                    "tracks the retrieved Lincoln texts. This response is well "
                    "grounded in the primary sources.")

    # ── Medium (everything else) ──────────────────────────────────────────────
    parts = []
    if len(displaced_quotes) > 0:
        parts.append("one or more quotes appear in the right document but "
                     "were drawn from a different passage than cited")
    if absence_type:
        parts.append("retrieval was partial — the corpus may contain related "
                     "material but Nicolay flagged incomplete coverage")
    elif direct_type and 0.25 <= r1 < 0.45:
        parts.append("the response only partially tracks the retrieved text — "
                     "some content may go beyond what the corpus directly supports")
    if not parts:
        parts.append("signals are mixed or inconclusive")
    explanation = ("Some caution is warranted. Specifically: "
                   + "; ".join(parts) + ". "
                   "Review the source passages before relying on specific claims.")
    return ("medium", "⚠️", "#fff3cd", explanation)


def render_confidence_summary(
    final_answer_text,
    reranked_results,
    verified_quotes,
    displaced_quotes,
    unverified_quotes,
    synth_type_num,
    query_text,
):
    """
    [U12] Render the Response Confidence Summary panel.

    Collapsed by default; header shows overall confidence rating so the
    researcher gets an instant read without opening the panel.  When opened,
    the top section explains the rating in plain language; the five diagnostic
    signals follow with accessible explanations of what each value means.

    Call after verify_final_answer_quotes() so quote lists are ready.
    """
    # Pre-compute signal data so the overall rating can be derived first
    rouge_data    = compute_corpus_grounding(final_answer_text, reranked_results)
    reranker_data = analyze_reranker_scores(reranked_results)

    rating, icon, _color, explanation = _compute_overall_confidence(
        verified_quotes, displaced_quotes, unverified_quotes,
        rouge_data, reranker_data, synth_type_num
    )
    rating_label = rating.capitalize()

    with st.expander(f"📊 Response Confidence: {icon} {rating_label}", expanded=False):

        # ── Overall rating banner ─────────────────────────────────────────────
        st.markdown(f"**Overall confidence: {icon} {rating_label}**")
        st.markdown(explanation)
        st.divider()

        # ── Signal 1: Quote verification ──────────────────────────────────────
        total_q = len(verified_quotes) + len(displaced_quotes) + len(unverified_quotes)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.caption("**Quote verification**")
        with col2:
            if total_q == 0:
                st.markdown("⬜ No direct quotes in this response")
                st.caption("Nicolay did not include any directly quoted text.")
            elif len(unverified_quotes) == 0 and len(displaced_quotes) == 0:
                st.markdown(f"✅ {total_q}/{total_q} quotes verified")
                st.caption("Every quoted passage was confirmed present in the Lincoln corpus.")
            elif len(unverified_quotes) > 0:
                st.markdown(
                    f"🔴 {len(verified_quotes)}/{total_q} verified — "
                    f"**{len(unverified_quotes)} not found in corpus**"
                )
                st.caption(
                    "One or more quoted passages could not be located anywhere "
                    "in the Lincoln corpus. These may be fabricated, misremembered, "
                    "or drawn from a source outside the collection."
                )
            else:
                st.markdown(
                    f"🔀 {len(verified_quotes)}/{total_q} verified — "
                    f"**{len(displaced_quotes)} displaced**"
                )
                st.caption(
                    "The displaced quote appears in the correct Lincoln document "
                    "but was drawn from a different passage than the one Nicolay "
                    "cited. The source is right; the specific chunk reference is off."
                )

        st.divider()

        # ── Signal 2: Corpus grounding ────────────────────────────────────────
        col1, col2 = st.columns([1, 2])
        with col1:
            st.caption("**How closely does the response track the sources?**")
        with col2:
            if rouge_data:
                r1_val = rouge_data['rouge1']
                r2_val = rouge_data['rouge2']
                st.markdown(
                    f"ROUGE-1: **{r1_val}** &nbsp;&nbsp; ROUGE-2: **{r2_val}**",
                    unsafe_allow_html=True,
                )
                st.caption(interpret_rouge(r1_val, synth_type_num))
                st.caption(
                    "These scores (range 0.0–1.0) measure how much of the response "
                    "uses words and phrases drawn from the retrieved Lincoln passages. "
                    "ROUGE-1 counts individual word overlap; ROUGE-2 counts "
                    "two-word phrase overlap — a stricter test. "
                    "As a rough guide: **above 0.45** suggests the response stays "
                    "close to the retrieved text; **0.25–0.45** indicates moderate "
                    "grounding with some inference beyond the sources; **below 0.25** "
                    "means the response has diverged significantly — which is expected "
                    "and correct for absence responses, but a caution flag for "
                    "direct-retrieval ones."
                )
            else:
                st.markdown("⬜ Not computed")

        st.divider()

        # ── Signals 3 + 4: Retrieval quality ─────────────────────────────────
        col1, col2 = st.columns([1, 2])
        with col1:
            st.caption("**Retrieval quality**")
        with col2:
            if reranker_data:
                st.markdown(
                    f"**{reranker_data['n_distinct_sources']}** distinct Lincoln "
                    f"document(s) retrieved"
                )
                spread_interp = interpret_spread(reranker_data['spread'], synth_type_num)
                st.markdown(
                    f"Score spread: **{reranker_data['spread']}** — {spread_interp}"
                )
                st.caption(
                    "The score spread (range 0.0–1.0) is the gap between the "
                    "highest- and lowest-scoring retrieved passages. "
                    "**Above ~0.20**: one passage clearly stood out as the best "
                    "match. **0.05–0.20**: several passages scored similarly — "
                    "normal for synthesis queries where multiple documents are "
                    "relevant, but worth noting for simpler questions where one "
                    "passage should dominate. **Near 0.0 with a high max score** "
                    "is the most concerning pattern: the system is confident but "
                    "not discriminating, which can indicate the corpus lacks the "
                    "right documents for this query."
                )
                if reranker_data['calibration_warning']:
                    st.warning(
                        "⚠️ **High confidence, low discrimination.** The retrieval "
                        "system returned high scores for all passages with little "
                        "difference between them. This sometimes happens when the "
                        "corpus doesn't contain exactly the right documents — the "
                        "system retrieves the closest available material, which may "
                        "not fully address the question. Consider rephrasing the "
                        "query or checking the Corpus Coverage panel."
                    )
            else:
                st.markdown("⬜ Not computed")

        st.divider()

        # ── Signal 5: Response depth ──────────────────────────────────────────
        col1, col2 = st.columns([1, 2])
        with col1:
            st.caption("**Response depth**")
        with col2:
            complexity_matched, complexity_msg = check_complexity_match(
                query_text, synth_type_num
            )
            st.markdown(complexity_msg)
            st.caption(
                "This check compares the apparent complexity of your question "
                "against the type of response Nicolay produced (Type 1–5). "
                "Questions with comparative or synthesis language — words like "
                "'compare,' 'how did Lincoln's position change,' 'throughout his "
                "career,' 'consistent' — are flagged as high-complexity. "
                "A mismatch (complex question, simple response) may mean the "
                "corpus lacks enough relevant material, or that rephrasing the "
                "query could surface more evidence. This is a heuristic estimate, "
                "not a definitive judgment."
            )
            if not complexity_matched:
                st.warning(
                    "The query appears to ask for a complex, multi-source answer, "
                    "but Nicolay produced a simpler response. The corpus may not "
                    "have enough relevant material, or rephrasing the query may "
                    "surface more evidence."
                )

        # ── Footer ─────────────────────────────────────────────────────────────
        st.divider()
        st.caption(
            "These indicators are based on what the system can check automatically "
            "— not a guarantee of accuracy. 'Source tracking' shows how closely "
            "the response draws on the retrieved Lincoln texts. Quote verification "
            "checks whether direct quotations actually appear in the corpus. "
            "Some signals are rough estimates. Always read the source passages "
            "yourself before relying on any quoted claim."
        )


# ── Legacy shim — kept so any surviving call sites don't crash ────────────────
def calibration_decoupling_warning(reranked_results, match_analysis):
    """Deprecated in v1.4 — use compute_diagnostic_signals instead."""
    if not reranked_results or not match_analysis:
        return False
    top_score = max(r.get('Relevance Score', 0) for r in reranked_results)
    if top_score < 0.70:
        return False
    ratings = [v.get("Relevance Assessment", "").lower()
               for v in match_analysis.values()]
    return all("low" in r for r in ratings if r)


# ── Card grid helpers ─────────────────────────────────────────────────────────
def relevance_badge_html(relevance_text):
    t = (relevance_text or "").lower()
    if "high"   in t: bg, fg = "#d4edda", "#155724"
    elif "medium" in t or "moderate" in t: bg, fg = "#fff3cd", "#856404"
    elif "low"  in t: bg, fg = "#f8d7da", "#721c24"
    else:             bg, fg = "#e2e3e5", "#383d41"
    label = re.split(r"[—,]", relevance_text or "N/A")[0].strip()[:30]
    return (f'<span style="background:{bg};color:{fg};padding:2px 10px;'
            f'border-radius:12px;font-size:0.8em;font-weight:700;">{label}</span>')


# ── Match Analysis card helpers ──────────────────────────────────────────────
# Layout strategy (informed by Streamlit docs 2025):
#   - st.container(border=True)  → card boundary, native, no CSS class needed
#   - right-flush badge via text-align:right on column div (version-safe)
#   - st.markdown blockquote (> *text*) → quote, immune to corpus special chars
#   - st.markdown for all body text; st.caption only for de-emphasised metadata
#   - unsafe_allow_html ONLY for badge <span>s which contain zero corpus text

def _clean_source(s: str) -> str:
    """Strip leading 'Source: ' prefix that some corpus entries include."""
    s = s.strip()
    if s.lower().startswith("source:"):
        s = s[len("source:"):].strip()
    return s


def relevance_badge_html(relevance_text):
    t = (relevance_text or "").lower()
    if "high" in t:
        bg, fg = "#d4edda", "#155724"
    elif "medium" in t or "moderate" in t:
        bg, fg = "#fff3cd", "#856404"
    elif "low" in t:
        bg, fg = "#f8d7da", "#721c24"
    else:
        bg, fg = "#e2e3e5", "#383d41"
    label = re.split(r"[\u2014,]", relevance_text or "N/A")[0].strip()[:30]
    return (f'<span style="background:{bg};color:{fg};padding:4px 12px;'
            f'border-radius:12px;font-size:0.85em;font-weight:700;'
            f'display:inline-block;">{label}</span>')


def render_match_analysis_cards(match_analysis, lincoln_dict, reranked_results=None):
    """
    Match Analysis cards — native Streamlit layout, no corpus text in HTML.

    Header:  [match_key bold]  [badge right-aligned via horizontal container]
    Body:    ID · Source (de-duplicated prefix) · reranker score
             blockquote (markdown > syntax, safe for any text)
             Nicolay's Analysis: [full relevance assessment text]
             verification badge
    Footer:  expander for full-text with highlight
    """
    st.markdown("### Match Analysis")

    score_map = {}
    if reranked_results:
        for r in reranked_results:
            score_map[str(r.get('Text ID', ''))] = r.get('Relevance Score', 0.0)

    col_left, col_right = st.columns(2)
    col_map = {0: col_left, 1: col_right}

    for i, (match_key, info) in enumerate(match_analysis.items()):
        text_id   = str(info.get("Text ID", ""))
        source    = _clean_source(info.get("Source", ""))
        key_quote = info.get("Key Quote", "")
        summary   = info.get("Summary", "")
        relevance = info.get("Relevance Assessment", "")
        hist_ctx  = info.get("Historical Context", "")
        reranker_score = score_map.get(text_id, None)

        verified, method = verify_quote(key_quote, text_id, lincoln_dict)
        rel_badge  = relevance_badge_html(relevance)
        badge_html = quote_badge_html(verified, method)

        with col_map[i % 2]:
            with st.container(border=True):

                # ── Header: title left, badge right ──────────────────────────
                # Use asymmetric columns. Badge column uses horizontal container
                # with right alignment — native API, no CSS hack needed.
                h_title, h_badge = st.columns([5, 1])
                with h_title:
                    st.markdown(f"**{match_key}**")
                with h_badge:
                    # Right-align via text-align on the column div.
                    # st.container(horizontal=True) requires Streamlit >= 1.41;
                    # inline CSS is version-safe and works on Streamlit Cloud.
                    st.markdown(
                        f'<div style="text-align:right;">{rel_badge}</div>',
                        unsafe_allow_html=True
                    )

                # ── Metadata line ─────────────────────────────────────────────
                st.markdown(f"**ID:** {text_id}  ·  **Source:** {source}")
                if reranker_score is not None:
                    st.caption(f"Reranker score: {reranker_score:.3f}")

                # ── Quote (markdown blockquote — safe for all corpus text) ────
                if key_quote:
                    q_display = key_quote[:320] + ("\u2026" if len(key_quote) > 320 else "")
                    st.markdown(f'> *\u201c{q_display}\u201d*')

                # ── Nicolay's Analysis (body text size, not caption) ──────────
                if summary:
                    analysis_text = summary[:300] + ("\u2026" if len(summary) > 300 else "")
                    st.markdown(f"**Nicolay's Analysis:** {analysis_text}")

                # ── Verification badge ────────────────────────────────────────
                st.markdown(badge_html, unsafe_allow_html=True)

                # ── Full text expander ────────────────────────────────────────
                with st.expander(f"Full text & highlight \u2014 {match_key}", expanded=False):
                    if hist_ctx:
                        st.markdown(f"**Historical Context:** {hist_ctx}")
                    entry = triple_lookup(lincoln_dict, text_id)
                    if entry:
                        ft = entry.get('full_text', '')
                        html_ft = ft.replace("\\n", "<br>")
                        if key_quote and key_quote in ft:
                            html_ft = html_ft.replace(key_quote, f"<mark>{key_quote}</mark>")
                        elif key_quote:
                            html_ft = highlight_key_quote(ft, key_quote).replace("\\n", "<br>")
                        st.markdown("**Full Text with Highlighted Quote:**")
                        st.markdown(
                            f'<div style="font-size:0.95em;line-height:1.65;">{html_ft}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.info("Full text not found in corpus for this text ID.")

# ── U6: Retrieval diagnostics panel ──────────────────────────────────────────
def render_retrieval_diagnostics(reranked_results, match_analysis):
    """
    [U6] Compact table: search method, text ID, source, reranker score,
    and whether the chunk made it into Nicolay's final Match Analysis.
    """
    if not reranked_results:
        return

    def _norm_id(raw):
        """Extract bare integer string from any text_id format."""
        m = re.search(r'(\d+)', str(raw))
        return m.group(1) if m else str(raw)

    # Normalise IDs from Nicolay CoT (bare ints) and reranked results (any format)
    used_ids = {_norm_id(v.get("Text ID", "")) for v in match_analysis.values()}
    rows = []
    for r in reranked_results:
        tid = str(r.get('Text ID', ''))
        rows.append({
            "Rank":           r.get('Rank', ''),
            "Search Type":    r.get('Search Type', ''),
            "Text ID":        tid,
            "Source":         r.get('Source', '')[:55] + ("…" if len(r.get('Source','')) > 55 else ""),
            "Reranker Score": f"{r.get('Relevance Score', 0.0):.3f}",
            "Used by Nicolay": "✅ Yes" if _norm_id(tid) in used_ids else "—",
        })
    diag_df = pd.DataFrame(rows)
    with st.expander("🔬 Retrieval Diagnostics — pipeline transparency", expanded=False):
        st.caption(
            "This table shows every document passed to Cohere reranking, "
            "its relevance score, and whether Nicolay selected it as a named match. "
            "High reranker scores with no Nicolay selection may indicate "
            "retrieval-synthesis decoupling (see Q11/RC-5 analysis)."
        )
        st.dataframe(diag_df, use_container_width=True, hide_index=True)


# ── U1: Hay keyword pills ─────────────────────────────────────────────────────
def render_keyword_pills(weighted_keywords, query_assessment):
    """[U1] Top weighted keywords as styled pills + query type label."""
    if not weighted_keywords:
        return
    # Sort by weight descending, show top 6
    sorted_kw = sorted(weighted_keywords.items(), key=lambda x: x[1], reverse=True)[:6]
    max_w = max(w for _, w in sorted_kw) or 1

    pills_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;margin:10px 0;">'
    for kw, w in sorted_kw:
        intensity = int(40 + 200 * (w / max_w))   # darker = higher weight
        bg   = f"rgb({255-intensity//3},{255-intensity//2},{255})"
        fg   = "#1a1a2e" if intensity < 160 else "#ffffff"
        size = 0.85 + 0.25 * (w / max_w)
        pills_html += (
            f'<span style="background:{bg};color:{fg};padding:4px 12px;'
            f'border-radius:14px;font-size:{size:.2f}em;font-weight:600;'
            f'border:1px solid #ccc;" title="Weight: {w}">{kw}</span>'
        )
    pills_html += "</div>"

    if query_assessment:
        # Extract query type if present (e.g. "Inferential Retrieval")
        m = re.search(
            r"(Direct|Inferential|Absence\s*Recognition|Absence|"
            r"Multi.passage|Multi-passage|Contrastive|Historiographical|Temporal)",
            query_assessment, re.IGNORECASE
        )
        qtype = m.group(1) if m else None
        if qtype:
            pills_html += (f'<div style="font-size:0.88em;color:#6c757d;margin-bottom:4px;">'
                           f'Query type detected by Hay: <strong>{qtype}</strong></div>')

    st.markdown("**Hay's keyword steering:**", unsafe_allow_html=False)
    st.markdown(pills_html, unsafe_allow_html=True)


# ── U4: Session query history (sidebar) ──────────────────────────────────────
def render_sidebar_history():
    """[U4] Sidebar: query history + corpus coverage notice."""
    with st.sidebar:
        # ── U5: Corpus coverage notice ────────────────────────────────────────
        st.markdown("### 📚 Corpus Coverage")
        st.markdown(
            "The Lincoln corpus contains **772 chunks** from major speeches, "
            "messages to Congress, and the Lincoln-Douglas Debates.\n\n"
            "**Known gaps** (queries on these will have limited retrieval):\n"
            "- Last Public Address (Apr 11, 1865) — *absent*\n"
            "- Letter to Horace Greeley (Aug 22, 1862) — *absent*\n"
            "- Trent Affair correspondence — *absent*"
        )
        st.divider()

        # ── Query history ─────────────────────────────────────────────────────
        st.markdown("### 🕑 Query History")
        history = st.session_state.get("query_history", [])
        if not history:
            st.caption("No queries yet this session.")
        else:
            for entry in reversed(history):
                synth_label = entry.get("synth_label", "")
                ts          = entry.get("timestamp", "")
                qtext       = entry.get("query", "")
                badge_color = entry.get("synth_color", "#e2e3e5")
                with st.expander(f"🔍 {qtext[:45]}{'…' if len(qtext)>45 else ''}", expanded=False):
                    st.caption(f"**Time:** {ts}")
                    if synth_label:
                        st.markdown(
                            f'<span style="background:{badge_color};padding:2px 8px;'
                            f'border-radius:10px;font-size:0.8em;">{synth_label}</span>',
                            unsafe_allow_html=True
                        )
                    fa_text = entry.get("final_answer", "")
                    if fa_text:
                        st.markdown(f"**Response preview:**\n{fa_text[:300]}{'…' if len(fa_text)>300 else ''}")
                    refs = entry.get("references", [])
                    if refs:
                        st.markdown("**References:** " + ", ".join(refs))


def record_history(query, model_output, synth_type_key):
    """Append a query result to session history for sidebar display."""
    fa       = model_output.get("FinalAnswer", {})
    bg_color = _SYNTH_META.get(synth_type_key, ("","","",""))[0] or "#e2e3e5"
    label    = _SYNTH_META.get(synth_type_key, ("", "", f"Type {synth_type_key}", ""))[2]
    st.session_state["query_history"].append({
        "timestamp":   dt.now().strftime("%H:%M:%S"),
        "query":       query,
        "final_answer": fa.get("Text", ""),
        "references":  fa.get("References", []),
        "synth_label": label,
        "synth_color": bg_color,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════════════════════
render_sidebar_history()

st.title("Exploring RAG with Nicolay and Hay")
image_url = 'http://danielhutchinson.org/wp-content/uploads/2024/01/nicolay_hay.png'
st.image(image_url, width=600)

st.subheader("**Navigating this App:**")
st.write("Expand the **How It Works?** box below for a walkthrough of the app. "
         "Continue to the search interface below to begin exploring Lincoln's speeches.")
with st.expander("**How It Works - Exploring RAG with Hay and Nicolay**"):
    st.write(app_intro)

# ── Search form ───────────────────────────────────────────────────────────────
with st.form("Search Interface"):
    st.markdown("Enter your query below:")
    user_query = st.text_input("Query")

    st.write("**Search Options:**")
    st.write("At least one search method must be selected to perform Response and Analysis.")
    perform_keyword_search  = st.toggle("Weighted Keyword Search",  value=True)
    perform_semantic_search = st.toggle("Semantic Search",           value=True)
    perform_reranking       = st.toggle("Response and Analysis",     value=True, key="reranking")

    if perform_reranking and not (perform_keyword_search or perform_semantic_search):
        st.warning("Response & Analysis requires at least one search method.")

    with st.expander("Additional Search Options (In Development)"):
        st.markdown("Override Hay's keyword suggestions with your own below.")
        st.markdown("**Weighted Keywords**")
        user_weighted_keywords = {}
        for i in range(1, 6):
            c1, c2 = st.columns(2)
            with c1:
                kw = st.text_input(f"Keyword {i}", key=f"keyword_{i}")
            with c2:
                wt = st.number_input(f"Weight {i}", min_value=0.0, value=1.0,
                                     step=0.1, key=f"weight_{i}")
            if kw:
                user_weighted_keywords[kw] = wt

        st.header("Year and Text Filters")
        user_year_keywords = st.text_input(
            "Year Keywords (comma-separated, e.g. 1861, 1862)"
        )
        user_text_keywords = st.multiselect("Text Selection:", [
            'At Peoria, Illinois', 'A House Divided', 'Eulogy on Henry Clay',
            'Farewell Address', 'Cooper Union Address', 'First Inaugural Address',
            'Second Inaugural Address', 'July 4th Message to Congress',
            'First Annual Message', 'Second Annual Message', 'Third Annual Message',
            'Fourth Annual Message', 'Emancipation Proclamation',
            'Public Letter to James Conkling', 'Gettysburg Address'
        ])

    submitted = st.form_submit_button("Submit")

# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
if submitted:
    if not (perform_keyword_search or perform_semantic_search):
        st.error("Please enable at least one search method.")
        st.stop()

    st.subheader("Starting RAG Process (takes about 30–60 seconds in total)")

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading Lincoln corpus and embeddings…"):
        lincoln_data_df  = load_lincoln_speech_corpus()
        voyant_data_df   = load_voyant_word_counts()
        lincoln_index_df = load_lincoln_index_embedded()

    lincoln_data = lincoln_data_df.to_dict("records")

    # Triple-keyed lookup dict
    lincoln_dict = {}
    for item in lincoln_data:
        tid = item.get("text_id", "")
        lincoln_dict[tid] = item
        m = re.search(r"(\d+)", str(tid))
        if m:
            n = int(m.group(1))
            lincoln_dict[n]       = item
            lincoln_dict[str(n)]  = item

    # Corpus terms
    if not voyant_data_df.empty and "corpusTerms" in voyant_data_df.columns:
        raw = voyant_data_df.at[0, "corpusTerms"]
        obj = json.loads(raw) if isinstance(raw, str) else raw
        corpus_terms = obj.get("terms", [])
    else:
        corpus_terms = []
        st.warning("Voyant corpus terms not found — keyword search limited.")

    # Parse embeddings
    def _parse_emb(x):
        if isinstance(x, (list, np.ndarray)):
            return np.array(x, dtype=float)
        if isinstance(x, str):
            try:
                return np.array(list(map(float, x.strip("[]").split(","))), dtype=float)
            except Exception:
                return np.zeros(1536)
        return np.zeros(1536)

    lincoln_index_df = lincoln_index_df.copy()
    lincoln_index_df["embedding"] = lincoln_index_df["embedding"].apply(_parse_emb)

    def get_src_sum(tid):
        e = lincoln_dict.get(tid, {})
        if not e:
            m = re.search(r"(\d+)", str(tid))
            if m:
                e = lincoln_dict.get(int(m.group(1)), {})
        return e.get("source", ""), e.get("summary", "")

    lincoln_index_df["source"], lincoln_index_df["summary"] = zip(
        *lincoln_index_df["text_id"].apply(get_src_sum)
    )

    if not user_query:
        st.warning("Please enter a query.")
        st.stop()

    # ── Hay model call ────────────────────────────────────────────────────────
    with st.spinner("Hay is analysing your query…"):
        hay_resp = client.chat.completions.create(
            # Hay model v.3
            #model="ft:gpt-4.1-mini-2025-04-14:personal:hays-v3:DEcb9s4u",
            # Hay model v.4
            model = "ft:gpt-4.1-mini-2025-04-14:personal:hays-v4:DI4PJ4Zt",
            messages=[{"role":"system","content":keyword_prompt},
                      {"role":"user","content":user_query}],
            temperature=0, max_tokens=800, top_p=1,
            frequency_penalty=0, presence_penalty=0
        )
    hay_raw = hay_resp.choices[0].message.content

    def _parse_model_json(raw: str, label: str) -> dict:
        """
        Robustly parse JSON from a model response.
        Handles: markdown fences, BOM, stray trailing text, and common
        single-character escaping issues that cause column-offset errors.
        Shows a clear st.error with the raw output on failure so the problem
        is diagnosable without Streamlit log access.
        """
        if not raw:
            st.error(f"{label}: model returned an empty response.")
            return {}

        s = raw.strip()

        # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()

        # Remove UTF-8 BOM if present
        s = s.lstrip("\ufeff")

        # Attempt 1: direct parse
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass

        # Attempt 2: extract the first {...} block (handles trailing text / preamble)
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        # Attempt 3: truncate at last valid closing brace
        last_brace = s.rfind("}")
        if last_brace != -1:
            try:
                return json.loads(s[:last_brace + 1])
            except json.JSONDecodeError:
                pass

        # All attempts failed — surface raw output for diagnosis
        st.error(
            f"**{label}: could not parse model JSON output.**\n\n"
            f"This usually means the model produced malformed JSON "
            f"(e.g. a truncated field at max_tokens, or an unescaped character). "
            f"Raw output is shown below for diagnosis."
        )
        with st.expander(f"Raw {label} output (for diagnosis)", expanded=True):
            st.code(raw, language="text")
        return {}

    hay_data = _parse_model_json(hay_raw, "Hay v3")
    if not hay_data:
        st.stop()

    initial_answer          = hay_data.get('initial_answer', '')
    model_weighted_keywords = hay_data.get('weighted_keywords', {})
    model_year_keywords     = hay_data.get('year_keywords', []) or []
    model_text_keywords     = hay_data.get('text_keywords', []) or []
    model_query_assessment  = hay_data.get('query_assessment', '')

    hays_data_logger.record_api_outputs({
        'query': user_query, 'initial_answer': initial_answer,
        'weighted_keywords': str(model_weighted_keywords),
        'year_keywords': str(model_year_keywords),
        'text_keywords': str(model_text_keywords),
        'query_assessment': model_query_assessment,
        'full_output': hay_raw
    })

    # Override from UI if user supplied keywords
    weighted_keywords = user_weighted_keywords or model_weighted_keywords
    year_keywords     = user_year_keywords.split(',') if user_year_keywords else model_year_keywords
    text_keywords     = user_text_keywords or model_text_keywords

    # ═══════════════════════════════════════════════════════════════════════════
    # RUN SEARCHES — single status block gives users step-by-step progress
    # ═══════════════════════════════════════════════════════════════════════════

    with st.status("⚙️ Running pipeline…", expanded=True) as pipeline_status:

        # ── Step 1: Keyword Search ────────────────────────────────────────────
        search_results_df = pd.DataFrame()
        if perform_keyword_search:
            st.write("🔑 **Step 1 of 5** — Running weighted keyword search…")
            results_list = search_with_dynamic_weights_expanded(
                user_keywords=weighted_keywords,
                corpus_terms={"terms": corpus_terms},
                data=lincoln_data,
                year_keywords=year_keywords,
                text_keywords=text_keywords,
                top_n_results=5
            )
            if results_list:
                search_results_df = pd.DataFrame(results_list)
                if "quote" in search_results_df.columns and "key_quote" not in search_results_df.columns:
                    search_results_df.rename(columns={"quote": "key_quote"}, inplace=True)
            n_kw = len(search_results_df)
            st.write(f"   ✅ Keyword search complete — {n_kw} result{'s' if n_kw != 1 else ''} found.")
            if not search_results_df.empty:
                log_keyword_search_results(
                    keyword_results_logger, search_results_df, user_query,
                    initial_answer, model_weighted_keywords,
                    model_year_keywords, model_text_keywords
                )
        else:
            st.write("🔑 **Step 1 of 5** — Keyword search skipped.")

        # ── Step 2: Semantic Search ───────────────────────────────────────────
        semantic_matches_df  = pd.DataFrame()
        user_query_embedding = None
        if perform_semantic_search:
            st.write("🧠 **Step 2 of 5** — Running semantic (embedding) search…")
            semantic_matches_df, user_query_embedding = search_text_local(
                lincoln_index_df, user_query + " " + initial_answer, n=5
            )
            top_segments = []
            for _, row in semantic_matches_df.iterrows():
                ft_v = row.get('full_text', '')
                segs = segment_text(ft_v)
                if segs and user_query_embedding is not None:
                    scores  = compare_segments_parallel(segs, user_query_embedding)
                    top_seg = max(scores, key=lambda x: x[1]) if scores else ("", 0)
                else:
                    top_seg = ("", 0)
                top_segments.append(top_seg[0])
            semantic_matches_df = semantic_matches_df.copy()
            semantic_matches_df["TopSegment"] = top_segments
            n_sem = len(semantic_matches_df)
            st.write(f"   ✅ Semantic search complete — {n_sem} result{'s' if n_sem != 1 else ''} found.")
            log_semantic_search_results(semantic_results_logger, semantic_matches_df, initial_answer)
        else:
            st.write("🧠 **Step 2 of 5** — Semantic search skipped.")

        # ── Step 3: Reranking ─────────────────────────────────────────────────
        full_reranked_results = []
        formatted_input       = ""
        model_output          = {}
        nic_raw               = ""

        if perform_reranking:

            def add_num_id(df):
                df = df.copy()
                df['_num_id'] = (df['text_id'].astype(str)
                                 .str.extract(r'(\d+)')[0]
                                 .astype(float).astype('Int64'))
                return df

            s_df = add_num_id(search_results_df)  if not search_results_df.empty  else pd.DataFrame(columns=['_num_id'])
            e_df = add_num_id(semantic_matches_df) if not semantic_matches_df.empty else pd.DataFrame(columns=['_num_id'])

            frames = [df for df in [s_df, e_df] if not df.empty and '_num_id' in df.columns]
            combined_df = (pd.concat(frames, ignore_index=True)
                           .drop_duplicates(subset=['_num_id'])
                           if frames else pd.DataFrame())

            all_combined_data = []
            kw_ids  = set(s_df['_num_id'].dropna()) if not s_df.empty else set()
            sem_ids = set(e_df['_num_id'].dropna()) if not e_df.empty else set()

            if not combined_df.empty:
                for _, row in combined_df.iterrows():
                    num_id = row.get('_num_id')
                    if num_id in kw_ids and perform_keyword_search:
                        stype = "Keyword"
                    elif num_id in sem_ids and perform_semantic_search:
                        stype = "Semantic"
                    else:
                        continue

                    tid  = str(row.get('text_id', str(num_id))).strip()
                    summ = str(row.get('summary', ''))
                    entry = triple_lookup(lincoln_dict, tid)
                    ft = entry.get('full_text') or ""
                    if not ft:
                        if stype == "Keyword":
                            ft = str(row.get('key_quote', row.get('quote', '')))
                        elif user_query_embedding is not None:
                            raw_ft = str(row.get('full_text', ''))
                            segs   = segment_text(raw_ft)
                            if segs:
                                scores = compare_segments_parallel(segs, user_query_embedding)
                                ft = max(scores, key=lambda x: x[1])[0] if scores else raw_ft[:500]

                    all_combined_data.append(
                        yaml.dump({"search_type": stype, "text_id": tid,
                                   "summary": summ, "full_text": ft},
                                  allow_unicode=True, default_flow_style=False, sort_keys=False)
                    )

            n_candidates = len(all_combined_data)
            st.write(f"📊 **Step 3 of 5** — Reranking {n_candidates} candidate"
                     f"{'s' if n_candidates != 1 else ''} with Cohere…")

            if all_combined_data:
                try:
                    reranked_resp = co.rerank(
                        model='rerank-v4.0-pro', query=user_query,
                        documents=all_combined_data, top_n=10
                    )
                    for idx, r in enumerate(reranked_resp.results):
                        doc_text = r.document['text'] if isinstance(r.document, dict) else str(r.document)
                        try:
                            parsed = yaml.safe_load(doc_text) or {}
                        except yaml.YAMLError:
                            parsed = {}
                        r_stype = str(parsed.get("search_type", "Unknown")).strip()
                        r_tid   = str(parsed.get("text_id", "Unknown")).strip()
                        r_summ  = str(parsed.get("summary", "")).strip()
                        r_ft    = str(parsed.get("full_text", "")).strip()
                        src_e   = triple_lookup(lincoln_dict, r_tid)
                        r_src   = src_e.get('source', 'Source not available')
                        full_reranked_results.append({
                            'Rank': idx + 1,
                            'Search Type': r_stype,
                            'Text ID': r_tid,
                            'Source': r_src,
                            'Summary': r_summ,
                            'Key Quote': r_ft,
                            'Relevance Score': r.relevance_score
                        })
                    top_score = full_reranked_results[0]['Relevance Score'] if full_reranked_results else 0
                    st.write(f"   ✅ Reranking complete — top relevance score: {top_score:.3f}.")
                except Exception as e:
                    st.error(f"Reranking error: {e}")
                    st.exception(e)

            formatted_input = format_reranked_for_nicolay(full_reranked_results, lincoln_dict)
            if full_reranked_results:
                log_reranking_results(reranking_results_logger,
                                      pd.DataFrame(full_reranked_results), user_query)

            # ── Step 4: Hay analysis display note ────────────────────────────
            st.write("🎩 **Step 4 of 5** — Hay's keyword analysis complete.")

            # ── Step 5: Nicolay synthesis ─────────────────────────────────────
            if formatted_input:
                st.write("📜 **Step 5 of 5** — Nicolay is synthesising a response…")
                nic_resp = client.chat.completions.create(
                    # nicolay_v3
                    #model="ft:gpt-4.1-mini-2025-04-14:personal:nicolay-v3:DEccNnWt",
                    # nicolay_v4
                    model="ft:gpt-4.1-mini-2025-04-14:personal:nicolay-v4:DIPD9hh5",
                    messages=[
                        {"role":"system","content":response_prompt},
                        {"role":"user","content":(
                            f"User Query: {user_query}\n\n"
                            f"Initial Answer: {initial_answer}\n\n"
                            f"{formatted_input}"
                        )}
                    ],
                    temperature=0, max_tokens=4000, top_p=1,
                    frequency_penalty=0, presence_penalty=0
                )
                nic_raw = nic_resp.choices[0].message.content
                if not nic_raw:
                    st.error("Nicolay returned an empty response.")
                    st.stop()
                model_output = _parse_model_json(nic_raw, "Nicolay v3")
                if not model_output:
                    st.stop()
                st.write("   ✅ Synthesis complete.")
            else:
                st.write("📜 **Step 5 of 5** — Skipped (no formatted input for Nicolay).")

        else:
            st.write("📊 **Steps 3–5** — Reranking and synthesis skipped.")

        pipeline_status.update(label="✅ Pipeline complete — see results below.", state="complete")

    # ═══════════════════════════════════════════════════════════════════════════
    # [U11] THREE-TAB RESULT LAYOUT  (revised schema)
    # Tab 1 — Answer & Sources : Hay + Nicolay synthesis + Match cards +
    #                             Retrieval diagnostics (primary view)
    # Tab 2 — Pipeline         : Keyword/Semantic search results + reranking
    #                             metadata + full CoT trace
    # Tab 3 — Model Feedback   : Nicolay's self-assessment and CoT sections
    # ═══════════════════════════════════════════════════════════════════════════

    tab_answer, tab_pipeline, tab_feedback = st.tabs([
        "💬 Answer & Sources",
        "🔬 Pipeline",
        "🤖 Model Feedback",
    ])

    # Precompute shared values needed across tabs
    match_analysis  = model_output.get("Match Analysis", {})  if model_output else {}
    qa_block        = model_output.get("User Query Analysis", {}) if model_output else {}
    synth_raw       = qa_block.get("synthesis_assessment", "")
    synth_badge, synth_tooltip = synthesis_type_badge(synth_raw)
    synth_key_m     = re.search(r"Type\s*([1-5])", str(synth_raw), re.IGNORECASE)
    synth_key       = synth_key_m.group(1) if synth_key_m else ""
    fa_block        = model_output.get("FinalAnswer", {}) if model_output else {}

    # ── Tab 1: Answer & Sources ────────────────────────────────────────────────
    with tab_answer:

        # ── Hay initial response ──────────────────────────────────────────────
        st.subheader("Hay's Initial Analysis")
        st.caption(
            "Hay is a fine-tuned model that provides an initial answer and "
            "steers keyword and semantic search. Compare its response with "
            "Nicolay's synthesis below to see how retrieval-augmented generation "
            "refines the output. **Note:** Hay's response is preliminary analysis "
            "generated before document retrieval — it is not verified against the "
            "corpus and may contain claims that go beyond the retrieved sources."
        )
        with st.expander("**Hay's Response**", expanded=True):
            st.markdown(initial_answer)
            render_keyword_pills(weighted_keywords, model_query_assessment)

        if not perform_reranking or not model_output:
            st.info("Enable 'Response and Analysis' to see Nicolay's synthesis.")
        else:
            st.divider()
            st.subheader("Nicolay's Synthesis")

            with st.expander("**How Does This Work?: Nicolay's Response and Analysis**"):
                st.write(nicolay_model_explainer)

            # ── [U9] Multi-signal diagnostic panel ───────────────────────────
            signals = compute_diagnostic_signals(
                full_reranked_results, match_analysis,
                synth_raw, fa_block.get("Text", "")
            )
            render_diagnostic_signals(signals)

            # ── [U12] Response Confidence Summary (plain container, not nested) ──
            fa_text = fa_block.get('Text', '')
            _verified_q, _displaced_q, _unverified_q = (
                verify_final_answer_quotes(
                    fa_text, full_reranked_results, lincoln_dict
                )
                if fa_text else ([], [], [])
            )
            try:
                render_confidence_summary(
                    final_answer_text   = fa_text,
                    reranked_results    = full_reranked_results,
                    verified_quotes     = _verified_q,
                    displaced_quotes    = _displaced_q,
                    unverified_quotes   = _unverified_q,
                    synth_type_num      = extract_synth_type_num(synth_raw),
                    query_text          = user_query,
                )
            except Exception as _cs_err:
                st.warning(f"⚠️ Confidence summary could not render: {_cs_err}")

            # ── [U2] Synthesis type badge + [U10] FinalAnswer verification ────
            with st.expander("**Nicolay's Response**", expanded=True):
                if synth_badge:
                    st.markdown(synth_badge, unsafe_allow_html=True)
                    if synth_tooltip:
                        st.caption(synth_tooltip)
                st.markdown("")
                # [U10] FinalAnswer with inline quote annotation.
                # Pass pre-computed quote lists (hoisted above) to avoid
                # running verification a second time.
                render_final_answer_with_verification(
                    fa_block, full_reranked_results, lincoln_dict,
                    precomputed_quotes=(_verified_q, _displaced_q, _unverified_q),
                )

            # ── Match Analysis cards directly below Nicolay response ──────────
            st.divider()
            if match_analysis:
                render_match_analysis_cards(
                    match_analysis, lincoln_dict, full_reranked_results
                )
            else:
                st.info("No match analysis available for this query.")

            # ── [U6] Retrieval diagnostics ────────────────────────────────────
            render_retrieval_diagnostics(full_reranked_results, match_analysis)

            # ── Log & history ─────────────────────────────────────────────────
            highlight_success = {}
            for mk, info in match_analysis.items():
                v, _ = verify_quote(info.get("Key Quote", ""),
                                    str(info.get("Text ID", "")), lincoln_dict)
                highlight_success[mk] = v

            log_nicolay_model_output(
                nicolay_data_logger, model_output, user_query,
                highlight_success, initial_answer
            )
            existing = st.session_state.get("query_history", [])
            if not existing or existing[-1].get("query") != user_query:
                record_history(user_query, model_output, synth_key)

    # ── Tab 2: Pipeline ────────────────────────────────────────────────────────
    with tab_pipeline:

        # Keyword search results
        if perform_keyword_search:
            st.markdown("### Keyword Search Results")
            with st.expander("**How Does This Work?: Dynamically Weighted Keyword Search**"):
                st.write(keyword_search_explainer)

            if search_results_df.empty:
                st.info("No keyword results found. Try modifying your query or "
                        "use Additional Search Options.")
            else:
                col_kw1, col_kw2 = st.columns(2)
                kw_col_map = {0: col_kw1, 1: col_kw2}
                for i, (_, row) in enumerate(search_results_df.iterrows(), 1):
                    label = f"**Keyword Match {i}**: *{row.get('source','')}* `{row.get('text_id','')}`"
                    with kw_col_map[(i - 1) % 2]:
                        with st.expander(label):
                            st.markdown(f"**Source:** {row.get('source','')}")
                            st.markdown(f"**Text ID:** {row.get('text_id','')}")
                            st.markdown(f"**Summary:**\n{row.get('summary','')}")
                            st.markdown(f"**Key Quote:**\n{row.get('key_quote', row.get('quote',''))}")
                            st.markdown(f"**Weighted Score:** {row.get('weighted_score','')}")
                            st.markdown("**Keyword Counts:**")
                            st.json(row.get('keyword_counts', {}))

            with st.expander("**Keyword Search Metadata**"):
                st.write("**User Query:**"); st.write(user_query)
                st.write("**Hay Initial Answer:**"); st.write(initial_answer)
                st.write("**Weighted Keywords:**"); st.json(weighted_keywords)
                st.write("**Year Keywords:**"); st.json(year_keywords)
                st.write("**Text Keywords:**"); st.json(text_keywords)
                st.write("**Query Assessment (Hay v3):**"); st.write(model_query_assessment)
                st.write("**Raw Results:**"); st.dataframe(search_results_df)
                st.write("**Full Hay Output:**"); st.write(hay_raw)

        st.divider()

        # Semantic search results
        if perform_semantic_search:
            st.markdown("### Semantic Search Results")
            with st.expander("**How Does This Work?: Semantic Search with HyDE**"):
                st.write(semantic_search_explainer)

            if not semantic_matches_df.empty:
                col_sem1, col_sem2 = st.columns(2)
                sem_col_map = {0: col_sem1, 1: col_sem2}
                for ctr, (_, row) in enumerate(semantic_matches_df.iterrows(), 1):
                    tid_v = row.get('text_id', row.name)
                    src_v = row.get('source', '')
                    label = f"**Semantic Match {ctr}**: *{src_v}* `{tid_v}`"
                    with sem_col_map[(ctr - 1) % 2]:
                        with st.expander(label, expanded=False):
                            st.markdown(f"**Source:** {src_v}")
                            st.markdown(f"**Text ID:** {tid_v}")
                            st.markdown(f"**Summary:**\n{row.get('summary','')}")
                            top_seg_text = row.get('TopSegment', '')
                            st.markdown(f"**Key Quote:** {top_seg_text}")
                            st.markdown(f"**Similarity Score:** {row.get('similarities', 0.0):.2f}")

                with st.expander("**Semantic Search Metadata**"):
                    st.dataframe(semantic_matches_df.drop(columns=["embedding"], errors="ignore"))

        st.divider()

        # Reranking
        if perform_reranking and full_reranked_results:
            st.markdown("### Ranked Search Results")
            with st.expander("**How Does This Work?: Relevance Ranking with Cohere's Rerank**"):
                st.write(relevance_ranking_explainer)

            for idx, r in enumerate(full_reranked_results[:3]):
                with st.expander(
                    f"**Reranked Match {idx+1} ({r.get('Search Type','')})**: "
                    f"`{r.get('Text ID','')}` — {r.get('Source','')[:50]}"
                ):
                    st.markdown(f"**Text ID:** {r.get('Text ID','')}")
                    st.markdown(f"**Source:** {r.get('Source','')}")
                    st.markdown(f"**Summary:** {r.get('Summary','')}")
                    ft_excerpt = r.get('Key Quote', '')
                    st.markdown(
                        f"**Full Text (excerpt):**\n"
                        f"{ft_excerpt[:500]}{'…' if len(ft_excerpt) > 500 else ''}"
                    )
                    st.markdown(f"**Relevance Score:** {r.get('Relevance Score', 0.0):.3f}")

            with st.expander("**Result Reranking Metadata**"):
                st.dataframe(pd.DataFrame(full_reranked_results))
                st.write("**Formatted input to Nicolay:**")
                st.write(formatted_input)

        # Full Chain-of-Thought trace — separate section for visibility
        if perform_reranking and model_output:
            st.divider()
            st.markdown("### 🧩 Nicolay's Chain-of-Thought Trace")
            st.caption(
                "The reasoning Nicolay performed before producing its final response — "
                "query analysis, initial answer review, match evaluation, and meta-analysis. "
                "Model Feedback is available in its own tab."
            )

            match_analysis_pipe = model_output.get("Match Analysis", {})

            # One expander per CoT section for easy scanning
            for section, label, icon in [
                ("User Query Analysis",   "User Query Analysis",   "🔍"),
                ("Initial Answer Review", "Initial Answer Review", "📝"),
                ("Match Analysis",        "Match Analysis",        "🎯"),
                ("Meta Analysis",         "Meta Analysis",         "🔬"),
            ]:
                block = (match_analysis_pipe if section == "Match Analysis"
                         else model_output.get(section, {}))
                if not block:
                    continue
                with st.expander(f"**{icon} {label}**", expanded=False):
                    if section == "Match Analysis":
                        for mk, info in block.items():
                            st.markdown(f"**{mk}:**")
                            for k, v in info.items():
                                st.markdown(f"- **{k}:** {v}")
                    else:
                        for k, v in block.items():
                            st.markdown(f"- **{k}:** {v}")

            with st.expander("**📄 Full Model Output (raw JSON)**", expanded=False):
                st.write(nic_raw)

    # ── Tab 3: Model Feedback ──────────────────────────────────────────────────
    with tab_feedback:
        if not perform_reranking or not model_output:
            st.info("Enable 'Response and Analysis' to see model feedback.")
        else:
            mf = model_output.get("Model Feedback", {})
            if not mf:
                st.info("No model feedback available for this query.")
            else:
                st.subheader("Nicolay's Self-Assessment")
                st.caption(
                    "This section surfaces Nicolay's own evaluation of its response: "
                    "the quality of retrieval, what evidence is missing, and suggestions "
                    "for improving the query or the corpus coverage."
                )

                # Retrieval Quality Notes
                rqn = mf.get("Retrieval Quality Notes", mf.get("Response Effectiveness", ""))
                if rqn:
                    with st.container(border=True):
                        st.markdown("**Retrieval Quality Notes**")
                        st.markdown(rqn)

                # Critical Missing Evidence Flag
                cmef = mf.get("Critical Missing Evidence Flag", "")
                if cmef:
                    with st.container(border=True):
                        st.markdown("**Critical Missing Evidence Flag**")
                        st.warning(cmef)

                # Suggested Improvements
                si = mf.get("Suggested Improvements", mf.get("Suggestions for Improvement", ""))
                if si:
                    with st.container(border=True):
                        st.markdown("**Suggested Improvements**")
                        st.markdown(si)

                # Any remaining fields not already displayed
                displayed = {
                    "Retrieval Quality Notes", "Response Effectiveness",
                    "Critical Missing Evidence Flag",
                    "Suggested Improvements", "Suggestions for Improvement"
                }
                extras = {k: v for k, v in mf.items() if k not in displayed}
                if extras:
                    with st.expander("Additional Model Feedback Fields"):
                        for k, v in extras.items():
                            st.markdown(f"**{k}:** {v}")
