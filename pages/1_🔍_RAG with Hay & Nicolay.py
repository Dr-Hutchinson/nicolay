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

# ── version 1.1 ──────────────────────────────────────────────────────────────
# UI Enhancements over v1.0:
#   [U1] Hay keyword transparency strip: top weighted keywords shown as
#        styled pills directly under Hay's response — no metadata dive needed.
#   [U2] Synthesis type badge on Nicolay response header: Type 1–5 label
#        with colour coding so users know immediately what kind of answer
#        they're getting before reading it.
#   [U3] Calibration decoupling warning: if reranker scores are high but
#        Nicolay rates all matches Low, a banner warns of possible retrieval
#        failure — surfaces the Q11/RC-5 failure mode honestly.
#   [U4] Session query history in sidebar: every submitted query is logged
#        with its synthesis type; click to expand the stored response.
#   [U5] Corpus coverage notice: a persistent sidebar panel listing known
#        corpus gaps (Last Public Address, Greeley Letter) so users know
#        what the system cannot see before querying.
#   [U6] Retrieval diagnostics panel: after reranking, a compact table shows
#        which search method surfaced each chunk, its reranker score, and
#        whether it made it into Nicolay's Match Analysis.
#   [U7] Card text size increased to match app body text.
#   [U8] Quote verification labels made fully descriptive and user-facing,
#        including explicit fabrication/displacement warning on failure.
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nicolay: Exploring the Speeches of Abraham Lincoln with AI",
    layout='wide',
    page_icon='🔍'
)

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
    """Format top-5 reranked results for Nicolay model input."""
    out = []
    for i, r in enumerate(reranked_results[:5], 1):
        tid = str(r.get('Text ID', 'Unknown')).strip()
        entry = triple_lookup(lincoln_dict, tid)
        full_text = entry.get('full_text') or r.get('Key Quote', 'No quote')
        out.append(
            f"Match {i}: "
            f"Search Type - {r.get('Search Type','Unknown')}, "
            f"Text ID - {tid}, "
            f"Summary (curatorial description only — not quotable corpus text) - {r.get('Summary','No summary')}, "
            f"Full Text (select the most relevant passage to quote directly) - {full_text}, "
            f"Relevance Score - {r.get('Relevance Score', 0.0):.2f}"
        )
    return "\n\n".join(out) if out else "No results to format"


# ── E1 / U8: Quote verification ───────────────────────────────────────────────
def verify_quote(key_quote, text_id, lincoln_dict):
    """
    Returns (verified: bool, method: str).
    method ∈ {'exact', 'fuzzy', 'not_found', 'chunk_missing', 'no_quote'}
    """
    if not key_quote or not key_quote.strip():
        return False, "no_quote"
    entry = triple_lookup(lincoln_dict, text_id)
    chunk = entry.get('full_text', '')
    if not chunk:
        return False, "chunk_missing"
    if key_quote.strip() in chunk:
        return True, "exact"
    parts = [p.strip() for p in key_quote.split("...") if p.strip()]
    if len(parts) >= 2 and parts[0] in chunk and parts[-1] in chunk:
        return True, "fuzzy"
    anchor = key_quote.strip()[:60]
    if len(anchor) > 20 and anchor in chunk:
        return True, "fuzzy"
    return False, "not_found"


def quote_badge_html(verified, method):
    """[U8] Fully descriptive, user-facing verification badge."""
    if method == "chunk_missing":
        return ('<span style="color:#6c757d;font-size:0.95em;font-weight:500;">'
                '⬜ Source chunk unavailable — quote cannot be verified</span>')
    if method == "no_quote":
        return ('<span style="color:#6c757d;font-size:0.95em;font-weight:500;">'
                '⬜ No quote provided by model</span>')
    if verified and method == "exact":
        return ('<span style="color:#155724;font-size:0.95em;font-weight:500;">'
                '✅ Quote verified — text confirmed present in Lincoln corpus</span>')
    if verified and method == "fuzzy":
        return ('<span style="color:#155724;font-size:0.95em;font-weight:500;">'
                '✅ Quote verified — partial match confirmed in Lincoln corpus</span>')
    return ('<span style="color:#721c24;font-size:0.95em;font-weight:600;">'
            '⚠️ Quote not found in corpus chunk — possible fabrication or displacement</span>')


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


# ── U3: Calibration decoupling detector ──────────────────────────────────────
def calibration_decoupling_warning(reranked_results, match_analysis):
    """
    Returns True if the top reranker score is high (≥0.70) but Nicolay
    rated all Match Analysis entries as Low relevance — the Q11/RC-5 pattern.
    """
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


_CARD_CSS = """
<style>
.match-card {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    font-size: 1rem;
}
.match-card .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.match-card .card-title {
    font-weight: 700;
    font-size: 1.05em;
    color: #212529;
}
.match-card .meta-line {
    font-size: 1em;
    color: #495057;
    margin-bottom: 5px;
    line-height: 1.5;
}
.match-card blockquote {
    border-left: 3px solid #6c757d;
    margin: 12px 0;
    padding: 6px 14px;
    color: #343a40;
    font-style: italic;
    font-size: 1em;
    line-height: 1.65;
    background: #f8f9fa;
    border-radius: 0 6px 6px 0;
}
.match-card .summary-text {
    font-size: 0.95em;
    color: #6c757d;
    margin-top: 8px;
    line-height: 1.55;
}
.match-card .verify-line {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid #f0f0f0;
}
</style>
"""


def render_match_analysis_cards(match_analysis, lincoln_dict, reranked_results=None):
    """
    [E2/U7] Two-column card grid for Match Analysis.
    Each card: title + relevance badge, ID, source, blockquote, summary,
    quote verification badge. Expandable full-text with highlighting inside.
    """
    st.markdown("### Match Analysis")
    st.markdown(_CARD_CSS, unsafe_allow_html=True)

    # Build a map of text_id → reranker score for the diagnostics tooltip
    score_map = {}
    if reranked_results:
        for r in reranked_results:
            score_map[str(r.get('Text ID', ''))] = r.get('Relevance Score', 0.0)

    items = list(match_analysis.items())
    cols  = st.columns(2)

    for i, (match_key, info) in enumerate(items):
        text_id   = str(info.get("Text ID", ""))
        source    = info.get("Source", "")
        key_quote = info.get("Key Quote", "")
        summary   = info.get("Summary", "")
        relevance = info.get("Relevance Assessment", "")
        hist_ctx  = info.get("Historical Context", "")
        reranker_score = score_map.get(text_id, None)

        verified, method = verify_quote(key_quote, text_id, lincoln_dict)
        badge_html  = quote_badge_html(verified, method)
        rel_badge   = relevance_badge_html(relevance)
        q_display   = f'"{key_quote[:320]}{"…" if len(key_quote) > 320 else ""}"'

        score_line = (f'<div class="meta-line" style="color:#6c757d;font-size:0.88em;">'
                      f'Reranker score: {reranker_score:.3f}</div>'
                      if reranker_score is not None else "")

        card_html = f"""
        <div class="match-card">
            <div class="card-header">
                <span class="card-title">{match_key}</span>
                {rel_badge}
            </div>
            <div class="meta-line"><strong>ID:</strong> {text_id}</div>
            <div class="meta-line"><strong>Source:</strong> {source}</div>
            {score_line}
            <blockquote>{q_display}</blockquote>
            <div class="summary-text">{summary[:220]}{"…" if len(summary) > 220 else ""}</div>
            <div class="verify-line">{badge_html}</div>
        </div>
        """
        with cols[i % 2]:
            st.markdown(card_html, unsafe_allow_html=True)
            with st.expander(f"Full text & highlight — {match_key}", expanded=False):
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
                    st.markdown("**Full Text with Highlighted Quote:**", unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size:0.95em;line-height:1.65;">{html_ft}</div>',
                                unsafe_allow_html=True)
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
    used_ids = {str(v.get("Text ID", "")) for v in match_analysis.values()}
    rows = []
    for r in reranked_results:
        tid = str(r.get('Text ID', ''))
        rows.append({
            "Rank":           r.get('Rank', ''),
            "Search Type":    r.get('Search Type', ''),
            "Text ID":        tid,
            "Source":         r.get('Source', '')[:55] + ("…" if len(r.get('Source','')) > 55 else ""),
            "Reranker Score": f"{r.get('Relevance Score', 0.0):.3f}",
            "Used by Nicolay": "✅ Yes" if tid in used_ids else "—",
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
        m = re.search(r"(Direct|Inferential|Absence|Multi.passage|Contrastive|Historiographical)",
                      query_assessment, re.IGNORECASE)
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
            model="ft:gpt-4.1-mini-2025-04-14:personal:hays-v3:DEcb9s4u",
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

    # ── Hay response display with keyword pills [U1] ──────────────────────────
    with st.expander("**Hay's Response**", expanded=True):
        st.markdown(initial_answer)
        render_keyword_pills(weighted_keywords, model_query_assessment)
        st.caption(
            "Hay is a fine-tuned model that provides an initial answer and steers "
            "keyword and semantic search. Compare its response with Nicolay's final "
            "answer below to see how retrieval-augmented generation refines the output."
        )

    col1, col2 = st.columns(2)

    # ── Keyword Search ────────────────────────────────────────────────────────
    search_results_df = pd.DataFrame()
    with col1:
        if perform_keyword_search:
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

            st.markdown("### Keyword Search Results")
            with st.expander("**How Does This Work?: Dynamically Weighted Keyword Search**"):
                st.write(keyword_search_explainer)

            if search_results_df.empty:
                st.info("No keyword results found. Try modifying your query or "
                        "use Additional Search Options.")
            else:
                for i, (_, row) in enumerate(search_results_df.iterrows(), 1):
                    label = f"**Keyword Match {i}**: *{row.get('source','')}* `{row.get('text_id','')}`"
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

            if not search_results_df.empty:
                log_keyword_search_results(
                    keyword_results_logger, search_results_df, user_query,
                    initial_answer, model_weighted_keywords,
                    model_year_keywords, model_text_keywords
                )

    # ── Semantic Search ───────────────────────────────────────────────────────
    semantic_matches_df  = pd.DataFrame()
    user_query_embedding = None
    with col2:
        if perform_semantic_search:
            st.markdown("### Semantic Search Results")
            with st.expander("**How Does This Work?: Semantic Search with HyDE**"):
                st.write(semantic_search_explainer)

            bar = st.progress(0, text="Semantic search in progress.")
            semantic_matches_df, user_query_embedding = search_text_local(
                lincoln_index_df, user_query + " " + initial_answer, n=5
            )
            bar.progress(50, text="Semantic search in progress.")

            top_segments = []
            for ctr, (_, row) in enumerate(semantic_matches_df.iterrows(), 1):
                bar.progress(min(50 + ctr * 9, 99), text="Semantic search in progress.")
                tid_v = row.get('text_id', row.name)
                src_v = row.get('source', '')
                ft_v  = row.get('full_text', '')
                label = f"**Semantic Match {ctr}**: *{src_v}* `{tid_v}`"
                with st.expander(label, expanded=False):
                    st.markdown(f"**Source:** {src_v}")
                    st.markdown(f"**Text ID:** {tid_v}")
                    st.markdown(f"**Summary:**\n{row.get('summary','')}")
                    segs = segment_text(ft_v)
                    if segs and user_query_embedding is not None:
                        scores   = compare_segments_parallel(segs, user_query_embedding)
                        top_seg  = max(scores, key=lambda x: x[1]) if scores else ("", 0)
                    else:
                        top_seg  = ("", 0)
                    top_segments.append(top_seg[0])
                    st.markdown(f"**Key Quote:** {top_seg[0]}")
                    st.markdown(f"**Similarity Score:** {top_seg[1]:.2f}")

            semantic_matches_df = semantic_matches_df.copy()
            semantic_matches_df["TopSegment"] = top_segments
            bar.progress(100, text="Semantic search completed.")
            time.sleep(1); bar.empty()

            with st.expander("**Semantic Search Metadata**"):
                st.dataframe(semantic_matches_df.drop(columns=["embedding"], errors="ignore"))

            log_semantic_search_results(semantic_results_logger, semantic_matches_df, initial_answer)

    # ── Reranking & Nicolay ───────────────────────────────────────────────────
    if perform_reranking:

        def add_num_id(df):
            df = df.copy()
            df['_num_id'] = (df['text_id'].astype(str)
                             .str.extract(r'(\d+)')[0]
                             .astype(float).astype('Int64'))
            return df

        s_df = add_num_id(search_results_df)   if not search_results_df.empty   else pd.DataFrame(columns=['_num_id'])
        e_df = add_num_id(semantic_matches_df)  if not semantic_matches_df.empty else pd.DataFrame(columns=['_num_id'])

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

        full_reranked_results = []
        if all_combined_data:
            st.markdown("### Ranked Search Results")
            try:
                reranked_resp = co.rerank(
                    model='rerank-v4.0-pro', query=user_query,
                    documents=all_combined_data, top_n=10
                )
                with st.expander("**How Does This Work?: Relevance Ranking with Cohere's Rerank**"):
                    st.write(relevance_ranking_explainer)

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

                    if idx < 3:
                        with st.expander(
                            f"**Reranked Match {idx+1} ({r_stype})**: `{r_tid}` — {r_src[:50]}"
                        ):
                            st.markdown(f"**Text ID:** {r_tid}")
                            st.markdown(f"**Source:** {r_src}")
                            st.markdown(f"**Summary:** {r_summ}")
                            st.markdown(
                                f"**Full Text (excerpt):**\n"
                                f"{r_ft[:500]}{'…' if len(r_ft)>500 else ''}"
                            )
                            st.markdown(f"**Relevance Score:** {r.relevance_score:.3f}")

            except Exception as e:
                st.error(f"Reranking error: {e}")
                st.exception(e)

        # Format & metadata
        formatted_input = format_reranked_for_nicolay(full_reranked_results, lincoln_dict)

        with st.expander("**Result Reranking Metadata**"):
            st.dataframe(pd.DataFrame(full_reranked_results))
            st.write("**Formatted input to Nicolay:**")
            st.write(formatted_input)

        if full_reranked_results:
            log_reranking_results(reranking_results_logger,
                                  pd.DataFrame(full_reranked_results), user_query)

        # ── Nicolay model call ────────────────────────────────────────────────
        if formatted_input:
            with st.spinner("Nicolay is synthesising a response…"):
                nic_resp = client.chat.completions.create(
                    model="ft:gpt-4.1-mini-2025-04-14:personal:nicolay-v3:DEccNnWt",
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
            match_analysis = model_output.get("Match Analysis", {})

            st.header("Nicolay's Response & Analysis:")

            with st.expander("**How Does This Work?: Nicolay's Response and Analysis**"):
                st.write(nicolay_model_explainer)

            # ── [U3] Calibration decoupling warning ───────────────────────────
            if calibration_decoupling_warning(full_reranked_results, match_analysis):
                st.warning(
                    "⚠️ **Retrieval calibration notice:** The reranker returned high-confidence "
                    "scores, but Nicolay rated all retrieved matches as Low relevance. "
                    "This pattern — seen in queries where the corpus lacks the required "
                    "documents — may indicate that the system has retrieved plausible but "
                    "ultimately off-target material. Treat this response with additional caution "
                    "and consider rephrasing your query."
                )

            # ── [U2] Nicolay response with synthesis type badge ───────────────
            qa_block  = model_output.get("User Query Analysis", {})
            synth_raw = qa_block.get("synthesis_assessment", "")
            synth_badge, synth_tooltip = synthesis_type_badge(synth_raw)

            # Extract type key for history logging
            synth_key_m = re.search(r"Type\s*([1-5])", str(synth_raw), re.IGNORECASE)
            synth_key   = synth_key_m.group(1) if synth_key_m else ""

            fa_block = model_output.get("FinalAnswer", {})
            with st.expander("**Nicolay's Response**", expanded=True):
                if synth_badge:
                    st.markdown(synth_badge, unsafe_allow_html=True)
                    if synth_tooltip:
                        st.caption(synth_tooltip)
                st.markdown("")
                st.markdown(f"**Response:**\n{fa_block.get('Text','No response available')}")
                refs = fa_block.get("References", [])
                if refs:
                    st.markdown("**References:**")
                    for ref in refs:
                        st.markdown(f"- {ref}")

            # ── Match Analysis card grid [E2/U7] + verification [E1/U8] ──────
            highlight_success = {}
            if match_analysis:
                render_match_analysis_cards(
                    match_analysis, lincoln_dict, full_reranked_results
                )
                for mk, info in match_analysis.items():
                    v, _ = verify_quote(info.get("Key Quote",""),
                                        str(info.get("Text ID","")), lincoln_dict)
                    highlight_success[mk] = v

            # ── [U6] Retrieval diagnostics ────────────────────────────────────
            render_retrieval_diagnostics(full_reranked_results, match_analysis)

            # ── Analysis Metadata (full chain-of-thought) ─────────────────────
            with st.expander("**Analysis Metadata — Full Chain-of-Thought**"):
                for section, label in [
                    ("User Query Analysis",  "User Query Analysis"),
                    ("Initial Answer Review","Initial Answer Review"),
                    ("Meta Analysis",        "Meta Analysis"),
                    ("Model Feedback",       "Model Feedback"),
                ]:
                    block = model_output.get(section, {})
                    if block:
                        st.markdown(f"**{label}:**")
                        for k, v in block.items():
                            st.markdown(f"- **{k}:** {v}")

                if match_analysis:
                    st.markdown("**Match Analysis (raw):**")
                    for mk, info in match_analysis.items():
                        st.markdown(f"- **{mk}:**")
                        for k, v in info.items():
                            st.markdown(f"  - {k}: {v}")

                st.write("**Full Model Output (JSON):**")
                st.write(nic_raw)

            # ── Log & history ─────────────────────────────────────────────────
            log_nicolay_model_output(
                nicolay_data_logger, model_output, user_query,
                highlight_success, initial_answer
            )
            record_history(user_query, model_output, synth_key)
            # Refresh sidebar after adding history entry
            render_sidebar_history()
