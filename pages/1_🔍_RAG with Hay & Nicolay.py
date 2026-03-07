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

# ── version 1.0 ──────────────────────────────────────────────────────────────
# Fixes applied vs. legacy 0.3 build:
#   [F1] Data loading: replaced fragile CSV/JSON loaders with data_utils.py
#        functions (load_lincoln_speech_corpus, load_lincoln_index_embedded).
#        Parquet-backed embeddings are pre-computed — no per-row API calls.
#   [F2] Triple-keyed lincoln_dict: mirrors rag_pipeline.py so lookups by
#        "Text #: N", integer N, or bare string "N" all succeed.
#   [F3] Keyword search: removed inline redefinition of
#        search_with_dynamic_weights_expanded; now calls the canonical
#        keyword_search.py version with correct (corpus_terms, data) args.
#   [F4] Semantic search: uses pre-loaded parquet embeddings; removed the
#        per-row get_embedding() .apply() that caused lockups.
#   [F5] Log functions: updated to use 'text_id' column (not 'Unnamed: 0').
#   [F6] lincoln_dict lookup in reranking block: uses triple-keyed dict so
#        integer text_ids resolve correctly after str.extract().
# Enhancements:
#   [E1] Quote verification: verify_quote() checks whether Nicolay's quoted
#        passage appears verbatim (or fuzzy-matches) in the corpus chunk.
#        Each Match Analysis card shows a ✅ / ⚠️ badge.
#   [E2] Match Analysis card-grid UI: replaces flat expander list with a
#        styled card grid (relevance badge, source, blockquote, summary).
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nicolay: Exploring the Speeches of Abraham Lincoln with AI (version 1.0)",
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


# ── Data loading (F1) ────────────────────────────────────────────────────────
# Import from data_utils so loading is cached and consistent with benchmark.
from modules.data_utils import (
    load_lincoln_speech_corpus,
    load_voyant_word_counts,
    load_lincoln_index_embedded,
)
from modules.keyword_search import search_with_dynamic_weights_expanded


# ── DataLogger ───────────────────────────────────────────────────────────────
class DataLogger:
    def __init__(self, gc, sheet_name):
        self.gc = gc
        self.sheet = self.gc.open(sheet_name).sheet1

    def record_api_outputs(self, data_dict):
        now = dt.now()
        data_dict['Timestamp'] = now
        df = pd.DataFrame([data_dict])
        end_row = len(self.sheet.get_all_records()) + 2
        self.sheet.set_dataframe(df, (end_row, 1), copy_head=False, extend=True)


hays_data_logger        = DataLogger(gc, 'hays_data')
keyword_results_logger  = DataLogger(gc, 'keyword_search_results')
semantic_results_logger = DataLogger(gc, 'semantic_search_results')
reranking_results_logger = DataLogger(gc, 'reranking_results')
nicolay_data_logger     = DataLogger(gc, 'nicolay_data')


# ── Logging helpers ───────────────────────────────────────────────────────────
def log_keyword_search_results(keyword_results_logger, search_results_df, user_query,
                                initial_answer, model_weighted_keywords,
                                model_year_keywords, model_text_keywords):
    """Log keyword search results. [F5] Uses 'key_quote' column (post-rename)."""
    now = dt.now()
    for _, result in search_results_df.iterrows():
        record = {
            'Timestamp': now,
            'UserQuery': user_query,
            'initial_Answer': initial_answer,
            'Weighted_Keywords': str(model_weighted_keywords),
            'Year_Keywords': str(model_year_keywords),
            'text_keywords': str(model_text_keywords),
            'TextID': result.get('text_id', ''),
            'KeyQuote': result.get('key_quote', result.get('quote', '')),
            'WeightedScore': result.get('weighted_score', ''),
            'KeywordCounts': json.dumps(result.get('keyword_counts', {}))
        }
        keyword_results_logger.record_api_outputs(record)


def log_semantic_search_results(semantic_results_logger, semantic_matches, initial_answer):
    """Log semantic search results. [F5] Uses 'text_id' (not 'Unnamed: 0')."""
    now = dt.now()
    for _, row in semantic_matches.iterrows():
        record = {
            'Timestamp': now,
            'UserQuery': row.get('UserQuery', ''),
            'HyDE_Query': initial_answer,
            'TextID': row.get('text_id', ''),
            'SimilarityScore': row.get('similarities', ''),
            'TopSegment': row.get('TopSegment', '')
        }
        semantic_results_logger.record_api_outputs(record)


def log_reranking_results(reranking_results_logger, reranked_df, user_query):
    now = dt.now()
    for _, row in reranked_df.iterrows():
        record = {
            'Timestamp': now,
            'UserQuery': user_query,
            'Rank': row.get('Rank', ''),
            'SearchType': row.get('Search Type', ''),
            'TextID': row.get('Text ID', ''),
            'KeyQuote': row.get('Key Quote', ''),
            'Relevance_Score': row.get('Relevance Score', '')
        }
        reranking_results_logger.record_api_outputs(record)


def log_nicolay_model_output(nicolay_data_logger, model_output, user_query,
                              highlight_success_dict, initial_answer):
    final_answer_text = model_output.get("FinalAnswer", {}).get("Text", "No response available")
    references_raw = model_output.get("FinalAnswer", {}).get("References", [])
    references = ", ".join(references_raw) if isinstance(references_raw, list) else str(references_raw)

    query_analysis = model_output.get("User Query Analysis", {})
    synthesis_assessment = query_analysis.get("synthesis_assessment", "")
    query_intent = query_analysis.get("Query Intent", "")
    historical_context = query_analysis.get("Historical Context", "")

    answer_evaluation = model_output.get("Initial Answer Review", {}).get("Answer Evaluation", "")
    quote_integration = model_output.get("Initial Answer Review", {}).get("Quote Integration Points", "")

    model_feedback = model_output.get("Model Feedback", {})
    retrieval_quality = model_feedback.get("Retrieval Quality Notes",
                            model_feedback.get("Response Effectiveness", ""))
    missing_evidence = model_feedback.get("Critical Missing Evidence Flag", "")
    suggested_improvements = model_feedback.get("Suggested Improvements",
                                model_feedback.get("Suggestions for Improvement", ""))

    match_analysis = model_output.get("Match Analysis", {})
    match_fields = ['Text ID', 'Source', 'Summary', 'Key Quote', 'Historical Context', 'Relevance Assessment']
    match_data = {}
    for match_key, match_details in match_analysis.items():
        match_info = [f"{field}: {match_details.get(field, '')}" for field in match_fields]
        match_data[match_key] = "; ".join(match_info)

    meta_strategy = model_output.get("Meta Analysis", {}).get("Strategy for Response Composition", {})
    meta_synthesis = model_output.get("Meta Analysis", {}).get("Synthesis", "")

    record = {
        'Timestamp': dt.now(),
        'UserQuery': user_query,
        'initial_Answer': initial_answer,
        'FinalAnswer': final_answer_text,
        'References': references,
        'SynthesisAssessment': synthesis_assessment,
        'QueryIntent': query_intent,
        'HistoricalContext': historical_context,
        'AnswerEvaluation': answer_evaluation,
        'QuoteIntegration': quote_integration,
        **match_data,
        'MetaStrategy': str(meta_strategy),
        'MetaSynthesis': meta_synthesis,
        'RetrievalQualityNotes': retrieval_quality,
        'CriticalMissingEvidence': missing_evidence,
        'SuggestedImprovements': suggested_improvements
    }
    for match_key, success in highlight_success_dict.items():
        record[f'{match_key}_HighlightSuccess'] = success

    nicolay_data_logger.record_api_outputs(record)


# ── Prompt loading ────────────────────────────────────────────────────────────
def load_prompt(file_name):
    with open(file_name, 'r') as file:
        return file.read()


def load_prompts():
    if 'keyword_model_system_prompt' not in st.session_state:
        st.session_state['keyword_model_system_prompt'] = load_prompt('prompts/keyword_model_system_prompt.txt')
    if 'response_model_system_prompt' not in st.session_state:
        st.session_state['response_model_system_prompt'] = load_prompt('prompts/response_model_system_prompt.txt')
    if 'app_into' not in st.session_state:
        st.session_state['app_intro'] = load_prompt('prompts/app_intro.txt')
    if 'keyword_search_explainer' not in st.session_state:
        st.session_state['keyword_search_explainer'] = load_prompt('prompts/keyword_search_explainer.txt')
    if 'semantic_search_explainer' not in st.session_state:
        st.session_state['semantic_search_explainer'] = load_prompt('prompts/semantic_search_explainer.txt')
    if 'relevance_ranking_explainer' not in st.session_state:
        st.session_state['relevance_ranking_explainer'] = load_prompt('prompts/relevance_ranking_explainer.txt')
    if 'nicolay_model_explainer' not in st.session_state:
        st.session_state['nicolay_model_explainer'] = load_prompt('prompts/nicolay_model_explainer.txt')


load_prompts()

keyword_prompt           = st.session_state['keyword_model_system_prompt']
response_prompt          = st.session_state['response_model_system_prompt']
app_intro                = st.session_state['app_intro']
keyword_search_explainer = st.session_state['keyword_search_explainer']
semantic_search_explainer = st.session_state['semantic_search_explainer']
relevance_ranking_explainer = st.session_state['relevance_ranking_explainer']
nicolay_model_explainer  = st.session_state['nicolay_model_explainer']


# ── Utility functions ─────────────────────────────────────────────────────────
def segment_text(text, segment_size=500, overlap=100):
    words = text.split()
    segments = []
    for i in range(0, len(words), segment_size - overlap):
        segments.append(' '.join(words[i:i + segment_size]))
    return segments


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def search_text_local(df, user_query, n=5):
    """Semantic search over pre-loaded parquet embeddings. [F4]"""
    user_query_embedding = get_embedding(user_query)
    df = df.copy()
    df["similarities"] = df['embedding'].apply(
        lambda x: cosine_similarity(np.array(x), user_query_embedding)
        if isinstance(x, (list, np.ndarray)) else 0.0
    )
    top_n = df.sort_values("similarities", ascending=False).head(n)
    top_n = top_n.copy()
    top_n["UserQuery"] = user_query
    return top_n, user_query_embedding


def compare_segments_with_query_parallel(segments, query_embedding):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(get_embedding, seg) for seg in segments]
        seg_embeddings = [f.result() for f in futures]
    return [(segments[i], cosine_similarity(seg_embeddings[i], query_embedding))
            for i in range(len(segments))]


def highlight_key_quote(text, key_quote):
    parts = key_quote.split("...")
    if len(parts) >= 2:
        pattern = re.escape(parts[0]) + r"\s*.*?\s*" + re.escape(parts[-1]) + r"[.;,]?"
    else:
        pattern = re.escape(key_quote) + r"\s*[.;,]?"
    regex = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for match in regex.findall(text):
        text = text.replace(match, f"<mark>{match}</mark>")
    return text


def format_reranked_results_for_model_input(reranked_results, lincoln_dict):
    """Format top-5 reranked results for Nicolay model input. [F6]"""
    formatted = []
    for idx, result in enumerate(reranked_results[:5], 1):
        text_id = str(result.get('Text ID', 'Unknown')).strip()

        # Triple-key lookup: try "Text #: N", integer N, bare string N
        full_text = None
        m = re.search(r"(\d+)", text_id)
        num = int(m.group(1)) if m else None
        for key in (f"Text #: {text_id}", text_id,
                    f"Text #: {num}" if num is not None else None,
                    num, str(num) if num is not None else None):
            if key is None:
                continue
            entry = lincoln_dict.get(key)
            if isinstance(entry, dict):
                full_text = entry.get('full_text')
                if full_text:
                    break

        text_field_value = full_text if full_text else result.get('Key Quote', 'No quote')

        formatted.append(
            f"Match {idx}: "
            f"Search Type - {result.get('Search Type', 'Unknown')}, "
            f"Text ID - {text_id}, "
            f"Summary (curatorial description only — not quotable corpus text) - {result.get('Summary', 'No summary')}, "
            f"Full Text (select the most relevant passage to quote directly) - {text_field_value}, "
            f"Relevance Score - {result.get('Relevance Score', 0.0):.2f}"
        )
    return "\n\n".join(formatted) if formatted else "No results to format"


# ── E1: Quote verification ────────────────────────────────────────────────────
def verify_quote(key_quote: str, text_id: str, lincoln_dict: dict) -> tuple[bool, str]:
    """
    Check whether key_quote appears in the corpus chunk for text_id.
    Returns (verified: bool, method: str) where method is one of:
      'exact', 'fuzzy', 'not_found', 'chunk_missing'.
    """
    if not key_quote or not key_quote.strip():
        return False, "no_quote"

    # Resolve corpus entry
    chunk_text = None
    m = re.search(r"(\d+)", str(text_id))
    num = int(m.group(1)) if m else None
    for key in (text_id, f"Text #: {text_id}",
                f"Text #: {num}" if num is not None else None,
                num, str(num) if num is not None else None):
        if key is None:
            continue
        entry = lincoln_dict.get(key)
        if isinstance(entry, dict):
            chunk_text = entry.get('full_text', '')
            break

    if not chunk_text:
        return False, "chunk_missing"

    # Exact match
    if key_quote.strip() in chunk_text:
        return True, "exact"

    # Fuzzy: handle ellipsis-truncated quotes
    parts = [p.strip() for p in key_quote.split("...") if p.strip()]
    if len(parts) >= 2:
        if parts[0] in chunk_text and parts[-1] in chunk_text:
            return True, "fuzzy"

    # Fuzzy: first 60 chars as anchor
    anchor = key_quote.strip()[:60]
    if len(anchor) > 20 and anchor in chunk_text:
        return True, "fuzzy"

    return False, "not_found"


def quote_verification_badge(verified: bool, method: str) -> str:
    """Return an HTML badge string for inline display."""
    if method == "chunk_missing":
        return '<span style="color:#888;font-size:0.8em;">⬜ chunk not found</span>'
    if method == "no_quote":
        return '<span style="color:#888;font-size:0.8em;">⬜ no quote</span>'
    if verified:
        label = "exact match" if method == "exact" else "fuzzy match"
        return f'<span style="color:#2d7a2d;font-size:0.8em;">✅ {label}</span>'
    return '<span style="color:#b85c00;font-size:0.8em;">⚠️ not verified</span>'


# ── E2: Match Analysis card-grid ──────────────────────────────────────────────
def relevance_badge_html(relevance_text: str) -> str:
    """Colour-coded relevance pill from Nicolay's Relevance Assessment field."""
    t = (relevance_text or "").lower()
    if "high" in t:
        bg, fg = "#d4edda", "#155724"
    elif "medium" in t or "moderate" in t:
        bg, fg = "#fff3cd", "#856404"
    elif "low" in t:
        bg, fg = "#f8d7da", "#721c24"
    else:
        bg, fg = "#e2e3e5", "#383d41"
    # Show only the first word/phrase before em-dash or comma
    label = re.split(r"[—,]", relevance_text or "N/A")[0].strip()[:30]
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:12px;font-size:0.75em;font-weight:600;">{label}</span>'
    )


def render_match_analysis_cards(match_analysis: dict, lincoln_dict: dict):
    """
    Render Match Analysis as a responsive card grid. [E2]
    Each card shows: match key + relevance badge, Text ID, Source,
    key quote in a styled blockquote, summary, and quote verification badge.
    """
    st.markdown("### Match Analysis")

    card_css = """
    <style>
    .match-card {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .match-card .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .match-card .card-title {
        font-weight: 700;
        font-size: 0.95em;
        color: #212529;
    }
    .match-card .meta-line {
        font-size: 0.8em;
        color: #6c757d;
        margin-bottom: 2px;
    }
    .match-card blockquote {
        border-left: 3px solid #adb5bd;
        margin: 8px 0;
        padding: 4px 10px;
        color: #495057;
        font-style: italic;
        font-size: 0.85em;
        line-height: 1.5;
    }
    .match-card .summary-text {
        font-size: 0.8em;
        color: #6c757d;
        margin-top: 6px;
    }
    </style>
    """
    st.markdown(card_css, unsafe_allow_html=True)

    items = list(match_analysis.items())
    # Two-column grid
    cols = st.columns(2)
    for i, (match_key, match_info) in enumerate(items):
        text_id   = str(match_info.get("Text ID", ""))
        source    = match_info.get("Source", "")
        key_quote = match_info.get("Key Quote", "")
        summary   = match_info.get("Summary", "")
        relevance = match_info.get("Relevance Assessment", "")
        hist_ctx  = match_info.get("Historical Context", "")

        verified, method = verify_quote(key_quote, text_id, lincoln_dict)
        badge_html   = quote_verification_badge(verified, method)
        rel_badge    = relevance_badge_html(relevance)
        quote_display = f'"{key_quote[:300]}{"…" if len(key_quote) > 300 else ""}"'

        card_html = f"""
        <div class="match-card">
            <div class="card-header">
                <span class="card-title">{match_key}</span>
                {rel_badge}
            </div>
            <div class="meta-line"><strong>ID:</strong> {text_id}</div>
            <div class="meta-line"><strong>Source:</strong> {source}</div>
            <blockquote>{quote_display}</blockquote>
            <div class="summary-text">{summary[:200]}{"…" if len(summary) > 200 else ""}</div>
            <div style="margin-top:6px;">{badge_html}</div>
        </div>
        """
        with cols[i % 2]:
            st.markdown(card_html, unsafe_allow_html=True)

            # Expandable full-text with highlighting (preserves original behaviour)
            with st.expander(f"Full text & highlight — {match_key}", expanded=False):
                if hist_ctx:
                    st.markdown(f"**Historical Context:** {hist_ctx}")
                if text_id:
                    # Build formatted text_id for dict lookup
                    lookup_key = f"Text #: {text_id}" if not text_id.startswith("Text") else text_id
                    speech = lincoln_dict.get(lookup_key)
                    # Also try numeric fallback
                    if speech is None:
                        m_num = re.search(r"(\d+)", text_id)
                        if m_num:
                            speech = lincoln_dict.get(int(m_num.group(1)))
                    if speech:
                        full_text = speech.get('full_text', '')
                        formatted_full_text = full_text.replace("\\n", "<br>")
                        if key_quote and key_quote in full_text:
                            formatted_full_text = formatted_full_text.replace(
                                key_quote, f"<mark>{key_quote}</mark>"
                            )
                        elif key_quote:
                            formatted_full_text = highlight_key_quote(full_text, key_quote)
                            formatted_full_text = formatted_full_text.replace("\\n", "<br>")
                        st.markdown("**Full Text with Highlighted Quote:**", unsafe_allow_html=True)
                        st.markdown(
                            f'<div style="font-size:0.85em;line-height:1.6;">{formatted_full_text}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.info("Full text not found in corpus for this text ID.")


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.title("Exploring RAG with Nicolay and Hay")

image_url = 'http://danielhutchinson.org/wp-content/uploads/2024/01/nicolay_hay.png'
st.image(image_url, width=600)

st.subheader("**Navigating this App:**")
st.write("Expand the **How It Works?** box below for a walkthrough of the app. Continue to the search interface below to begin exploring Lincoln's speeches.")

with st.expander("**How It Works - Exploring RAG with Hay and Nicolay**"):
    st.write(app_intro)

# ── Search form ───────────────────────────────────────────────────────────────
with st.form("Search Interface"):
    st.markdown("Enter your query below:")
    user_query = st.text_input("Query")

    st.write("**Search Options**:")
    st.write("Note that at least one search method must be selected to perform Response and Analysis.")
    perform_keyword_search = st.toggle("Weighted Keyword Search", value=True)
    perform_semantic_search = st.toggle("Semantic Search", value=True)
    perform_reranking = st.toggle("Response and Analysis", value=True, key="reranking")

    if perform_reranking and not (perform_keyword_search or perform_semantic_search):
        st.warning("Response & Analysis requires at least one of the search methods (keyword or semantic).")

    with st.expander("Additional Search Options (In Development)"):
        st.markdown("The Hay model will suggest keywords based on your query, but you can select your own criteria for more focused keyword search using the interface below.")
        st.markdown("Weighted Keywords")
        default_values = [1.0, 1.0, 1.0, 1.0, 1.0]
        user_weighted_keywords = {}

        for i in range(1, 6):
            col1, col2 = st.columns(2)
            with col1:
                keyword = st.text_input(f"Keyword {i}", key=f"keyword_{i}")
            with col2:
                weight = st.number_input(f"Weight for Keyword {i}", min_value=0.0,
                                         value=default_values[i-1], step=0.1, key=f"weight_{i}")
            if keyword:
                user_weighted_keywords[keyword] = weight

        st.header("Year and Text Filters")
        user_year_keywords = st.text_input("Year Keywords (comma-separated - example: 1861, 1862, 1863)")
        user_text_keywords = st.multiselect("Text Selection:", [
            'At Peoria, Illinois', 'A House Divided', 'Eulogy on Henry Clay',
            'Farewell Address', 'Cooper Union Address', 'First Inaugural Address',
            'Second Inaugural Address', 'July 4th Message to Congress',
            'First Annual Message', 'Second Annual Message', 'Third Annual Message',
            'Fourth Annual Message', 'Emancipation Proclamation',
            'Public Letter to James Conkling', 'Gettysburg Address'
        ])

    submitted = st.form_submit_button("Submit")

# ── Pipeline execution ────────────────────────────────────────────────────────
if submitted:
    valid_search_condition = perform_keyword_search or perform_semantic_search

    if not valid_search_condition:
        st.error("Search halted: Invalid search condition. Please ensure at least one search method is selected.")
        st.stop()

    st.subheader("Starting RAG Process: (takes about 30–60 seconds in total)")

    # ── [F1] Load data via data_utils (cached, parquet-backed) ───────────────
    with st.spinner("Loading Lincoln corpus and embeddings…"):
        lincoln_data_df  = load_lincoln_speech_corpus()
        voyant_data_df   = load_voyant_word_counts()
        lincoln_index_df = load_lincoln_index_embedded()

    lincoln_data = lincoln_data_df.to_dict("records")

    # ── [F2] Triple-keyed lincoln_dict ────────────────────────────────────────
    lincoln_dict = {}
    for item in lincoln_data:
        tid = item.get("text_id", "")
        lincoln_dict[tid] = item                          # "Text #: N"
        m_num = re.search(r"(\d+)", str(tid))
        if m_num:
            num = int(m_num.group(1))
            lincoln_dict[num] = item                      # integer N
            lincoln_dict[str(num)] = item                 # bare string "N"

    # Corpus terms for keyword search
    if not voyant_data_df.empty and "corpusTerms" in voyant_data_df.columns:
        corpus_terms_raw = voyant_data_df.at[0, "corpusTerms"]
        corpus_terms_obj = (json.loads(corpus_terms_raw)
                            if isinstance(corpus_terms_raw, str)
                            else corpus_terms_raw)
        corpus_terms = corpus_terms_obj.get("terms", [])
    else:
        corpus_terms = []
        st.warning("Voyant corpus terms not found — keyword search will be limited.")

    # Prepare embeddings from parquet (already stored, no API calls) [F4]
    def _parse_embedding(x):
        if isinstance(x, (list, np.ndarray)):
            return np.array(x, dtype=float)
        if isinstance(x, str):
            try:
                return np.array(list(map(float, x.strip("[]").split(","))), dtype=float)
            except Exception:
                return np.zeros(1536)
        return np.zeros(1536)

    lincoln_index_df = lincoln_index_df.copy()
    lincoln_index_df["embedding"] = lincoln_index_df["embedding"].apply(_parse_embedding)

    # Enrich index df with source/summary from corpus
    def get_source_and_summary(text_id_val):
        entry = lincoln_dict.get(text_id_val, {})
        if not entry:
            m_num = re.search(r"(\d+)", str(text_id_val))
            if m_num:
                entry = lincoln_dict.get(int(m_num.group(1)), {})
        return entry.get("source", ""), entry.get("summary", "")

    lincoln_index_df["source"], lincoln_index_df["summary"] = zip(
        *lincoln_index_df["text_id"].apply(get_source_and_summary)
    )

    if not user_query:
        st.warning("Please enter a query.")
        st.stop()

    # ── Hay model call ────────────────────────────────────────────────────────
    messages_for_model = [
        {"role": "system", "content": keyword_prompt},
        {"role": "user", "content": user_query}
    ]
    with st.spinner("Calling Hay model…"):
        response = client.chat.completions.create(
            model="ft:gpt-4.1-mini-2025-04-14:personal:hays-v3:DEcb9s4u",
            messages=messages_for_model,
            temperature=0,
            max_tokens=800,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
    msg = response.choices[0].message.content
    api_response_data    = json.loads(msg)
    initial_answer       = api_response_data.get('initial_answer', '')
    model_weighted_keywords = api_response_data.get('weighted_keywords', {})
    model_year_keywords  = api_response_data.get('year_keywords', []) or []
    model_text_keywords  = api_response_data.get('text_keywords', []) or []
    model_query_assessment = api_response_data.get('query_assessment', '')

    hays_data_logger.record_api_outputs({
        'query': user_query,
        'initial_answer': initial_answer,
        'weighted_keywords': str(model_weighted_keywords),
        'year_keywords': str(model_year_keywords),
        'text_keywords': str(model_text_keywords),
        'query_assessment': model_query_assessment,
        'full_output': msg
    })

    # Keyword override from UI
    if user_weighted_keywords:
        weighted_keywords = user_weighted_keywords
        year_keywords = user_year_keywords.split(',') if user_year_keywords else []
        text_keywords = user_text_keywords if user_text_keywords else []
    else:
        weighted_keywords = model_weighted_keywords
        year_keywords = model_year_keywords
        text_keywords = model_text_keywords

    with st.expander("**Hay's Response**", expanded=True):
        st.markdown(initial_answer)
        st.write("**How Does This Work?**")
        st.write("The Initial Response based on the user query is given by Hay, a fine-tuned large language model. This response helps Hay steer the search process by guiding the selection of weighted keywords and informing the semantic search over the Lincoln speech corpus. Compare Hay's Response with Nicolay's Response and Analysis at the end of the RAG process to see how AI techniques can be used for historical sources.")

    col1, col2 = st.columns(2)

    # ── [F3] Keyword Search ───────────────────────────────────────────────────
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
            # Normalise column name: keyword_search returns 'quote', pipeline expects 'key_quote'
            if results_list:
                search_results_df = pd.DataFrame(results_list)
                if "quote" in search_results_df.columns and "key_quote" not in search_results_df.columns:
                    search_results_df.rename(columns={"quote": "key_quote"}, inplace=True)

            st.markdown("### Keyword Search Results")
            with st.expander("**How Does This Work?: Dynamically Weighted Keyword Search**"):
                st.write(keyword_search_explainer)

            if search_results_df.empty:
                st.info("No keyword search results found. Try modifying your query or use the Additional Search Options.")
            else:
                for idx, result in search_results_df.iterrows():
                    expander_label = f"**Keyword Match {idx+1}**: *{result.get('source','')}* `{result.get('text_id','')}`"
                    with st.expander(expander_label):
                        st.markdown(f"**Source:** {result.get('source','')}")
                        st.markdown(f"**Text ID:** {result.get('text_id','')}")
                        st.markdown(f"**Summary:**\n{result.get('summary','')}")
                        st.markdown(f"**Key Quote:**\n{result.get('key_quote', result.get('quote',''))}")
                        st.markdown(f"**Weighted Score:** {result.get('weighted_score','')}")
                        st.markdown("**Keyword Counts:**")
                        st.json(result.get('keyword_counts', {}))

            with st.expander("**Keyword Search Metadata**"):
                st.write("**User Query:**")
                st.write(user_query)
                st.write("**Model Response:**")
                st.write(initial_answer)
                st.write("**Weighted Keywords:**")
                st.json(weighted_keywords)
                st.write("**Year Keywords:**")
                st.json(year_keywords)
                st.write("**Text Keywords:**")
                st.json(text_keywords)
                st.write("**Query Assessment (Hay v3):**")
                st.write(model_query_assessment)
                st.write("**Raw Search Results:**")
                st.dataframe(search_results_df)
                st.write("**Full Hay Output:**")
                st.write(msg)

            if not search_results_df.empty:
                log_keyword_search_results(
                    keyword_results_logger, search_results_df, user_query,
                    initial_answer, model_weighted_keywords,
                    model_year_keywords, model_text_keywords
                )

    # ── [F4] Semantic Search ──────────────────────────────────────────────────
    semantic_matches_df = pd.DataFrame()
    user_query_embedding = None
    with col2:
        if perform_semantic_search:
            st.markdown("### Semantic Search Results")
            with st.expander("**How Does This Work?: Semantic Search with HyDE**"):
                st.write(semantic_search_explainer)

            progress_text = "Semantic search in progress."
            my_bar = st.progress(0, text=progress_text)

            # Single embedding call for the HyDE query — no per-row calls
            hyde_query = user_query + " " + initial_answer
            semantic_matches_df, user_query_embedding = search_text_local(
                lincoln_index_df, hyde_query, n=5
            )
            my_bar.progress(50, text=progress_text)

            top_segments = []
            match_counter = 1
            for idx, row in semantic_matches_df.iterrows():
                progress_update = min(50 + (match_counter / len(semantic_matches_df)) * 45, 99)
                my_bar.progress(int(progress_update), text=progress_text)

                text_id_val = row.get('text_id', row.name)
                source_val  = row.get('source', '')
                full_text_val = row.get('full_text', '')

                semantic_expander_label = f"**Semantic Match {match_counter}**: *{source_val}* `{text_id_val}`"
                with st.expander(semantic_expander_label, expanded=False):
                    st.markdown(f"**Source:** {source_val}")
                    st.markdown(f"**Text ID:** {text_id_val}")
                    st.markdown(f"**Summary:**\n{row.get('summary','')}")

                    segments = segment_text(full_text_val)
                    if segments and user_query_embedding is not None:
                        segment_scores = compare_segments_with_query_parallel(segments, user_query_embedding)
                        top_segment = max(segment_scores, key=lambda x: x[1]) if segment_scores else ("", 0)
                    else:
                        top_segment = ("", 0)
                    top_segments.append(top_segment[0])
                    st.markdown(f"**Key Quote:** {top_segment[0]}")
                    st.markdown(f"**Similarity Score:** {top_segment[1]:.2f}")

                match_counter += 1

            semantic_matches_df = semantic_matches_df.copy()
            semantic_matches_df["TopSegment"] = top_segments
            my_bar.progress(100, text="Semantic search completed.")
            time.sleep(1)
            my_bar.empty()

            with st.expander("**Semantic Search Metadata**"):
                st.write("**Semantic Search Metadata**")
                st.dataframe(semantic_matches_df.drop(columns=["embedding"], errors="ignore"))

            log_semantic_search_results(semantic_results_logger, semantic_matches_df, initial_answer)

    # ── Reranking & Nicolay ───────────────────────────────────────────────────
    if perform_reranking:
        # Normalise search_results text_ids for merging [F6]
        if not search_results_df.empty:
            # Extract numeric id for dedup — store as int
            search_results_df = search_results_df.copy()
            search_results_df['_num_id'] = (
                search_results_df['text_id'].astype(str)
                .str.extract(r'(\d+)')[0]
                .astype(float).astype('Int64')
            )
        else:
            search_results_df = pd.DataFrame(columns=['_num_id'])

        if not semantic_matches_df.empty:
            semantic_matches_df = semantic_matches_df.copy()
            semantic_matches_df['_num_id'] = (
                semantic_matches_df['text_id'].astype(str)
                .str.extract(r'(\d+)')[0]
                .astype(float).astype('Int64')
            )
        else:
            semantic_matches_df = pd.DataFrame(columns=['_num_id'])

        # Combine and deduplicate on numeric id
        _frames = [df for df in [search_results_df, semantic_matches_df] if not df.empty and '_num_id' in df.columns]
        if _frames:
            combined_df = pd.concat(_frames, ignore_index=True).drop_duplicates(subset=['_num_id'])
        else:
            combined_df = pd.DataFrame()

        all_combined_data = []
        if not combined_df.empty:
            kw_ids  = set(search_results_df['_num_id'].dropna()) if not search_results_df.empty else set()
            sem_ids = set(semantic_matches_df['_num_id'].dropna()) if not semantic_matches_df.empty else set()

            for _, result in combined_df.iterrows():
                num_id = result.get('_num_id')
                if num_id in kw_ids and perform_keyword_search:
                    search_type = "Keyword"
                elif num_id in sem_ids and perform_semantic_search:
                    search_type = "Semantic"
                else:
                    continue

                text_id = str(result.get('text_id', str(num_id))).strip()
                summary = str(result.get('summary', ''))

                # Full chunk text lookup [F6] — triple-key
                full_text = None
                for key in (text_id, f"Text #: {text_id}",
                            f"Text #: {num_id}" if num_id is not None else None,
                            num_id, str(int(num_id)) if num_id is not None else None):
                    if key is None:
                        continue
                    entry = lincoln_dict.get(key)
                    if isinstance(entry, dict):
                        full_text = entry.get('full_text')
                        if full_text:
                            break

                if not full_text:
                    if search_type == "Keyword":
                        full_text = str(result.get('key_quote', result.get('quote', '')))
                    else:
                        ft = str(result.get('full_text', ''))
                        if ft and user_query_embedding is not None:
                            segs = segment_text(ft)
                            if segs:
                                scores = compare_segments_with_query_parallel(segs, user_query_embedding)
                                full_text = max(scores, key=lambda x: x[1])[0] if scores else ft[:500]
                        else:
                            full_text = ft[:500] if ft else ""

                doc_dict = {
                    "search_type": search_type,
                    "text_id": text_id,
                    "summary": summary,
                    "full_text": full_text,
                }
                all_combined_data.append(
                    yaml.dump(doc_dict, allow_unicode=True, default_flow_style=False, sort_keys=False)
                )

        full_reranked_results = []
        if all_combined_data:
            st.markdown("### Ranked Search Results")
            try:
                reranked_response = co.rerank(
                    model='rerank-v4.0-pro',
                    query=user_query,
                    documents=all_combined_data,
                    top_n=10
                )
                with st.expander("**How Does This Work?: Relevance Ranking with Cohere's Rerank**"):
                    st.write(relevance_ranking_explainer)

                for idx, result in enumerate(reranked_response.results):
                    combined_data = result.document
                    doc_text = combined_data['text'] if isinstance(combined_data, dict) else str(combined_data)
                    try:
                        doc_parsed = yaml.safe_load(doc_text) or {}
                    except yaml.YAMLError:
                        doc_parsed = {}

                    r_search_type = str(doc_parsed.get("search_type", "Unknown")).strip()
                    r_text_id     = str(doc_parsed.get("text_id", "Unknown")).strip()
                    r_summary     = str(doc_parsed.get("summary", "")).strip()
                    r_full_text   = str(doc_parsed.get("full_text", "")).strip()

                    lookup_key = f"Text #: {r_text_id}" if not r_text_id.startswith("Text") else r_text_id
                    r_source = lincoln_dict.get(lookup_key, {})
                    if not r_source:
                        m_num = re.search(r"(\d+)", r_text_id)
                        if m_num:
                            r_source = lincoln_dict.get(int(m_num.group(1)), {})
                    r_source_str = (r_source.get('source', 'Source information not available')
                                    if isinstance(r_source, dict) else 'Source information not available')

                    full_reranked_results.append({
                        'Rank': idx + 1,
                        'Search Type': r_search_type,
                        'Text ID': r_text_id,
                        'Source': r_source_str,
                        'Summary': r_summary,
                        'Key Quote': r_full_text,
                        'Relevance Score': result.relevance_score
                    })

                    if idx < 3:
                        expander_label = f"**Reranked Match {idx+1} ({r_search_type} Search)**: `Text ID: {r_text_id}`"
                        with st.expander(expander_label):
                            st.markdown(f"**Text ID:** {r_text_id}")
                            st.markdown(f"**Source:** {r_source_str}")
                            st.markdown(f"**Summary:** {r_summary}")
                            st.markdown(f"**Full Text (excerpt):**\n{r_full_text[:500]}{'...' if len(r_full_text) > 500 else ''}")
                            st.markdown(f"**Relevance Score:** {result.relevance_score:.2f}")

            except Exception as e:
                st.error(f"Error in reranking: {str(e)}")
                st.exception(e)

        # Format for Nicolay
        formatted_input_for_model = format_reranked_results_for_model_input(
            full_reranked_results, lincoln_dict
        )

        with st.expander("**Result Reranking Metadata**"):
            reranked_df = pd.DataFrame(full_reranked_results)
            st.dataframe(reranked_df)
            st.write("**Formatted Results:**")
            st.write(formatted_input_for_model)

        if full_reranked_results:
            log_reranking_results(reranking_results_logger, pd.DataFrame(full_reranked_results), user_query)

        # ── Nicolay model call ────────────────────────────────────────────────
        if formatted_input_for_model:
            messages_for_second_model = [
                {"role": "system", "content": response_prompt},
                {"role": "user", "content": (
                    f"User Query: {user_query}\n\n"
                    f"Initial Answer: {initial_answer}\n\n"
                    f"{formatted_input_for_model}"
                )}
            ]
            with st.spinner("Calling Nicolay model…"):
                second_model_response = client.chat.completions.create(
                    model="ft:gpt-4.1-mini-2025-04-14:personal:nicolay-v3:DEccNnWt",
                    messages=messages_for_second_model,
                    temperature=0,
                    max_tokens=4000,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )

            response_content = second_model_response.choices[0].message.content

            if response_content:
                model_output = json.loads(response_content)

                st.header("Nicolay's Response & Analysis:")

                with st.expander("**How Does This Work?: Nicolay's Response and Analysis**"):
                    st.write(nicolay_model_explainer)

                # Final Answer
                with st.expander("**Nicolay's Response**", expanded=True):
                    final_answer = model_output.get("FinalAnswer", {})
                    st.markdown(f"**Response:**\n{final_answer.get('Text', 'No response available')}")
                    if final_answer.get("References"):
                        st.markdown("**References:**")
                        for reference in final_answer["References"]:
                            st.markdown(f"- {reference}")

                # ── [E2] Match Analysis card-grid with [E1] quote verification
                highlight_success_dict = {}
                if "Match Analysis" in model_output:
                    render_match_analysis_cards(model_output["Match Analysis"], lincoln_dict)
                    # Populate highlight_success_dict for logging
                    for match_key, match_info in model_output["Match Analysis"].items():
                        kq = match_info.get("Key Quote", "")
                        tid = str(match_info.get("Text ID", ""))
                        verified, _ = verify_quote(kq, tid, lincoln_dict)
                        highlight_success_dict[match_key] = verified

                # Analysis Metadata
                with st.expander("**Analysis Metadata**"):
                    if "User Query Analysis" in model_output:
                        st.markdown("**User Query Analysis:**")
                        for key, value in model_output["User Query Analysis"].items():
                            st.markdown(f"- **{key}:** {value}")

                    if "Initial Answer Review" in model_output:
                        st.markdown("**Initial Answer Review:**")
                        for key, value in model_output["Initial Answer Review"].items():
                            st.markdown(f"- **{key}:** {value}")

                    if "Match Analysis" in model_output:
                        st.markdown("**Match Analysis (raw):**")
                        for match_key, match_info in model_output["Match Analysis"].items():
                            st.markdown(f"- **{match_key}:**")
                            for key, value in match_info.items():
                                st.markdown(f"  - {key}: {value}")

                    if "Meta Analysis" in model_output:
                        st.markdown("**Meta Analysis:**")
                        for key, value in model_output["Meta Analysis"].items():
                            st.markdown(f"- **{key}:** {value}")

                    if "Model Feedback" in model_output:
                        st.markdown("**Model Feedback:**")
                        for key, value in model_output["Model Feedback"].items():
                            st.markdown(f"- **{key}:** {value}")

                    st.write("**Full Model Output:**")
                    st.write(response_content)

                log_nicolay_model_output(
                    nicolay_data_logger, model_output, user_query,
                    highlight_success_dict, initial_answer
                )
