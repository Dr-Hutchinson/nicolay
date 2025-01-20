# modules/rag_pipeline.py

import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime as dt
from openai import OpenAI
import cohere
import pygsheets
from google.oauth2 import service_account
from concurrent.futures import ThreadPoolExecutor
import re
import time

# Logging & helper functions from your modules
from modules.data_logging import (
    log_keyword_search_results,
    log_semantic_search_results,
    log_reranking_results,
    log_nicolay_model_output,
    DataLogger
)

# Data-loading functions
from modules.data_utils import (
    load_lincoln_speech_corpus,
    load_voyant_word_counts,
    load_lincoln_index_embedded
)

# Miscellaneous helpers
from modules.misc_helpers import (
    remove_duplicates,
    highlight_key_quote
    # Add other helpers as needed
)

# Prompt loader
from modules.prompt_loader import load_prompts

# Keyword search functions
from modules.keyword_search import search_with_dynamic_weights_expanded

# Define semantic search and reranking imports as needed
from modules.semantic_search import (
    search_text,
    compare_segments_with_query_parallel
)
from modules.reranking import rerank_results, format_reranked_results_for_model_input

#######################################################
# HELPER FUNCTIONS (Mirroring rag_process's approach)
#######################################################
def extract_full_text(combined_text):
    markers = ["Full Text:\n", "Full Text: \n", "Full Text:"]
    if isinstance(combined_text, str):
        for marker in markers:
            marker_index = combined_text.find(marker)
            if marker_index != -1:
                # Extract the full text starting from the marker
                return combined_text[marker_index + len(marker):].strip()
        return ""
    else:
        return ""

def segment_text(text, segment_size=100, overlap=50):
    words = text.split()
    segments = []
    for i in range(0, len(words), segment_size - overlap):
        segment = words[i:i + segment_size]
        segments.append(' '.join(segment))
    return segments

#######################################################
# MAIN PIPELINE FUNCTION
#######################################################
def run_rag_pipeline(
    user_query,
    perform_keyword_search=True,
    perform_semantic_search=True,
    perform_reranking=True,
    # Logging objects
    hays_data_logger=None,
    keyword_results_logger=None,
    semantic_results_logger=None,
    reranking_results_logger=None,
    nicolay_data_logger=None,
    # Cloud credentials
    gc=None,
    openai_api_key=None,
    cohere_api_key=None,
    # Possibly toggles for top_n, etc.
    top_n_results=5
):
    """
    Mirrors the 'rag_process.py' approach in a single function.
    Returns a dict with keys:
      - "initial_answer"
      - "search_results" (keyword)
      - "semantic_matches"
      - "reranked_results" (df)
      - "nicolay_output" (JSON from second model)
      - "hay_output" (the raw JSON from the first model)
    """

    # 1. Load Prompts
    load_prompts()
    keyword_prompt = st.session_state["keyword_model_system_prompt"]
    response_prompt = st.session_state["response_model_system_prompt"]

    # 2. Initialize Clients (OpenAI, Cohere)
    if openai_api_key:
        st.secrets["openai_api_key"] = openai_api_key
    if cohere_api_key:
        st.secrets["cohere_api_key"] = cohere_api_key

    openai_client = OpenAI(api_key=openai_api_key)
    cohere_client = cohere.Client(api_key=cohere_api_key)

    # 3. Load the data
    lincoln_data_df = load_lincoln_speech_corpus()       # DataFrame
    voyant_data_df = load_voyant_word_counts()           # DataFrame
    lincoln_index_df = load_lincoln_index_embedded()     # DataFrame

    lincoln_data = lincoln_data_df.to_dict("records")
    lincoln_dict = {item["text_id"]: item for item in lincoln_data}

    if not voyant_data_df.empty and "corpusTerms" in voyant_data_df.columns:
        corpus_terms_json = voyant_data_df.at[0, "corpusTerms"]
        if isinstance(corpus_terms_json, str):
            corpus_terms = json.loads(corpus_terms_json)["terms"]
        elif isinstance(corpus_terms_json, dict):
            corpus_terms = corpus_terms_json["terms"]
        else:
            raise ValueError("Unexpected format for 'corpusTerms'")
    else:
        st.error("No corpusTerms found in voyant_data_df.")
        corpus_terms = []  # fallback

    # Prepare the Lincoln index DF
    lincoln_index_df["embedding"] = lincoln_index_df["embedding"].apply(
        lambda x: list(map(float, x.strip("[]").split(","))) if isinstance(x, str) else []
    )
    lincoln_index_df["full_text"] = lincoln_index_df["combined"].apply(extract_full_text)

    def get_source_and_summary(text_id_str):
        entry = lincoln_dict.get(text_id_str, {})
        return entry.get("source", ""), entry.get("summary", "")

    lincoln_index_df["source"], lincoln_index_df["summary"] = zip(
        *lincoln_index_df["text_id"].apply(get_source_and_summary)
    )

    # 4. Call "Hay" Model to get JSON
    messages_for_model = [
        {"role": "system", "content": keyword_prompt},
        {"role": "user", "content": user_query}
    ]
    response = openai_client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal:hays-gpt4o:9tFqrYwI",
        messages=messages_for_model,
        temperature=0,
        max_tokens=500
    )
    raw_hay_output = response.choices[0].message.content

    # Debug
    st.write("**Raw Hay output**:")
    st.write(raw_hay_output)

    try:
        hay_output = json.loads(raw_hay_output)
    except json.JSONDecodeError as e:
        st.error(f"Hay model output was not valid JSON: {e}")
        return {}  # or st.stop()

    # Extract fields
    initial_answer = hay_output.get("initial_answer", "")
    model_weighted_keywords = hay_output.get("weighted_keywords", {})
    model_year_keywords = hay_output.get("year_keywords", [])
    model_text_keywords = hay_output.get("text_keywords", [])

    # Log to hays_data_logger
    if hays_data_logger:
        hays_data_logger.record_api_outputs({
            "query": user_query,
            "initial_answer": initial_answer,
            "weighted_keywords": model_weighted_keywords,
            "year_keywords": model_year_keywords,
            "text_keywords": model_text_keywords,
            "full_output": raw_hay_output,
            "Timestamp": dt.now()
        })

    # 5. If toggled, do Weighted Keyword Search
    search_results_df = pd.DataFrame()
    if perform_keyword_search and corpus_terms:
        results_list = search_with_dynamic_weights_expanded(
            user_keywords=model_weighted_keywords,
            corpus_terms={"terms": corpus_terms},  # Adjust as per your JSON structure
            data=lincoln_data,
            year_keywords=model_year_keywords,
            text_keywords=model_text_keywords,
            top_n_results=top_n_results
        )
        search_results_df = pd.DataFrame(results_list)
        # Rename “quote” -> “key_quote” if needed
        if "quote" in search_results_df.columns:
            search_results_df.rename(columns={"quote": "key_quote"}, inplace=True)

        # Log the keyword results
        if keyword_results_logger and not search_results_df.empty:
            log_keyword_search_results(
                keyword_results_logger,
                search_results=search_results_df,
                user_query=user_query,
                initial_answer=initial_answer,
                model_weighted_keywords=model_weighted_keywords,
                model_year_keywords=model_year_keywords,
                model_text_keywords=model_text_keywords
            )

    # 6. If toggled, do Semantic Search
    semantic_matches_df = pd.DataFrame()
    user_query_embedding = None
    if perform_semantic_search and not lincoln_index_df.empty:
        semantic_results, user_query_embedding = search_text(
            lincoln_index_df,
            user_query + initial_answer,
            client=openai_client,
            n=top_n_results
        )
        if not semantic_results.empty:
            # For each top match, find a "key quote" by segmenting
            top_segments = []
            for idx, row in semantic_results.iterrows():
                segments = segment_text(row["full_text"], segment_size=300)
                # compare in parallel
                segment_scores = compare_segments_with_query_parallel(
                    segments, user_query_embedding, openai_client
                )
                best_segment = max(segment_scores, key=lambda x: x[1]) if segment_scores else ("", 0)
                top_segments.append(best_segment[0])
            semantic_results["TopSegment"] = top_segments

            # Rename "Unnamed: 0" -> "text_id" if present
            if "Unnamed: 0" in semantic_results.columns:
                semantic_results.rename(columns={"Unnamed: 0": "text_id"}, inplace=True)

            semantic_matches_df = semantic_results

            # Log
            if semantic_results_logger:
                log_semantic_search_results(semantic_results_logger, semantic_matches_df, initial_answer)

    # 7. Combine & Deduplicate
    combined_df = pd.concat([search_results_df, semantic_matches_df], ignore_index=True)
    if not combined_df.empty:
        combined_df = combined_df.drop_duplicates(subset=["text_id"], keep="first")

    # 8. Rerank if toggled
    reranked_df = pd.DataFrame()
    if perform_reranking and not combined_df.empty:
        # Build doc strings like "Keyword|Text ID: #|Summary|KeyQuote"
        docs_for_cohere = []
        for _, row in combined_df.iterrows():
            search_type = "Keyword" if row.get("key_quote", "") else "Semantic"
            text_id_str = row.get("text_id", "")
            summary_str = row.get("summary", "")
            quote_str = row.get("key_quote", row.get("TopSegment", ""))
            docs_for_cohere.append(
                f"{search_type}|Text ID: {text_id_str}|{summary_str}|{quote_str}"
            )

        # Verify documents are valid
        invalid_docs = [doc for doc in docs_for_cohere if not isinstance(doc, str) or "|" not in doc]
        if invalid_docs:
            st.error(f"Invalid documents for reranking: {invalid_docs}")
            return {}

        try:
            cohere_results = rerank_results(
                query=user_query,
                documents=docs_for_cohere,
                api_key=cohere_api_key,
                top_n=10
            )
            # Convert to DataFrame
            reranked_data = []
            for i, item in enumerate(cohere_results):
                doc_text = item.document["text"] if isinstance(item.document, dict) else item.document
                data_parts = doc_text.split("|")
                if len(data_parts) >= 4:
                    search_type = data_parts[0].strip()
                    text_id_part = data_parts[1].strip()
                    summary = data_parts[2].strip()
                    quote = data_parts[3].strip()

                    # Parse out actual text ID
                    text_id_str = text_id_part.replace("Text ID:", "").strip()

                    # Get source from lincoln_dict if you wish
                    source_str = lincoln_dict.get(text_id_str, {}).get("source", "Unknown Source")

                    reranked_data.append({
                        "Rank": i + 1,
                        "Search Type": search_type,
                        "Text ID": text_id_str,
                        "Source": source_str,
                        "Summary": summary,
                        "Key Quote": quote,
                        "Relevance Score": item.relevance_score
                    })

            reranked_df = pd.DataFrame(reranked_data)

            # Log reranking
            if reranking_results_logger and not reranked_df.empty:
                log_reranking_results(reranking_results_logger, reranked_df, user_query)

        except Exception as e:
            st.error("Error in reranking: " + str(e))

    # 9. Final "Nicolay" model call
    nicolay_output = {}
    if perform_reranking and not reranked_df.empty:
        # Format top 3 for the second model
        formatted_for_nicolay = format_reranked_results_for_model_input(reranked_df.to_dict("records"))

        # Build your message
        nicolay_messages = [
            {"role": "system", "content": response_prompt},
            {
                "role": "user",
                "content": f"User Query: {user_query}\n\n"
                           f"Initial Answer: {initial_answer}\n\n"
                           f"{formatted_for_nicolay}"
            }
        ]
        second_model_response = openai_client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal:nicolay-gpt4o:9tG7Cypl",
            messages=nicolay_messages,
            temperature=0,
            max_tokens=2000
        )
        raw_nicolay = second_model_response.choices[0].message.content
        try:
            nicolay_output = json.loads(raw_nicolay)
        except json.JSONDecodeError:
            st.error("Nicolay model output was not valid JSON.")

        if nicolay_data_logger and nicolay_output:
            highlight_success_dict = {}
            log_nicolay_model_output(
                nicolay_data_logger,
                model_output=nicolay_output,
                user_query=user_query,
                initial_answer=initial_answer,
                highlight_success_dict=highlight_success_dict
            )

    # 10. Return the final dictionary
    return {
        "hay_output": hay_output,
        "initial_answer": initial_answer,
        "search_results": search_results_df,
        "semantic_results": semantic_matches_df,
        "reranked_results": reranked_df,
        "nicolay_output": nicolay_output,
    }
