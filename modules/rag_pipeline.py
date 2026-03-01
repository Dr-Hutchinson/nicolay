# modules/rag_pipeline.py

import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime as dt
from openai import OpenAI
import cohere
from concurrent.futures import ThreadPoolExecutor
import re
import time

# Import all necessary modules
from modules.data_logging import (
    log_keyword_search_results,
    log_semantic_search_results,
    log_reranking_results,
    log_nicolay_model_output,
    DataLogger
)

from modules.data_utils import (
    load_lincoln_speech_corpus,
    load_voyant_word_counts,
    load_lincoln_index_embedded
)

from modules.misc_helpers import (
    remove_duplicates,
    highlight_key_quote
)

from modules.prompt_loader import load_prompts
from modules.keyword_search import search_with_dynamic_weights_expanded
from modules.semantic_search import (
    search_text,
    compare_segments_with_query_parallel
)
from modules.reranking import (
    prepare_documents_for_reranking,
    rerank_results,
    format_reranked_results_for_model_input
)

# Remove the direct import of ColBERTSearcher
# Instead, we'll accept a ColBERT searcher instance as a parameter


def extract_full_text(combined_text):
    markers = ["Full Text:\n", "Full Text: \n", "Full Text:"]
    if isinstance(combined_text, str):
        for marker in markers:
            marker_index = combined_text.find(marker)
            if marker_index != -1:
                return combined_text[marker_index + len(marker):].strip()
        return ""
    return ""

def segment_text(text, segment_size=100, overlap=50):
    words = text.split()
    segments = []
    for i in range(0, len(words), segment_size - overlap):
        segment = words[i:i + segment_size]
        segments.append(' '.join(segment))
    return segments

def run_rag_pipeline(
    user_query,
    perform_keyword_search=True,
    perform_semantic_search=True,
    perform_colbert_search=True,
    perform_reranking=True,
    colbert_searcher=None,  # Accept a ColBERT searcher instance
    hays_data_logger=None,
    keyword_results_logger=None,
    semantic_results_logger=None,
    reranking_results_logger=None,
    nicolay_data_logger=None,
    gc=None,
    openai_api_key=None,
    cohere_api_key=None,
    top_n_results=5
):
    try:
        # 1. Load Prompts
        load_prompts()
        keyword_prompt = st.session_state["keyword_model_system_prompt"]
        response_prompt = st.session_state["response_model_system_prompt"]

        # 2. Initialize Clients
        openai_client = OpenAI(api_key=openai_api_key or st.secrets["openai_api_key"])
        cohere_client = cohere.Client(api_key=cohere_api_key or st.secrets["cohere_api_key"])

        # 3. Load Data
        lincoln_data_df = load_lincoln_speech_corpus()
        voyant_data_df = load_voyant_word_counts()
        lincoln_index_df = load_lincoln_index_embedded()

        lincoln_data = lincoln_data_df.to_dict("records")
        lincoln_dict = {item["text_id"]: item for item in lincoln_data}

        # Use the provided colbert_searcher instance or create a local one if needed
        if colbert_searcher is None:
            # Import locally to avoid circular imports
            try:
                from modules.colbert_search import ColBERTSearcher
                colbert_searcher = ColBERTSearcher()
                try:
                    colbert_searcher.load_index()  # This will raise FileNotFoundError if index is missing
                except FileNotFoundError:
                    st.error("ColBERT index not found in data directory")
            except ImportError:
                st.error("No ColBERT implementation available. Please provide a colbert_searcher instance.")
                colbert_searcher = None
                perform_colbert_search = False  # Disable ColBERT search if no implementation available

        # Process Voyant data
        if not voyant_data_df.empty and "corpusTerms" in voyant_data_df.columns:
            corpus_terms_json = voyant_data_df.at[0, "corpusTerms"]
            corpus_terms = (json.loads(corpus_terms_json) if isinstance(corpus_terms_json, str)
                          else corpus_terms_json)["terms"]
        else:
            st.error("No corpusTerms found in voyant_data_df.")
            corpus_terms = []

        # Prepare Lincoln index
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

        # 4. Call Hay Model
        messages_for_model = [
            {"role": "system", "content": keyword_prompt},
            {"role": "user", "content": user_query}
        ]
        response = openai_client.chat.completions.create(
            model="ft:gpt-4.1-mini-2025-04-14:personal:hays-v3:DEcb9s4u",
            messages=messages_for_model,
            temperature=0,
            max_tokens=800  # Increased for v3 query_assessment field
        )
        raw_hay_output = response.choices[0].message.content

        hay_output = json.loads(raw_hay_output)
        initial_answer = hay_output.get("initial_answer", "")
        model_weighted_keywords = hay_output.get("weighted_keywords", {})
        model_year_keywords = hay_output.get("year_keywords", [])
        model_text_keywords = hay_output.get("text_keywords", [])
        model_query_assessment = hay_output.get("query_assessment", "")  # New in Hay v3 — plain string

        if hays_data_logger:
            hays_data_logger.record_api_outputs({
                "query": user_query,
                "initial_answer": initial_answer,
                "weighted_keywords": model_weighted_keywords,
                "year_keywords": model_year_keywords,
                "text_keywords": model_text_keywords,
                "query_assessment": model_query_assessment,  # String — no json.dumps needed
                "full_output": raw_hay_output,
                "Timestamp": dt.now()
            })

        # 5. Keyword Search
        search_results_df = pd.DataFrame()
        if perform_keyword_search and corpus_terms:
            results_list = search_with_dynamic_weights_expanded(
                user_keywords=model_weighted_keywords,
                corpus_terms={"terms": corpus_terms},
                data=lincoln_data,
                year_keywords=model_year_keywords,
                text_keywords=model_text_keywords,
                top_n_results=top_n_results
            )
            search_results_df = pd.DataFrame(results_list)
            if "quote" in search_results_df.columns:
                search_results_df.rename(columns={"quote": "key_quote"}, inplace=True)

            if keyword_results_logger and not search_results_df.empty:
                log_keyword_search_results(
                    keyword_results_logger,
                    search_results_df,
                    user_query,
                    initial_answer,
                    model_weighted_keywords,
                    model_year_keywords,
                    model_text_keywords
                )

        # 6. Semantic Search
        semantic_matches_df = pd.DataFrame()
        if perform_semantic_search and not lincoln_index_df.empty:
            semantic_results, user_query_embedding = search_text(
                lincoln_index_df,
                user_query + initial_answer,
                client=openai_client,
                n=top_n_results
            )

            if not semantic_results.empty:
                top_segments = []
                for idx, row in semantic_results.iterrows():
                    segments = segment_text(row["full_text"], segment_size=300)
                    segment_scores = compare_segments_with_query_parallel(
                        segments,
                        user_query_embedding,
                        openai_client
                    )
                    best_segment = max(segment_scores, key=lambda x: x[1]) if segment_scores else ("", 0)
                    top_segments.append(best_segment[0])

                semantic_results["TopSegment"] = top_segments
                if "Unnamed: 0" in semantic_results.columns:
                    semantic_results.rename(columns={"Unnamed: 0": "text_id"}, inplace=True)
                semantic_matches_df = semantic_results

                if semantic_results_logger:
                    try:
                        log_semantic_search_results(
                            semantic_results_logger,
                            semantic_matches_df,
                            initial_answer
                        )
                    except Exception as e:
                        st.error(f"Error logging semantic search results: {str(e)}")

        # 7. ColBERT Search
        colbert_matches_df = pd.DataFrame()
        if perform_colbert_search and colbert_searcher is not None:
            try:
                colbert_matches_df = colbert_searcher.search(
                    user_query,
                    k=top_n_results
                )
            except Exception as e:
                st.error(f"Error in ColBERT search: {str(e)}")

        # 8. Combine Results & Rerank
        combined_df = pd.concat([
            search_results_df,
            semantic_matches_df,
            colbert_matches_df
        ])

        combined_df = combined_df.drop_duplicates(subset=["text_id"]) if not combined_df.empty else combined_df

        reranked_df = pd.DataFrame()
        if perform_reranking and not combined_df.empty:
            try:
                # Prepare documents using the new function from reranking.py
                documents_for_cohere = prepare_documents_for_reranking(combined_df, user_query)

                # Use the new reranking function
                reranked_df = rerank_results(
                    query=user_query,
                    documents=documents_for_cohere,
                    cohere_client=cohere_client
                )

                if not reranked_df.empty and reranking_results_logger:
                    log_reranking_results(reranking_results_logger, reranked_df, user_query)

            except Exception as e:
                st.error(f"Error in reranking: {str(e)}")
                st.exception(e)  # Show full traceback
                reranked_df = pd.DataFrame()  # Ensure we have an empty DataFrame on error

        # 9. Final "Nicolay" model call
        nicolay_output = {}
        if perform_reranking and not reranked_df.empty:
            try:
                # Convert reranked_df to records for formatting if needed
                reranked_records = reranked_df.to_dict('records') if isinstance(reranked_df, pd.DataFrame) else []

                # Format top 3 for the second model
                formatted_for_nicolay = format_reranked_results_for_model_input(reranked_records)

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
                    model="ft:gpt-4.1-mini-2025-04-14:personal:nicolay-v3:DEccNnWt",
                    messages=nicolay_messages,
                    temperature=0,
                    max_tokens=3000  # Increased for v3: k=5 matches + longer Type 4/5 FinalAnswers
                )

                raw_nicolay = second_model_response.choices[0].message.content
                try:
                    nicolay_output = json.loads(raw_nicolay)

                    if nicolay_data_logger and nicolay_output:
                        highlight_success_dict = {}
                        log_nicolay_model_output(
                            nicolay_data_logger,
                            model_output=nicolay_output,
                            user_query=user_query,
                            highlight_success_dict=highlight_success_dict,
                            initial_answer=initial_answer
                        )

                except json.JSONDecodeError as e:
                    st.error(f"Nicolay model output was not valid JSON: {str(e)}")
                    st.write("Raw output:", raw_nicolay)

            except Exception as e:
                st.error(f"Error in Nicolay model processing: {str(e)}")
                st.exception(e)

        # Return results
        return {
            "hay_output": hay_output,
            "initial_answer": initial_answer,
            "query_assessment": model_query_assessment,  # New in Hay v3
            "search_results": search_results_df,
            "semantic_results": semantic_matches_df,
            "colbert_results": colbert_matches_df,
            "reranked_results": reranked_df,
            "nicolay_output": nicolay_output,
        }

    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        return {}
