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

# Importing logging & helper functions from your modules
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

# Semantic search and reranking imports
from modules.semantic_search import (
    search_text,
    compare_segments_with_query_parallel
)
from modules.reranking import rerank_results, format_reranked_results_for_model_input

#######################################################
# HELPER FUNCTIONS (Mirroring rag_process's approach)
#######################################################

def extract_full_text(combined_text):
    """
    Extracts the full text from a combined text string using predefined markers.
    """
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
    """
    Splits the text into segments of specified size with a defined overlap.
    """
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
    # Additional parameters
    top_n_results=5
):
    """
    Runs the RAG pipeline integrating keyword search, semantic search, reranking, and final response generation.

    Returns a dictionary containing outputs from each stage of the pipeline.
    """

    # 1. Load Prompts
    load_prompts()
    keyword_prompt = st.session_state["keyword_model_system_prompt"]
    response_prompt = st.session_state["response_model_system_prompt"]

    # 2. Initialize Clients (OpenAI, Cohere)
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
    else:
        openai_client = OpenAI(api_key=st.secrets["openai_api_key"])

    if cohere_api_key:
        cohere_client = cohere.Client(api_key=cohere_api_key)
    else:
        cohere_client = cohere.Client(api_key=st.secrets["cohere_api_key"])

    # 3. Load the data
    lincoln_data_df = load_lincoln_speech_corpus()       # DataFrame
    voyant_data_df = load_voyant_word_counts()           # DataFrame
    lincoln_index_df = load_lincoln_index_embedded()     # DataFrame

    lincoln_data = lincoln_data_df.to_dict("records")
    lincoln_dict = {item["text_id"]: item for item in lincoln_data}

    # Handling 'corpusTerms'
    if not voyant_data_df.empty and "corpusTerms" in voyant_data_df.columns:
        corpus_terms_json = voyant_data_df.at[0, "corpusTerms"]
        if isinstance(corpus_terms_json, str):
            corpus_terms = json.loads(corpus_terms_json).get("terms", [])
        elif isinstance(corpus_terms_json, dict):
            corpus_terms = corpus_terms_json.get("terms", [])
        else:
            st.error("Unexpected format for 'corpusTerms'")
            corpus_terms = []
    else:
        st.error("No 'corpusTerms' found in voyant_data_df.")
        corpus_terms = []

    # Prepare the Lincoln index DF
    lincoln_index_df["embedding"] = lincoln_index_df["embedding"].apply(
        lambda x: np.array(json.loads(x)) if isinstance(x, str) else np.array(x)
    )
    lincoln_index_df["full_text"] = lincoln_index_df["combined"].apply(extract_full_text)

    def get_source_and_summary(text_id_str):
        """
        Retrieves the source and summary for a given text ID.
        """
        entry = lincoln_dict.get(text_id_str, {})
        return entry.get("source", "Unknown Source"), entry.get("summary", "")

    lincoln_index_df["source"], lincoln_index_df["summary"] = zip(
        *lincoln_index_df["text_id"].apply(get_source_and_summary)
    )

    # 4. Call "Hay" Model to get JSON
    messages_for_model = [
        {"role": "system", "content": keyword_prompt},
        {"role": "user", "content": user_query}
    ]
    try:
        response = openai_client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal:hays-gpt4o:9tFqrYwI",
            messages=messages_for_model,
            temperature=0,
            max_tokens=500
        )
        raw_hay_output = response.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling Hay model: {e}")
        return {}

    # Debug: Display Raw Hay Output
    st.write("**Raw Hay Output:**")
    st.write(raw_hay_output)

    # Parse the Hay model's JSON output
    try:
        hay_output = json.loads(raw_hay_output)
    except json.JSONDecodeError as e:
        st.error(f"Hay model output was not valid JSON: {e}")
        return {}

    # Extract fields
    initial_answer = hay_output.get("initial_answer", "No initial answer available.")
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
            "full_output": raw_hay_output
        })

    # 5. If toggled, do Weighted Keyword Search
    search_results_df = pd.DataFrame()
    if perform_keyword_search and corpus_terms:
        try:
            search_results = search_with_dynamic_weights_expanded(
                user_keywords=model_weighted_keywords,
                corpus_terms={"terms": corpus_terms},  # Adjust as per your JSON structure
                data=lincoln_data,
                year_keywords=model_year_keywords,
                text_keywords=model_text_keywords,
                top_n_results=top_n_results
            )
            search_results_df = pd.DataFrame(search_results)
            # Rename “quote” -> “key_quote” if needed
            if "quote" in search_results_df.columns:
                search_results_df.rename(columns={"quote": "key_quote"}, inplace=True)
            else:
                # If 'quote' column is missing, create 'key_quote' with default values
                search_results_df["key_quote"] = "No key quote available."
        except Exception as e:
            st.error(f"Error during keyword search: {e}")

        # Log the keyword results
        if keyword_results_logger and not search_results_df.empty:
            log_keyword_search_results(
                keyword_results_logger=keyword_results_logger,
                search_results=search_results_df,
                user_query=user_query,
                initial_answer=initial_answer,
                model_weighted_keywords=model_weighted_keywords,
                model_year_keywords=model_year_keywords,
                model_text_keywords=model_text_keywords
            )
        elif keyword_results_logger and search_results_df.empty:
            st.warning("No keyword search results to log.")

    # 6. If toggled, do Semantic Search
    semantic_matches_df = pd.DataFrame()
    user_query_embedding = None
    if perform_semantic_search and not lincoln_index_df.empty:
        try:
            semantic_results, user_query_embedding = search_text(
                lincoln_index_df,
                user_query + " " + initial_answer,  # Combining query with initial answer
                client=openai_client,
                n=top_n_results
            )
            if not semantic_results.empty:
                # For each top match, find a "key quote" by segmenting
                top_segments = []
                for idx, row in semantic_results.iterrows():
                    segments = segment_text(row["full_text"], segment_size=300)
                    # Compare in parallel
                    segment_scores = compare_segments_with_query_parallel(
                        segments, user_query_embedding, openai_client
                    )
                    if segment_scores:
                        best_segment = max(segment_scores, key=lambda x: x[1])
                        top_segments.append(best_segment[0])
                    else:
                        top_segments.append("No relevant segment found.")
                semantic_matches_df = semantic_results.copy()
                semantic_matches_df["TopSegment"] = top_segments

                # Rename "Unnamed: 0" -> "text_id" if present
                if "Unnamed: 0" in semantic_matches_df.columns:
                    semantic_matches_df.rename(columns={"Unnamed: 0": "text_id"}, inplace=True)

                # Log semantic search results
                if semantic_results_logger:
                    log_semantic_search_results(
                        semantic_results_logger=semantic_results_logger,
                        semantic_matches=semantic_matches_df
                    )
            else:
                st.warning("No semantic search results found.")
        except Exception as e:
            st.error(f"Error during semantic search: {e}")

    # 7. Combine & Deduplicate
    combined_df = pd.concat([search_results_df, semantic_matches_df], ignore_index=True)
    if not combined_df.empty:
        combined_df = remove_duplicates(combined_df)
    else:
        st.warning("No results to combine from keyword or semantic searches.")

    # 8. Rerank if toggled
    reranked_df = pd.DataFrame()
    if perform_reranking and not combined_df.empty:
        try:
            # Build doc strings like "Keyword|Text ID: #|Summary|KeyQuote"
            docs_for_cohere = []
            text_id_mapping = {}  # To map back reranked quotes to their metadata

            for _, row in combined_df.iterrows():
                search_type = "Keyword" if pd.notna(row.get("key_quote", "")) else "Semantic"
                text_id_str = row.get("text_id", "")
                summary_str = row.get("summary", "")
                quote_str = row.get("key_quote", row.get("TopSegment", "No key quote available."))
                # Ensure 'quote_str' is a string
                if not isinstance(quote_str, str) or pd.isna(quote_str):
                    quote_str = "No key quote available."
                # Clean the quote to remove excessive whitespace and newlines
                cleaned_quote = ' '.join(quote_str.split())
                # Construct the document string
                doc_string = f"{search_type}|Text ID: {text_id_str}|{summary_str}|{cleaned_quote}"
                docs_for_cohere.append(doc_string)
                # Map the document string back to its metadata
                text_id_mapping[doc_string] = {
                    "Search Type": search_type,
                    "Text ID": text_id_str,
                    "Source": row.get("source", "Unknown Source"),
                    "Summary": summary_str,
                    "Key Quote": quote_str
                }

            # Rerank the documents using Cohere's Rerank API
            reranked_results = rerank_results(
                query=user_query,
                documents=docs_for_cohere,
                api_key=cohere_api_key,
                top_n=10
            )

            if reranked_results:
                reranked_data = []
                for i, item in enumerate(reranked_results):
                    doc_text = item.document if isinstance(item.document, str) else item.document.get('text', '')
                    relevance_score = item.relevance_score
                    metadata = text_id_mapping.get(doc_text, {})

                    reranked_data.append({
                        "Rank": i + 1,
                        "Search Type": metadata.get("Search Type", "N/A"),
                        "Text ID": metadata.get("Text ID", "N/A"),
                        "Source": metadata.get("Source", "Unknown Source"),
                        "Summary": metadata.get("Summary", "N/A"),
                        "Key Quote": metadata.get("Key Quote", "No key quote available."),
                        "Relevance Score": relevance_score
                    })

                reranked_df = pd.DataFrame(reranked_data)

                # Log reranking results
                if reranking_results_logger and not reranked_df.empty:
                    log_reranking_results(
                        reranking_results_logger=reranking_results_logger,
                        reranked_df=reranked_df,
                        user_query=user_query
                    )

                # Format reranked results for Nicolay model input
                formatted_input_for_model = format_reranked_results_for_model_input(reranked_df.to_dict("records"))

                # Display full reranked results in an expander
                with st.expander("**Result Reranking Metadata**"):
                    st.dataframe(reranked_df)
                    st.write("**Formatted Results for Nicolay:**")
                    st.write(formatted_input_for_model)

            else:
                st.warning("No reranked results returned from Cohere.")
        except Exception as e:
            st.error(f"Error in reranking: {e}")

    # 9. Final "Nicolay" Model Call
    nicolay_output = {}
    if perform_reranking and not reranked_df.empty:
        try:
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

            # Send the messages to the Nicolay finetuned model
            second_model_response = openai_client.chat.completions.create(
                model="ft:gpt-4o-mini-2024-07-18:personal:nicolay-gpt4o:9tG7Cypl",
                messages=nicolay_messages,
                temperature=0,
                max_tokens=2000
            )
            raw_nicolay = second_model_response.choices[0].message.content

            # Parse Nicolay's JSON output
            try:
                nicolay_output = json.loads(raw_nicolay)
            except json.JSONDecodeError as e:
                st.error(f"Nicolay model output was not valid JSON: {e}")
                nicolay_output = {}

            if nicolay_output:
                # Displaying the Final Answer
                st.header("Nicolay's Response & Analysis:")

                with st.expander("**How Does This Work?: Nicolay's Response and Analysis**"):
                    st.write(st.session_state["nicolay_model_explainer"])

                with st.expander("**Nicolay's Response**", expanded=True):
                    final_answer = nicolay_output.get("FinalAnswer", {})
                    st.markdown(f"**Response:**\n{final_answer.get('Text', 'No response available')}")
                    if final_answer.get("References"):
                        st.markdown("**References:**")
                        for reference in final_answer["References"]:
                            st.markdown(f"{reference}")

                doc_match_counter = 0
                highlight_success_dict = {}

                highlight_style = """
                <style>
                mark {
                    background-color: #90ee90;
                    color: black;
                }
                </style>
                """

                if "Match Analysis" in nicolay_output:
                    st.markdown(highlight_style, unsafe_allow_html=True)
                    for match_key, match_info in nicolay_output["Match Analysis"].items():
                        text_id = match_info.get("Text ID")
                        formatted_text_id = f"Text #: {text_id}"
                        key_quote = match_info.get("Key Quote", "")

                        speech = lincoln_dict.get(text_id, None)

                        # Increment the counter for each match
                        doc_match_counter += 1

                        # Initialize highlight_success for each iteration
                        highlight_success = False  # Flag to track highlighting success

                        if speech:
                            expander_label = f"**Match {doc_match_counter}**: *{speech['source']}* `{speech['text_id']}`"
                            with st.expander(expander_label, expanded=False):
                                st.markdown(f"**Source:** {speech['source']}")
                                st.markdown(f"**Text ID:** {speech['text_id']}")
                                st.markdown(f"**Summary:**\n{speech['summary']}")

                                # Replace line breaks for HTML display
                                formatted_full_text = speech['full_text'].replace("\\n", "<br>")

                                # Attempt direct highlighting
                                if key_quote and key_quote in speech['full_text']:
                                    formatted_full_text = formatted_full_text.replace(key_quote, f"<mark>{key_quote}</mark>")
                                    highlight_success = True
                                else:
                                    # If direct highlighting fails, use regex-based approach
                                    formatted_full_text = highlight_key_quote(speech['full_text'], key_quote)
                                    formatted_full_text = formatted_full_text.replace("\\n", "<br>")
                                    # Check if highlighting was successful with regex approach
                                    highlight_success = key_quote in formatted_full_text

                                st.markdown(f"**Key Quote:**\n{key_quote}")
                                st.markdown(f"**Full Text with Highlighted Quote:**", unsafe_allow_html=True)
                                st.markdown(formatted_full_text, unsafe_allow_html=True)

                                # Update highlight_success_dict for the current match
                                highlight_success_dict[match_key] = highlight_success
                        else:
                            with st.expander(f"**Match {doc_match_counter}**: Not Found", expanded=False):
                                st.markdown("Full text not found.")
                                highlight_success_dict[match_key] = False  # Indicate failure as text not found

                # Displaying the Analysis Metadata
                with st.expander("**Analysis Metadata**"):
                    # Displaying User Query Analysis
                    if "User Query Analysis" in nicolay_output:
                        st.markdown("**User Query Analysis:**")
                        for key, value in nicolay_output["User Query Analysis"].items():
                            st.markdown(f"- **{key}:** {value}")

                    # Displaying Initial Answer Review
                    if "Initial Answer Review" in nicolay_output:
                        st.markdown("**Initial Answer Review:**")
                        for key, value in nicolay_output["Initial Answer Review"].items():
                            st.markdown(f"- **{key}:** {value}")

                    # Displaying Match Analysis
                    if "Match Analysis" in nicolay_output:
                        st.markdown("**Match Analysis:**")
                        for match_key, match_info in nicolay_output["Match Analysis"].items():
                            st.markdown(f"- **{match_key}:**")
                            for key, value in match_info.items():
                                st.markdown(f"  - {key}: {value}")

                    # Displaying Meta Analysis
                    if "Meta Analysis" in nicolay_output:
                        st.markdown("**Meta Analysis:**")
                        for key, value in nicolay_output["Meta Analysis"].items():
                            st.markdown(f"- **{key}:** {value}")

                    # Displaying Model Feedback
                    if "Model Feedback" in nicolay_output:
                        st.markdown("**Model Feedback:**")
                        for key, value in nicolay_output["Model Feedback"].items():
                            st.markdown(f"- **{key}:** {value}")

                    st.write("**Full Model Output:**")
                    st.write(json.dumps(nicolay_output, indent=2))

                # Log Nicolay model output
                if nicolay_data_logger and nicolay_output:
                    log_nicolay_model_output(
                        nicolay_data_logger=nicolay_data_logger,
                        model_output=nicolay_output,
                        user_query=user_query,
                        highlight_success_dict=highlight_success_dict
                    )

        # 10. Return the final dictionary
        return{
            "hay_output": hay_output,
            "initial_answer": initial_answer,
            "search_results": search_results_df,
            "semantic_results": semantic_matches_df,
            "reranked_results": reranked_df,
            "nicolay_output": nicolay_output,
        }
