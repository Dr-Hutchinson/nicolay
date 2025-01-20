# rag_pipeline.py

import json
import pandas as pd
import streamlit as st
import numpy as np

# Import modules that handle each step:
from modules.prompt_loader import load_prompts  # If you want to ensure they're loaded here
from modules.data_logging import (
    log_keyword_search_results,
    log_semantic_search_results,
    log_reranking_results,
    log_nicolay_model_output,
)
from modules.keyword_search import search_with_dynamic_weights_expanded
from modules.semantic_search import (
    search_text,
    compare_segments_with_query_parallel,
    segment_text,
    get_embedding,
)
from modules.misc_helpers import remove_duplicates, highlight_key_quote, get_source_and_summary
from modules.reranking import rerank_results, format_reranked_results_for_model_input

# You might also import or pass in these clients, depending on your architecture:
from openai import OpenAI
import cohere
from google.oauth2 import service_account
import pygsheets
from json import JSONDecodeError


def run_rag_pipeline(
    user_query,
    perform_keyword_search=True,
    perform_semantic_search=True,
    perform_reranking=True,
    user_weighted_keywords=None,
    user_year_keywords=None,
    user_text_keywords=None,
    keyword_json_path='data/voyant_word_counts.json',
    lincoln_speeches_json='data/lincoln_speech_corpus.json',
    lincoln_embedded_csv='lincoln_index_embedded.csv',
    top_n_results=5,
    # Logging classes/objects:
    hays_data_logger=None,
    keyword_results_logger=None,
    semantic_results_logger=None,
    reranking_results_logger=None,
    nicolay_data_logger=None,
    # API keys or clients:
    openai_api_key=None,
    cohere_api_key=None,
    # Google Sheets client:
    gc=None,
    # Extra toggles or arguments if needed...
):
    """
    Orchestrates the entire RAG pipeline for a user query.

    Returns:
        dict: A dictionary of outputs (Hay’s initial answer,
              search results, reranked results, Nicolay’s final answer, etc.)
    """

    # 1. Load prompts if needed (or rely on them being in st.session_state)
    load_prompts()  # If not already called in main script
    keyword_prompt = st.session_state['keyword_model_system_prompt']
    response_prompt = st.session_state['response_model_system_prompt']

    # 2. Initialize clients if not provided
    #    (You might also do this in your main script and pass them in.)
    if openai_api_key:
        st.secrets["openai_api_key"] = openai_api_key
    if cohere_api_key:
        st.secrets["cohere_api_key"] = cohere_api_key

    openai_client = OpenAI(api_key=openai_api_key)
    cohere_client = cohere.Client(api_key=cohere_api_key)

    # 3. Call "Hay" model to get initial answer + recommended keywords
    messages_for_model = [
        {"role": "system", "content": keyword_prompt},
        {"role": "user", "content": user_query}
    ]
    response = openai_client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal:hays-gpt4o:9tFqrYwI",
        messages=messages_for_model,
        temperature=0,
        max_tokens=2000
    )
    msg = response.choices[0].message.content
    hay_output_str = response.choices[0].message.content
    st.write("Raw Hays Output:")
    st.write(hay_output_str)
    st.write("REPR output:")
    st.write(repr(hay_output_str))

    # Parse the JSON
    try:
        hay_output = json.loads(msg)
        initial_answer = hay_output.get("initial_answer", "")
        model_weighted_keywords = hay_output.get("weighted_keywords", {})
        model_year_keywords = hay_output.get("year_keywords", [])
        model_text_keywords = hay_output.get("text_keywords", [])
    except JSONDecodeError as e:
        st.error(f"Failed to parse JSON: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()
    #except:
    #    st.error("Hay model output was not valid JSON.")
    #    return {}

    # Log the “Hay” data
    if hays_data_logger:
        hays_data = {
            'query': user_query,
            'initial_answer': initial_answer,
            'weighted_keywords': model_weighted_keywords,
            'year_keywords': model_year_keywords,
            'text_keywords': model_text_keywords,
            'full_output': msg
        }
        hays_data_logger.record_api_outputs(hays_data)

    # 4. Decide final keywords/filters: user override vs. model
    if user_weighted_keywords and len(user_weighted_keywords) > 0:
        final_weighted_keywords = user_weighted_keywords
        final_year_keywords = user_year_keywords.split(',') if user_year_keywords else []
        final_text_keywords = user_text_keywords if user_text_keywords else []
    else:
        final_weighted_keywords = model_weighted_keywords
        final_year_keywords = model_year_keywords
        final_text_keywords = model_text_keywords

    # 5. If toggled, perform Weighted Keyword Search
    keyword_search_results_df = pd.DataFrame()
    if perform_keyword_search:
        import os
        if not os.path.exists(keyword_json_path):
            st.error(f"Keyword JSON not found at {keyword_json_path}")
        else:
            # The 'json_data' parameter in search_with_dynamic_weights_expanded
            # expects the loaded JSON from voyante.
            import json
            with open(keyword_json_path, 'r') as f:
                corpus_json = json.load(f)

            # Now perform the search
            results_list = search_with_dynamic_weights_expanded(
                user_keywords=final_weighted_keywords,
                json_data=corpus_json,
                year_keywords=final_year_keywords,
                text_keywords=final_text_keywords,
                top_n_results=top_n_results
            )

            if results_list:
                # Convert list to DataFrame
                keyword_search_results_df = pd.DataFrame(results_list)
                # rename the “quote” column so we can unify with semantic
                keyword_search_results_df.rename(
                    columns={"quote": "key_quote"},
                    inplace=True
                )

            # Log the results
            if keyword_results_logger and not keyword_search_results_df.empty:
                log_keyword_search_results(
                    keyword_results_logger,
                    search_results=keyword_search_results_df,
                    user_query=user_query,
                    initial_answer=initial_answer,
                    model_weighted_keywords=final_weighted_keywords,
                    model_year_keywords=final_year_keywords,
                    model_text_keywords=final_text_keywords
                )

    # 6. If toggled, perform Semantic Search
    semantic_matches_df = pd.DataFrame()
    user_query_embedding = None

    if perform_semantic_search:
        # Load the embedded CSV
        if not lincoln_embedded_csv:
            st.error("No embedded CSV path provided.")
        else:
            df = pd.read_csv(lincoln_embedded_csv)
            # Convert stored embeddings from string to np.array
            # (depending on how you saved them)
            df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))

            # Do the actual search
            semantic_results, user_query_embedding = search_text(df, user_query + initial_answer, openai_client, n=top_n_results)

            # For each top match, find a “key quote” by segmenting
            top_segments = []
            from semantic_search import segment_text, compare_segments_with_query_parallel

            if not semantic_results.empty:
                for idx, row in semantic_results.iterrows():
                    segments = segment_text(row['full_text'], segment_size=300)  # or any size
                    best_segment, best_score = None, 0
                    segment_scores = compare_segments_with_query_parallel(segments, user_query_embedding, openai_client)
                    best_segment = max(segment_scores, key=lambda x: x[1])
                    top_segments.append(best_segment[0])

                semantic_results["TopSegment"] = top_segments
                # rename columns for consistency
                semantic_results.rename(columns={"Unnamed: 0": "text_id"}, inplace=True)

                semantic_matches_df = semantic_results

            # Log semantic results
            if semantic_results_logger and not semantic_matches_df.empty:
                log_semantic_search_results(semantic_results_logger, semantic_matches_df, initial_answer)

    # 7. Combine results & deduplicate
    combined_results = pd.DataFrame()
    if not keyword_search_results_df.empty or not semantic_matches_df.empty:
        from misc_helpers import remove_duplicates
        if not keyword_search_results_df.empty and not semantic_matches_df.empty:
            combined_results = remove_duplicates(keyword_search_results_df, semantic_matches_df)
        else:
            # If one of them is empty, just use the other
            combined_results = keyword_search_results_df if not keyword_search_results_df.empty else semantic_matches_df

    # 8. Rerank with Cohere (if toggled)
    reranked_df = pd.DataFrame()
    if perform_reranking and not combined_results.empty:
        # We must format combined results so that each row is a doc for Cohere
        # For example:
        documents_for_cohere = []
        for idx, row in combined_results.iterrows():
            search_type = "Keyword" if row["text_id"] in keyword_search_results_df["text_id"].values else "Semantic"
            doc_string = (
                f"{search_type}|Text ID: {row['text_id']}|"
                f"{row.get('summary', '')}|"
                f"{row.get('key_quote', row.get('TopSegment', ''))}"
            )
            documents_for_cohere.append(doc_string)

        # Call rerank
        from reranker import rerank_results
        results = rerank_results(
            query=user_query,
            documents=documents_for_cohere,
            api_key=cohere_api_key,
            top_n=10
        )

        # Convert results into a DataFrame
        if results:
            # The reranker returns a list of objects with .document, .relevance_score, and index
            for idx, r in enumerate(results):
                # Example parsing:
                raw_text = r.document['text']  # or r.document if it’s just a string
                data_parts = raw_text.split("|")
                # data_parts[0] = "Keyword" or "Semantic"
                # data_parts[1] = "Text ID: #"
                # data_parts[2] = summary
                # data_parts[3] = quote
                # etc...
                # Parse into row form.

            # Suppose you build a list of dicts:
            #   { "Rank": ..., "Search Type":..., "Text ID":..., etc. }
            # Then turn that into reranked_df = pd.DataFrame(...)

            # Example (simplified):
            reranked_data = []
            for i, item in enumerate(results):
                doc_parts = item.document['text'].split("|")
                search_type = doc_parts[0].strip()
                text_id_part = doc_parts[1].strip()
                # Extract the actual ID
                text_id_str = text_id_part.split(":")[-1].strip()

                summary = doc_parts[2].strip() if len(doc_parts) > 2 else ""
                quote = doc_parts[3].strip() if len(doc_parts) > 3 else ""

                # Relevance Score:
                score = item.relevance_score

                reranked_data.append({
                    "Rank": i + 1,
                    "Search Type": search_type,
                    "Text ID": text_id_str,
                    "Summary": summary,
                    "Key Quote": quote,
                    "Relevance Score": score
                })

            reranked_df = pd.DataFrame(reranked_data)

            # Log
            if reranking_results_logger:
                log_reranking_results(reranking_results_logger, reranked_df, user_query)

    # 9. Final Call to “Nicolay” Model
    final_nicolay_output = {}
    if perform_reranking and not reranked_df.empty:
        from reranker import format_reranked_results_for_model_input
        formatted_for_nicolay = format_reranked_results_for_model_input(reranked_df.to_dict('records'))
        # Build your message
        messages_for_second_model = [
            {"role": "system", "content": response_prompt},
            {"role": "user", "content": f"User Query: {user_query}\n\n"
                                        f"Initial Answer: {initial_answer}\n\n"
                                        f"{formatted_for_nicolay}"}
        ]
        second_model_response = openai_client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal:nicolay-gpt4o:9tG7Cypl",
            messages=messages_for_second_model,
            temperature=0,
            max_tokens=2000
        )
        nicolay_response = second_model_response.choices[0].message.content
        try:
            final_nicolay_output = json.loads(nicolay_response)
        except:
            st.error("Nicolay model output was not valid JSON.")

        if nicolay_data_logger and final_nicolay_output:
            # You may also want a highlight_success_dict
            highlight_success_dict = {}
            log_nicolay_model_output(
                nicolay_data_logger,
                model_output=final_nicolay_output,
                user_query=user_query,
                initial_answer=initial_answer,
                highlight_success_dict=highlight_success_dict
            )

    # 10. Return a dictionary of everything that might be displayed or further processed
    return {
        "hay_output": hay_output,
        "keyword_results": keyword_search_results_df,
        "semantic_results": semantic_matches_df,
        "combined_results": combined_results,
        "reranked_results": reranked_df,
        "nicolay_output": final_nicolay_output
    }
