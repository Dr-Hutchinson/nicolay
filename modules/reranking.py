# modules/reranking.py

import cohere
import yaml
import pandas as pd
import streamlit as st


def prepare_documents_for_reranking(combined_df, user_query, lincoln_dict=None):
    """
    Prepares documents for Cohere reranking.

    Formats each candidate as a YAML string — the format Cohere recommends for
    structured data (rerank-v4.0 docs). Full chunk text is passed when lincoln_dict
    is provided, giving Cohere the complete corpus content rather than the key quote
    window fragment. Summary is passed in full (no truncation) as supporting context.

    Args:
        combined_df (DataFrame): Combined keyword + semantic + ColBERT results.
        user_query (str): The user's query (unused here but kept for API symmetry).
        lincoln_dict (dict): Optional. Corpus dictionary keyed as "Text #: {text_id}".
                             When provided, full_text is looked up and passed to Cohere
                             instead of the key quote window. Falls back gracefully if
                             lookup fails.

    Returns:
        list: List of dicts with 'text' (YAML string) and 'id' (row index as string).
    """
    documents = []
    for idx, row in combined_df.iterrows():
        try:
            search_type = row.get("search_type", "Unknown")
            if pd.isna(search_type):
                search_type = "Keyword" if "key_quote" in row and pd.notna(row["key_quote"]) else "Semantic"

            text_id = str(row.get("text_id", "")).strip()

            # Full summary — no truncation. Previously capped at 200 chars,
            # which cut off most of the curatorial context Cohere could use.
            summary = str(row.get("summary", "")).strip()

            # Full chunk text lookup from corpus (parallel to dev log #50 for Nicolay).
            # Falls back to key quote window if lincoln_dict not provided or lookup fails.
            full_text = None
            if lincoln_dict is not None:
                lookup_key = f"Text #: {text_id}"
                corpus_entry = lincoln_dict.get(lookup_key, {})
                full_text = corpus_entry.get("full_text", None)

            if not full_text:
                # Fallback: use the key quote window or TopSegment from the search result
                quote_field = "TopSegment" if search_type == "ColBERT" else (
                    "key_quote" if search_type == "Keyword" else "TopSegment"
                )
                full_text = str(row.get(quote_field, "")).strip()

            # YAML format: Cohere rerank-v4.0 documentation recommends YAML strings
            # for structured data. This lets Cohere attend to full_text as the primary
            # relevance signal while treating summary as supporting context, rather than
            # scoring a flat pipe-delimited string where field boundaries are invisible.
            doc_dict = {
                "search_type": search_type,
                "text_id": text_id,
                "summary": summary,
                "full_text": full_text,
            }
            yaml_text = yaml.dump(doc_dict, allow_unicode=True, default_flow_style=False, sort_keys=False)

            documents.append({"text": yaml_text, "id": str(idx)})

        except Exception as e:
            st.error(f"Error preparing document {idx}: {str(e)}")
            continue

    return documents


def rerank_results(query, documents, cohere_client, model='rerank-v4.0-pro', top_n=15):
    """
    Reranks documents using Cohere's reranking API.

    Updated from rerank-v3.5 to rerank-v4.0-pro (released December 2025).
    Documents are now YAML-formatted strings; parsing uses yaml.safe_load()
    rather than pipe splitting.

    Args:
        query (str): The user's query.
        documents (list): List of dicts with 'text' (YAML string) and 'id'.
        cohere_client: Initialized Cohere client.
        model (str): Cohere reranking model. Default: rerank-v4.0-pro.
        top_n (int): Number of top results to return.

    Returns:
        DataFrame: Reranked results with Rank, Search Type, Text ID, Summary,
                   Key Quote (full text), Relevance Score, UserQuery columns.
    """
    try:
        cohere_docs = [d['text'] for d in documents]

        reranked = cohere_client.rerank(
            query=query,
            documents=cohere_docs,
            model=model,
            top_n=top_n
        )

        reranked_data = []
        for rank, result in enumerate(reranked.results, 1):
            try:
                # Get document text — handle both string and dict responses
                if isinstance(result.document, dict):
                    doc_text = result.document.get('text', '')
                else:
                    doc_text = str(result.document)

                # Parse YAML — replaces the previous pipe-split approach.
                # Falls back to empty dict if YAML parse fails (e.g. malformed doc).
                try:
                    doc_parsed = yaml.safe_load(doc_text) or {}
                except yaml.YAMLError:
                    doc_parsed = {}

                search_type = str(doc_parsed.get("search_type", "Unknown")).strip()
                text_id = str(doc_parsed.get("text_id", "Unknown")).strip()
                summary = str(doc_parsed.get("summary", "")).strip()
                full_text = str(doc_parsed.get("full_text", "")).strip()

                reranked_data.append({
                    'Rank': rank,
                    'Search Type': search_type,
                    'Text ID': text_id,
                    'Summary': summary,
                    'Key Quote': full_text,   # 'Key Quote' column retained for downstream
                                              # compatibility; now contains full chunk text.
                    'Relevance Score': float(result.relevance_score),
                    'UserQuery': query
                })

            except Exception as e:
                st.error(f"Error processing reranked result {rank}: {str(e)}")
                st.write(f"Full result object: {result}")
                continue

        if reranked_data:
            return pd.DataFrame(reranked_data)

        st.warning("No results were successfully processed")
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Reranking error: {str(e)}")
        st.exception(e)
        return pd.DataFrame()


def format_reranked_results_for_model_input(reranked_results, max_results=5, lincoln_dict=None):
    """
    Formats reranked results for input to the Nicolay model.

    Args:
        reranked_results (list): List of dictionaries containing reranked results
        max_results (int): Maximum number of results to format (v3 uses k=5)
        lincoln_dict (dict): Optional. Corpus dictionary keyed as "Text #: {text_id}",
                             used to look up full chunk text for each match.
                             If provided, full corpus text is passed instead of the
                             key quote window, resolving the key quote bottleneck (dev log #50).

    Returns:
        str: Formatted string of results
    """
    formatted_results = []

    # Take only the top N results
    for idx, result in enumerate(reranked_results[:max_results], 1):
        try:
            text_id = result.get('Text ID', 'Unknown')

            # Resolve full chunk text from corpus if lincoln_dict is available (dev log #50).
            # Falls back to the key quote window if lookup fails or lincoln_dict is not provided.
            full_text = None
            if lincoln_dict is not None:
                lookup_key = f"Text #: {text_id}"
                corpus_entry = lincoln_dict.get(lookup_key, {})
                full_text = corpus_entry.get('full_text', None)

            if full_text:
                text_field_label = "Full Text (select the most relevant passage to quote directly)"
                text_field_value = full_text
            else:
                # Fallback: key quote window (pre-dev-log-#50 behavior)
                text_field_label = "Full Text (select the most relevant passage to quote directly)"
                text_field_value = result.get('Key Quote', 'No quote')

            formatted_entry = (
                f"Match {idx}: "
                f"Search Type - {result.get('Search Type', 'Unknown')}, "
                f"Text ID - {text_id}, "
                f"Summary (curatorial description only — not quotable corpus text) - {result.get('Summary', 'No summary')}, "
                f"{text_field_label} - {text_field_value}, "
                f"Relevance Score - {result.get('Relevance Score', 0.0):.2f}"
            )
            formatted_results.append(formatted_entry)

        except Exception as e:
            st.error(f"Error formatting result {idx}: {str(e)}")
            continue

    return "\n\n".join(formatted_results) if formatted_results else "No results to format"
