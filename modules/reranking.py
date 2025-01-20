# modules/reranking.py

import cohere
import pandas as pd
import streamlit as st

def prepare_documents_for_reranking(combined_df, user_query, max_length=1000):
    """
    Prepares documents for Cohere's reranking in the correct format.
    """
    documents = []

    for idx, row in combined_df.iterrows():
        # Determine search type
        search_type = "Keyword" if "weighted_score" in row else "Semantic"

        # Get text content (either quote or top segment)
        text_content = row.get('quote', row.get('TopSegment', ''))
        if len(text_content) > max_length:
            text_content = text_content[:max_length] + "..."

        # Format text ID consistently
        text_id = row.get('text_id', row.get('Unnamed: 0', ''))
        if not text_id.startswith('Text #:'):
            text_id = f"Text #: {text_id}"

        # Create summary text
        summary = row.get('summary', '')
        if len(summary) > 200:  # Limit summary length
            summary = summary[:200] + "..."

        # Create a clean document string
        doc = {
            "text": f"{search_type}|{text_id}|{summary}|{text_content}",
            "id": str(idx)  # Required by Cohere
        }
        documents.append(doc)

    return documents

def rerank_results(query, documents, cohere_client, model='rerank-english-v2.0', top_n=10):
    """
    Reranks documents using Cohere's reranking API.
    """
    try:
        reranked = cohere_client.rerank(
            query=query,
            documents=documents,
            model=model,
            top_n=top_n
        )

        # Convert results to DataFrame
        reranked_data = []
        for rank, result in enumerate(reranked.results, 1):
            # Parse the document text
            doc_parts = result.document.text.split('|')
            if len(doc_parts) >= 4:
                search_type, text_id, summary, quote = doc_parts

                reranked_data.append({
                    'Rank': rank,
                    'Search Type': search_type,
                    'Text ID': text_id.strip(),
                    'Summary': summary.strip(),
                    'Key Quote': quote.strip(),
                    'Relevance Score': result.relevance_score
                })

        return pd.DataFrame(reranked_data)

    except Exception as e:
        st.error(f"Reranking error: {str(e)}")
        return pd.DataFrame()

def format_reranked_results_for_model_input(reranked_results):
    """
    Formats reranked results for input to the Nicolay model.
    """
    formatted_results = []
    top_three = reranked_results.head(3)

    for _, row in top_three.iterrows():
        formatted_entry = (
            f"Match {row['Rank']}: "
            f"Search Type - {row['Search Type']}, "
            f"Text ID - {row['Text ID']}, "
            f"Source - {row.get('Source', 'N/A')}, "
            f"Summary - {row['Summary']}, "
            f"Key Quote - {row['Key Quote']}, "
            f"Relevance Score - {row['Relevance Score']:.2f}"
        )
        formatted_results.append(formatted_entry)

    return "\n\n".join(formatted_results)
