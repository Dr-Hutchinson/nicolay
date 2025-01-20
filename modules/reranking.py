# modules/reranking.py

import cohere
import pandas as pd
import streamlit as st

# In reranking.py
def prepare_documents_for_reranking(combined_df, user_query, max_length=1000):
    """
    Prepares documents for Cohere's reranking in the correct format.
    """
    documents = []

    for idx, row in combined_df.iterrows():
        try:
            # Determine search type and handle potential missing values
            search_type = "Keyword" if pd.notna(row.get('key_quote')) else "Semantic"

            # Get text ID, ensuring it's a string
            text_id = str(row.get('text_id', ''))
            if not text_id.startswith('Text #:'):
                text_id = f"Text #: {text_id}"

            # Get summary, ensuring it's a string
            summary = str(row.get('summary', ''))
            if len(summary) > 200:
                summary = summary[:200] + "..."

            # Get quote content, ensuring it's a string
            quote = str(row.get('key_quote', row.get('TopSegment', '')))
            if len(quote) > max_length:
                quote = quote[:max_length] + "..."

            # Create document object
            doc = {
                "text": f"{search_type}|{text_id}|{summary}|{quote}",
                "id": str(idx)
            }

            # Verify document is valid before adding
            if all(isinstance(v, str) for v in [doc["text"], doc["id"]]) and len(doc["text"]) > 0:
                documents.append(doc)

        except Exception as e:
            st.error(f"Error preparing document {idx}: {str(e)}")
            continue

    # Verify we have valid documents
    if not documents:
        raise ValueError("No valid documents created for reranking")

    return documents

def rerank_results(query, documents, cohere_client, model='rerank-english-v2.0', top_n=10):
    """
    Reranks documents using Cohere's reranking API with improved error handling.
    """
    try:
        # Verify inputs
        if not documents:
            raise ValueError("No documents provided for reranking")

        if not isinstance(query, str) or len(query.strip()) == 0:
            raise ValueError("Invalid query")

        # Verify document format
        for doc in documents:
            if not isinstance(doc, dict) or 'text' not in doc or 'id' not in doc:
                raise ValueError(f"Invalid document format: {doc}")

        # Call Cohere API
        reranked = cohere_client.rerank(
            query=query,
            documents=documents,
            model=model,
            top_n=top_n
        )

        # Process results
        reranked_data = []
        for rank, result in enumerate(reranked.results, 1):
            try:
                # Parse the document text safely
                doc_parts = result.document.text.split('|')
                if len(doc_parts) >= 4:
                    search_type, text_id, summary, quote = doc_parts

                    reranked_data.append({
                        'Rank': rank,
                        'Search Type': search_type.strip(),
                        'Text ID': text_id.strip(),
                        'Summary': summary.strip(),
                        'Key Quote': quote.strip(),
                        'Relevance Score': float(result.relevance_score)
                    })
            except Exception as e:
                st.error(f"Error processing reranked result {rank}: {str(e)}")
                continue

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
