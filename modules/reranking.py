# modules/reranking.py

import cohere
import pandas as pd
import streamlit as st

# In reranking.py
def prepare_documents_for_reranking(combined_df, user_query):
    """
    Prepares documents for Cohere's reranking with simpler format.
    """
    documents = []

    for idx, row in combined_df.iterrows():
        try:
            # Determine search type
            search_type = "Keyword" if "key_quote" in row and pd.notna(row["key_quote"]) else "Semantic"

            # Get text components
            text_id = str(row.get("text_id", "")).strip()
            summary = str(row.get("summary", ""))[:200].strip()
            quote = str(row.get("key_quote" if search_type == "Keyword" else "TopSegment", "")).strip()

            # Create formatted text
            formatted_text = f"{search_type}|{text_id}|{summary}|{quote}"

            # Add to documents list
            doc = {
                "text": formatted_text,
                "id": str(idx)
            }

            st.write(f"Prepared document {idx}:", doc)  # Debug output
            documents.append(doc)

        except Exception as e:
            st.error(f"Error preparing document {idx}: {str(e)}")
            continue

    return documents

def rerank_results(query, documents, cohere_client, model='rerank-english-v2.0', top_n=10):
    """
    Reranks documents using Cohere's reranking API with improved error handling.
    """
    try:
        # Convert documents list to correct format for Cohere
        cohere_docs = [d['text'] for d in documents]

        # Call Cohere API
        reranked = cohere_client.rerank(
            query=query,
            documents=cohere_docs,
            model=model,
            top_n=top_n
        )

        # Process results
        reranked_data = []
        for rank, result in enumerate(reranked.results, 1):
            try:
                # Debug the result structure
                st.write(f"Result {rank} document type: {type(result.document)}")
                st.write(f"Result {rank} document content: {result.document}")

                # Get the document text - handle both string and dict cases
                if isinstance(result.document, dict):
                    doc_text = result.document.get('text', '')
                else:
                    doc_text = str(result.document)

                # Split the document text into parts
                doc_parts = doc_text.split('|')
                if len(doc_parts) >= 4:
                    search_type = doc_parts[0].strip()
                    text_id = doc_parts[1].strip()
                    summary = doc_parts[2].strip()
                    quote = doc_parts[3].strip()

                    reranked_data.append({
                        'Rank': rank,
                        'Search Type': search_type,
                        'Text ID': text_id,
                        'Summary': summary,
                        'Key Quote': quote,
                        'Relevance Score': float(result.relevance_score),
                        'UserQuery': query
                    })

                    st.write(f"Successfully processed result {rank}")

            except Exception as e:
                st.error(f"Error processing reranked result {rank}: {str(e)}")
                st.write(f"Full result object: {result}")  # Additional debugging
                continue

        # Create DataFrame from results
        if reranked_data:
            df = pd.DataFrame(reranked_data)
            st.write("Created DataFrame with shape:", df.shape)  # Debug output
            return df

        st.warning("No results were successfully processed")
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Reranking error: {str(e)}")
        st.exception(e)  # Show full traceback
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
