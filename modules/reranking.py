# modules/reranker.py

import cohere
import streamlit as st

def rerank_results(query, documents, api_key, model='rerank-english-v2.0', top_n=10):
    """
    Reranks a list of documents based on their relevance to a given query using Cohere's rerank model.
    """
    co = cohere.Client(api_key)

    # Debug: Display documents being sent for reranking
    st.write("### Documents sent to Cohere for reranking:")
    st.write(documents)

    # Validate that all documents are strings
    non_string_docs = [doc for doc in documents if not isinstance(doc, str)]
    if non_string_docs:
        st.error(f"Some documents are not strings: {non_string_docs}")
        return []

    # Validate that no documents are empty strings
    empty_docs = [doc for doc in documents if not doc.strip()]
    if empty_docs:
        st.error(f"Some documents are empty strings: {empty_docs}")
        return []

    # Additional Debugging: Check types and lengths
    st.write("### Documents types:")
    st.write([type(doc) for doc in documents])

    st.write("### Number of documents:")
    st.write(len(documents))

    try:
        reranked_response = co.rerank(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n
        )

        # Debug: Display the reranked response
        st.write("### Reranked Response from Cohere:")
        st.write(reranked_response)

        if reranked_response is None or reranked_response.results is None:
            st.error("Reranking response or results are None.")
            return []

        return reranked_response.results
    except cohere.CohereAPIError as e:
        st.error(f"Cohere API error: {e}")
        return []
    except Exception as e:
        st.error(f"Error in reranking: {str(e)}")
        return []

def format_reranked_results_for_model_input(reranked_results):
    """
    Formats the reranked results for model input, limiting to the top 3 results.
    """
    formatted_results = []

    # Limiting to the top 3 results
    top_three_results = reranked_results[:3]

    for result in top_three_results:
        # Extract fields safely with default values
        rank = result.get('Rank', 'N/A')
        search_type = result.get('Search Type', 'N/A')
        text_id = result.get('Text ID', 'N/A')
        source = result.get('Source', 'N/A')
        summary = result.get('Summary', 'N/A')
        key_quote = result.get('Key Quote', 'N/A')
        relevance_score = result.get('Relevance Score', 0.0)

        # Format the entry
        formatted_entry = (
            f"Match {rank}: "
            f"Search Type - {search_type}, "
            f"Text ID - {text_id}, "
            f"Source - {source}, "
            f"Summary - {summary}, "
            f"Key Quote - {key_quote}, "
            f"Relevance Score - {relevance_score:.2f}"
        )

        formatted_results.append(formatted_entry)

    # Join all formatted entries with double newlines
    return "\n\n".join(formatted_results)
