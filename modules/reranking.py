# modules/reranker.py

import cohere
import streamlit as st
import re
import pandas as pd

def clean_text(text):
    """
    Cleans the input text by removing non-printable characters,
    replacing multiple spaces with a single space, and stripping whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Remove non-printable characters
    text = ''.join(filter(lambda x: x.isprintable(), text))
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def rerank_results(query, documents, api_key, model='rerank-english-v2.0', top_n=10):
    """
    Reranks a list of documents based on their relevance to a given query using Cohere's rerank model.

    Parameters:
    - query (str): The query string.
    - documents (list of str): List of documents to be reranked.
    - api_key (str): API key for Cohere.
    - model (str): The rerank model to be used. Default is 'rerank-english-v2.0'.
    - top_n (int): Number of top results to return. Default is 10.

    Returns:
    - list: Reranked results.
    """
    co = cohere.Client(api_key)

    # Clean documents
    cleaned_documents = [clean_text(doc) for doc in documents]

    # Debug: Display documents being sent for reranking
    st.write("### Documents sent to Cohere for reranking:")
    for idx, doc in enumerate(cleaned_documents):
        display_doc = doc[:100] + "..." if len(doc) > 100 else doc
        st.write(f"Document {idx + 1}: {display_doc}")

    # Validate that all documents are strings and non-empty
    non_string_docs = [doc for doc in cleaned_documents if not isinstance(doc, str)]
    if non_string_docs:
        st.error(f"Some documents are not strings: {non_string_docs}")
        return []

    empty_docs = [doc for doc in cleaned_documents if not doc]
    if empty_docs:
        st.error(f"Some documents are empty strings: {empty_docs}")
        return []

    # Additional Debugging: Check types and lengths
    st.write("### Documents types:")
    st.write([type(doc) for doc in cleaned_documents])

    st.write("### Number of documents:")
    st.write(len(cleaned_documents))

    try:
        reranked_response = co.rerank(
            model=model,
            query=query,
            documents=cleaned_documents,
            top_n=top_n
        )

        # Debug: Display the reranked response
        st.write("### Reranked Response from Cohere:")
        st.json(reranked_response)

        if reranked_response is None or reranked_response.results is None:
            st.error("Reranking response or results are None.")
            return []

        # Validate each result
        valid_results = []
        for result in reranked_response.results:
            # Check if 'document' and 'relevance_score' exist
            if hasattr(result, 'document') and hasattr(result, 'relevance_score'):
                # Ensure 'document' is a string or bytes-like
                if isinstance(result.document, (str, bytes)):
                    valid_results.append(result)
                else:
                    st.error(f"Invalid document type in reranked results: {result.document}")
            else:
                st.error(f"Missing expected attributes in reranked result: {result}")

        if not valid_results:
            st.error("No valid reranked results returned.")
            return []

        return valid_results
    except cohere.CohereAPIError as e:
        st.error(f"Cohere API error: {e}")
        return []
    except Exception as e:
        st.error(f"Error in reranking: {str(e)}")
        return []

def format_reranked_results_for_model_input(reranked_results):
    """
    Formats the reranked results for model input, limiting to the top 3 results.

    Parameters:
    - reranked_results (list): List of reranked results as Cohere's result objects.

    Returns:
    - str: Formatted string of top results.
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
        key_quote = result.get('Key Quote', "No key quote available.")
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

# Optional: Test Function for Reranker
def test_rerank():
    """
    A test function to verify reranker functionality with sample data.
    """
    import streamlit as st

    test_documents = [
        "This is a test quote 1.",
        "This is a test quote 2.",
        "This is a test quote 3."
    ]
    test_query = "Test query for reranking"
    test_api_key = st.secrets["cohere_api_key"]  # Ensure your API key is stored securely

    reranked_results = rerank_results(test_query, test_documents, test_api_key)
    st.write("### Reranked Results:")
    st.write(reranked_results)

    if reranked_results:
        formatted = format_reranked_results_for_model_input(reranked_results)
        st.write("### Formatted Results for Nicolay:")
        st.write(formatted)
    else:
        st.write("No reranked results returned.")

# Uncomment the following lines to run the test function independently
# if __name__ == "__main__":
#     import streamlit as st
#     test_rerank()
