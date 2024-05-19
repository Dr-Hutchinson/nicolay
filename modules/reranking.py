import cohere

def rerank_results(query, documents, api_key, model='rerank-english-v2.0', top_n=10):
    """
    Reranks a list of documents based on their relevance to a given query using Cohere's rerank model.

    Parameters:
    query (str): The query string.
    documents (list of str): List of documents to be reranked.
    api_key (str): API key for Cohere.
    model (str): The rerank model to be used. Default is 'rerank-english-v2.0'.
    top_n (int): Number of top results to return. Default is 10.

    Returns:
    list: Reranked results.
    """
    co = cohere.Client(api_key)
    st.write("Documents sent to Cohere for reranking:")
    st.write(documents)  # Debug print to check documents
    reranked_response = co.rerank(
        model=model,
        query=query,
        documents=documents,
        top_n=top_n
    )
    #st.write("Reranked response from Cohere:")  # Debug print for the response
    #st.write(reranked_response)
    print("Reranked response from Cohere:")  # Debug print for the response
    print(reranked_response)
    return reranked_response.results

def format_reranked_results_for_model_input(reranked_results):
    """
    Formats the reranked results for model input, limiting to the top 3 results.

    Parameters:
    reranked_results (list): List of reranked results.

    Returns:
    str: Formatted string of top results.
    """
    formatted_results = []
    # Limiting to the top 3 results
    top_three_results = reranked_results[:3]
    for result in top_three_results:
        formatted_entry = (
            f"Match {result['index'] + 1}: "  # Assuming 'index' is provided by the rerank response
            f"Search Type - {result['document']['Search Type']}, "
            f"Text ID - {result['document']['Text ID']}, "
            f"Source - {result['document']['Source']}, "
            f"Summary - {result['document']['Summary']}, "
            f"Key Quote - {result['document']['Key Quote']}, "
            f"Relevance Score - {result['relevance_score']:.2f}"
        )
        formatted_results.append(formatted_entry)
    return "\n\n".join(formatted_results)
