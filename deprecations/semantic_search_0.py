import numpy as np
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

def get_embedding(text, client, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def semantic_search(user_query, df, client, top_n=5):
    user_query_embedding = get_embedding(user_query, client)
    df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, user_query_embedding))
    top_n_results = df.sort_values("similarities", ascending=False).head(top_n)
    return top_n_results, user_query_embedding

def search_text(df, user_query, client, n=5):
    user_query_embedding = get_embedding(user_query, client)
    df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, user_query_embedding))
    top_n = df.sort_values("similarities", ascending=False).head(n)
    top_n["UserQuery"] = user_query  # Add 'UserQuery' column to the DataFrame
    return top_n, user_query_embedding

def segment_text(text, segment_size=100):
    words = text.split()
    return [' '.join(words[i:i+segment_size]) for i in range(0, len(words), segment_size)]

def compare_segments_with_query_parallel(segments, query_embedding, client):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(get_embedding, segment, client) for segment in segments]
        segment_embeddings = [future.result() for future in futures]
        return [(segments[i], cosine_similarity(segment_embeddings[i], query_embedding)) for i in range(len(segments))]
