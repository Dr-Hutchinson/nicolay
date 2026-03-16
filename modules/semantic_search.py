# semantic_search.py
# Revised 2026-03-16
#
# Changes from previous version:
#
# [MEDIUM] Vectorized cosine similarity (search_text, semantic_search):
#   Previously: df['embedding'].apply(lambda x: cosine_similarity(x, query_vec))
#   — a Python-level loop calling the custom cosine_similarity() function once
#   per corpus row (886 calls per search). Each call recomputed the corpus
#   embedding's L2 norm even though corpus norms never change across queries.
#
#   Now: build_corpus_matrix() pre-normalizes all corpus embeddings to unit
#   vectors once at app startup. At search time a single matrix multiply
#   (corpus_matrix_normalized @ query_vec_normalized) replaces all 886 dot
#   products in one BLAS call. Results are identical; overhead is eliminated.
#
#   Integration: call build_corpus_matrix(df) once after loading lincoln_index_df,
#   cache the result in st.session_state (or a module-level variable), and pass
#   it as corpus_matrix to search_text() and semantic_search(). If corpus_matrix
#   is not supplied, both functions fall back to the original .apply() path so
#   existing call sites continue to work without modification.
#
# [MEDIUM] Mutation-safe dataframe handling (search_text, semantic_search):
#   Previously: df["similarities"] = ... mutated the shared dataframe in place.
#   The 'similarities' column would persist on lincoln_index_df between searches,
#   and any code iterating over the dataframe outside a search context would see
#   stale similarity scores from the previous query.
#
#   Now: similarities are computed as a local numpy array and written only to a
#   .copy() of the top-n slice before returning. The shared dataframe is never
#   modified. The returned DataFrame is a proper independent copy.
#
# [INFO] Dead code removed (segment_text, compare_segments_with_query_parallel):
#   These functions implemented a per-segment parallel embedding approach that
#   is not used anywhere in the main pipeline (superseded by the chunked corpus).
#   Removed to keep the module clean. If needed for future work, they are
#   preserved in the git history / original file.
#
# [UNCHANGED] get_embedding() default model left as "text-embedding-ada-002".
#   The active deployment overrides this at the call site. Do not change the
#   default here without also updating all call sites in rag_pipeline.py and
#   the main app.
#
# [UNCHANGED] cosine_similarity() retained for use by external callers (e.g.
#   the quote verification pipeline) that call it directly with two vectors.

import numpy as np
from openai import OpenAI


def get_embedding(text, client, model="text-embedding-ada-002"):
    """Embed a single text string via the OpenAI embeddings API.

    model default is left as ada-002; the active deployment overrides this
    at the call site via rag_pipeline.py. Do not change this default without
    updating all call sites.
    """
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)


def cosine_similarity(vec1, vec2):
    """Cosine similarity between two 1-D numpy arrays.

    Retained for direct use by external callers (e.g. quote verification).
    The main search path uses the vectorized matrix multiply instead.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def build_corpus_matrix(df, embedding_col="embedding"):
    """Pre-normalize corpus embeddings into a unit-vector matrix for fast search.

    Call this ONCE after loading lincoln_index_df and cache the result.
    Passing the returned matrix to search_text() and semantic_search() enables
    the vectorized path (single matrix multiply instead of 886 .apply() calls).

    Returns:
        corpus_matrix_normalized: np.ndarray of shape (n_chunks, embedding_dim),
            each row is the unit-normalized embedding for that chunk.
        row_index: list of DataFrame index labels in the same order as the matrix
            rows, used to align similarity scores back to the DataFrame.
    """
    embeddings = np.stack(df[embedding_col].values)          # (n, 1536)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Guard against zero-norm embeddings (should not occur but avoids NaN)
    norms = np.where(norms == 0, 1.0, norms)
    corpus_matrix_normalized = embeddings / norms             # unit vectors
    row_index = list(df.index)
    return corpus_matrix_normalized, row_index


def _vectorized_similarities(query_vec, corpus_matrix_normalized):
    """Return cosine similarities for all corpus rows via a single matrix multiply.

    Args:
        query_vec: 1-D np.ndarray, raw (unnormalized) query embedding.
        corpus_matrix_normalized: (n, dim) unit-vector matrix from build_corpus_matrix().

    Returns:
        similarities: 1-D np.ndarray of shape (n,), cosine similarity of each
            corpus row to the query.
    """
    norm = np.linalg.norm(query_vec)
    if norm == 0:
        return np.zeros(corpus_matrix_normalized.shape[0])
    query_normalized = query_vec / norm
    return corpus_matrix_normalized @ query_normalized        # (n,) — one BLAS call


def search_text(df, user_query, client, n=5, model="text-embedding-ada-002",
                corpus_matrix=None, row_index=None):
    """Semantic search returning the top-n most similar corpus chunks.

    Args:
        df: lincoln_index_df (read-only; this function never mutates it).
        user_query: query string (will be embedded via the OpenAI API).
        client: OpenAI client instance.
        n: number of top results to return.
        model: embedding model name passed to get_embedding().
        corpus_matrix: optional pre-normalized matrix from build_corpus_matrix().
            If supplied, uses the fast vectorized path. If None, falls back to
            the original .apply() path (slower but compatible with old call sites).
        row_index: list of DataFrame index labels from build_corpus_matrix(),
            required when corpus_matrix is supplied.

    Returns:
        top_n: DataFrame (copy) of the n most similar rows, with added columns:
            'similarities' (float), 'UserQuery' (str).
        user_query_embedding: raw (unnormalized) query embedding as np.ndarray.
    """
    user_query_embedding = get_embedding(user_query, client, model=model)

    if corpus_matrix is not None and row_index is not None:
        # --- Vectorized path ---
        similarities = _vectorized_similarities(user_query_embedding, corpus_matrix)
        # Align scores back to the DataFrame without mutating it
        top_idx = np.argsort(similarities)[::-1][:n]
        top_df_idx = [row_index[i] for i in top_idx]
        top_n = df.loc[top_df_idx].copy()
        top_n["similarities"] = similarities[top_idx]
    else:
        # --- Fallback path (original behavior, for backward compatibility) ---
        # Computes similarities row-by-row; does NOT mutate df.
        sims = np.array([
            cosine_similarity(emb, user_query_embedding)
            for emb in df["embedding"].values
        ])
        top_idx = np.argsort(sims)[::-1][:n]
        top_n = df.iloc[top_idx].copy()
        top_n["similarities"] = sims[top_idx]

    top_n["UserQuery"] = user_query
    return top_n, user_query_embedding


def semantic_search(user_query, df, client, top_n=5, model="text-embedding-ada-002",
                    corpus_matrix=None, row_index=None):
    """Semantic search (alternate signature retained for backward compatibility).

    Identical to search_text() but with the original argument order
    (user_query, df, client) used by older call sites.

    Returns:
        top_n_results: DataFrame (copy) of the top_n most similar rows,
            with added 'similarities' column.
        user_query_embedding: raw query embedding as np.ndarray.
    """
    user_query_embedding = get_embedding(user_query, client, model=model)

    if corpus_matrix is not None and row_index is not None:
        similarities = _vectorized_similarities(user_query_embedding, corpus_matrix)
        top_idx = np.argsort(similarities)[::-1][:top_n]
        top_df_idx = [row_index[i] for i in top_idx]
        top_n_results = df.loc[top_df_idx].copy()
        top_n_results["similarities"] = similarities[top_idx]
    else:
        sims = np.array([
            cosine_similarity(emb, user_query_embedding)
            for emb in df["embedding"].values
        ])
        top_idx = np.argsort(sims)[::-1][:top_n]
        top_n_results = df.iloc[top_idx].copy()
        top_n_results["similarities"] = sims[top_idx]

    return top_n_results, user_query_embedding
