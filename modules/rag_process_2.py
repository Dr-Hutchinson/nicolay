import json
import numpy as np
import pandas as pd
from datetime import datetime as dt
from google.oauth2 import service_account
from openai import OpenAI
import cohere
import re
from concurrent.futures import ThreadPoolExecutor
import pygsheets
import streamlit as st
from modules.data_utils import (
    load_lincoln_speech_corpus,
    load_voyant_word_counts,
    load_lincoln_index_embedded
)
from modules.data_logging import (
    DataLogger,
    log_keyword_search_results,
    log_semantic_search_results,
    log_reranking_results,
    log_nicolay_model_output
)
import time

class RAGProcess:
    def __init__(self, openai_api_key, cohere_api_key, gcp_service_account, hays_data_logger, keyword_results_logger):
        # Initialize OpenAI and Cohere clients
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.cohere_client = cohere.Client(api_key=cohere_api_key)

        # Initialize Google Sheets client
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        credentials = service_account.Credentials.from_service_account_info(gcp_service_account, scopes=scope)
        self.gc = pygsheets.authorize(custom_credentials=credentials)

        # Store loggers
        self.hays_data_logger = hays_data_logger
        self.keyword_results_logger = keyword_results_logger

        # Load data using cached functions
        self.lincoln_data = load_lincoln_speech_corpus()
        self.voyant_data = load_voyant_word_counts()
        self.lincoln_index_df = load_lincoln_index_embedded()

    def log_and_debug(self, logger_func, data, label):
        logger_func(data)
        st.write(f"{label}: {data}")

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        response = self.openai_client.embeddings.create(input=[text], model=model)
        return np.array(response.data[0].embedding)

    def cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def search_with_dynamic_weights_expanded(self, user_keywords, json_data, year_keywords=None, text_keywords=None, top_n_results=5, lincoln_data=None):
        # Calculate weights
        total_words = sum(term['rawFreq'] for term in json_data['corpusTerms']['terms'])
        relative_frequencies = {term['term'].lower(): term['rawFreq'] / total_words for term in json_data['corpusTerms']['terms']}
        inverse_weights = {keyword: 1 / relative_frequencies.get(keyword.lower(), 1) for keyword in user_keywords}
        max_weight = max(inverse_weights.values())
        normalized_weights = {keyword: (weight / max_weight) * 10 for keyword, weight in inverse_weights.items()}

        results = self.find_instances_expanded_search(
            dynamic_weights=normalized_weights,
            original_weights=user_keywords,
            data=lincoln_data,
            year_keywords=year_keywords,
            text_keywords=text_keywords,
            top_n=top_n_results
        )
        self.log_and_debug(log_keyword_search_results, results, "Keyword Search Results")
        return results

    def find_instances_expanded_search(self, dynamic_weights, original_weights, data, year_keywords=None, text_keywords=None, top_n=5):
        instances = []
        for entry in data:
            if 'full_text' in entry and 'source' in entry:
                combined_text = f"{entry['full_text'].lower()} {entry.get('summary', '').lower()} {entry.get('keywords', '').lower()}"
                total_score = sum(dynamic_weights.get(kw.lower(), 0) * combined_text.count(kw.lower()) for kw in original_weights)
                if total_score > 0:
                    instances.append({
                        "text_id": entry['text_id'],
                        "source": entry['source'],
                        "summary": entry.get('summary', ''),
                        "key_quote": combined_text[:300],
                        "weighted_score": total_score
                    })
        instances.sort(key=lambda x: x['weighted_score'], reverse=True)
        return instances[:top_n]

    def remove_duplicates(self, combined_results):
        deduplicated = combined_results.drop_duplicates(subset='text_id', keep='first').reset_index(drop=True)
        st.write(f"Deduplicated Results: {deduplicated}")
        return deduplicated

    def rerank_results(self, user_query, combined_data):
        combined_data_strs = list(set(
            f"{row['search_type']}|Text ID: {row['text_id']}|Summary: {row['summary']}|Key Quote: {row['key_quote']}"
            for row in combined_data
        ))
        reranked_response = self.cohere_client.rerank(
            model='rerank-english-v2.0',
            query=user_query,
            documents=combined_data_strs,
            top_n=10
        )
        reranked_results = [
            {
                "Rank": idx + 1,
                "Search Type": doc.split("|")[0],
                "Text ID": doc.split("|")[1].replace("Text ID: ", ""),
                "Summary": doc.split("|")[2].replace("Summary: ", ""),
                "Key Quote": doc.split("|")[3].replace("Key Quote: ", ""),
                "Relevance Score": result.relevance_score
            }
            for idx, (result, doc) in enumerate(zip(reranked_response.results, combined_data_strs))
        ]
        self.log_and_debug(log_reranking_results, reranked_results, "Reranked Results")
        return reranked_results

    def run_rag_process(self, user_query):
        try:
            lincoln_data = self.lincoln_data.to_dict('records')
            df = self.lincoln_index_df
            df['embedding'] = df['embedding'].apply(lambda x: list(map(float, x.strip("[]").split(","))))

            keyword_results = self.search_with_dynamic_weights_expanded(
                user_keywords=["freedom", "constitution"],
                json_data=self.voyant_data.iloc[0],
                lincoln_data=lincoln_data
            )
            semantic_results, query_embedding = self.search_text(df, user_query, n=5)

            combined_results = pd.concat([pd.DataFrame(keyword_results), semantic_results])
            deduplicated_results = self.remove_duplicates(combined_results)
            reranked_results = self.rerank_results(user_query, deduplicated_results.to_dict('records'))

            return {
                "keyword_results": keyword_results,
                "semantic_results": semantic_results,
                "reranked_results": reranked_results
            }
        except Exception as e:
            st.write(f"Error in run_rag_process: {e}")
            raise
