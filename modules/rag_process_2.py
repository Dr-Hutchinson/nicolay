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
from modules.data_utils import load_lincoln_speech_corpus, load_voyant_word_counts, load_lincoln_index_embedded
from modules.data_logging import DataLogger
import time

class RAGProcess:
    def __init__(self, openai_api_key, cohere_api_key, gcp_service_account, hays_data_logger):
        # Initialize OpenAI and Cohere clients
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.cohere_client = cohere.Client(api_key=cohere_api_key)

        # Initialize Google Sheets client
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        credentials = service_account.Credentials.from_service_account_info(gcp_service_account, scopes=scope)
        self.gc = pygsheets.authorize(custom_credentials=credentials)

        # Store the hays_data_logger
        self.hays_data_logger = hays_data_logger

        # Load data using cached functions
        self.lincoln_data = load_lincoln_speech_corpus()
        self.voyant_data = load_voyant_word_counts()
        self.lincoln_index_df = load_lincoln_index_embedded()

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        response = self.openai_client.embeddings.create(input=[text], model=model)
        return np.array(response.data[0].embedding)

    def cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def remove_duplicates(self, combined_results):
        """
        Deduplicate combined search results.
        Args:
            combined_results (DataFrame): Combined results DataFrame.
        Returns:
            DataFrame: Deduplicated results with the highest-priority entries retained.
        """
        # Sort by priority (e.g., weighted_score > similarity) before deduplication
        if 'weighted_score' in combined_results.columns:
            combined_results = combined_results.sort_values(by='weighted_score', ascending=False)
        elif 'similarities' in combined_results.columns:
            combined_results = combined_results.sort_values(by='similarities', ascending=False)

        # Deduplicate based on `text_id`, keeping the highest-priority row
        deduplicated = combined_results.drop_duplicates(subset='text_id', keep='first').reset_index(drop=True)

        # Fill missing columns like `key_quote` and `summary` with defaults
        deduplicated['key_quote'] = deduplicated.get('key_quote', '').fillna('')
        deduplicated['summary'] = deduplicated.get('summary', '').fillna('')
        return deduplicated

    def rerank_results(self, user_query, combined_data):
        """
        Rerank results using a model.
        Args:
            user_query (str): User query.
            combined_data (DataFrame): Combined and deduplicated results.
        Returns:
            DataFrame: Reranked results.
        """
        # Create standardized input strings for reranking
        combined_data['rerank_input'] = combined_data.apply(
            lambda row: f"{row.get('search_type', 'Unknown')}|Text ID: {row['text_id']}|Summary: {row['summary']}|Key Quote: {row['key_quote']}",
            axis=1
        )

        # Ensure unique entries for reranking
        unique_inputs = combined_data['rerank_input'].drop_duplicates().tolist()
        st.write(f"Combined data size before reranking: {len(unique_inputs)}")

        # Perform reranking
        reranked_response = self.cohere_client.rerank(
            model='rerank-english-v2.0',
            query=user_query,
            documents=unique_inputs,
            top_n=10
        )

        # Map reranked results back to the DataFrame
        reranked_results = pd.DataFrame([
            {
                "Rank": idx + 1,
                "Text ID": result.document.split("|Text ID:")[1].split("|Summary:")[0].strip(),
                "Summary": result.document.split("|Summary:")[1].split("|Key Quote:")[0].strip(),
                "Key Quote": result.document.split("|Key Quote:")[1].strip(),
                "Relevance Score": result.relevance_score
            }
            for idx, result in enumerate(reranked_response.results)
        ])
        return reranked_results

    def run_rag_process(self, user_query):
        try:
            start_time = time.time()

            # Step 1: Perform keyword and semantic searches
            keyword_results = self.search_with_dynamic_weights_expanded(...)
            semantic_matches, query_embedding = self.search_text(...)

            # Step 2: Combine and deduplicate results
            combined_results = pd.concat([keyword_results, semantic_matches], ignore_index=True)
            combined_results = self.remove_duplicates(combined_results)

            # Step 3: Rerank combined results
            reranked_results = self.rerank_results(user_query, combined_results)

            # Step 4: Display reranked results
            st.write(f"Final Reranked Results: {reranked_results}")
            return reranked_results

        except Exception as e:
            st.error(f"Error in RAG process: {e}")
            raise

# Helper Functions
def segment_text(text, segment_size=100):
    words = text.split()
    return [' '.join(words[i:i+segment_size]) for i in range(0, len(words), segment_size)]

def load_prompt(file_name):
    """Load prompt from a file."""
    with open(file_name, 'r') as file:
        return file.read()

def load_prompts():
    if 'keyword_model_system_prompt' not in st.session_state:
        st.session_state['keyword_model_system_prompt'] = load_prompt('prompts/keyword_model_system_prompt.txt')
    if 'response_model_system_prompt' not in st.session_state:
        st.session_state['response_model_system_prompt'] = load_prompt('prompts/response_model_system_prompt.txt')

# Ensure prompts are loaded
load_prompts()
keyword_prompt = st.session_state['keyword_model_system_prompt']
response_prompt = st.session_state['response_model_system_prompt']
