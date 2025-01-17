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
    def __init__(self, openai_api_key, cohere_api_key, gcp_service_account, hays_data_logger, keyword_results_logger):
        # Initialize OpenAI and Cohere clients
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.cohere_client = cohere.Client(api_key=cohere_api_key)

        # Initialize Google Sheets client
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        credentials = service_account.Credentials.from_service_account_info(gcp_service_account, scopes=scope)
        self.gc = pygsheets.authorize(custom_credentials=credentials)

        # Store the loggers
        self.hays_data_logger = hays_data_logger
        self.keyword_results_logger = keyword_results_logger

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

    def search_with_dynamic_weights_expanded(self, user_keywords, json_data, year_keywords=None, text_keywords=None, top_n_results=5, lincoln_data=None):
        total_words = sum(term['rawFreq'] for term in json_data['corpusTerms']['terms'])
        relative_frequencies = {term['term'].lower(): term['rawFreq'] / total_words for term in json_data['corpusTerms']['terms']}
        inverse_weights = {keyword: 1 / relative_frequencies.get(keyword.lower(), 1) for keyword in user_keywords}
        max_weight = max(inverse_weights.values())
        normalized_weights = {keyword: (weight / max_weight) * 10 for keyword, weight in inverse_weights.items()}

        return self.find_instances_expanded_search(
            dynamic_weights=normalized_weights,
            original_weights=user_keywords,
            data=lincoln_data,
            year_keywords=year_keywords,
            text_keywords=text_keywords,
            top_n=top_n_results
        )

    def find_instances_expanded_search(self, dynamic_weights, original_weights, data, year_keywords=None, text_keywords=None, top_n=5):
        instances = []
        text_keywords_list = [keyword.strip().lower() for keyword in text_keywords] if text_keywords else []

        for entry in data:
            if 'full_text' in entry and 'source' in entry:
                entry_text_lower = entry['full_text'].lower()
                source_lower = entry['source'].lower()
                summary_lower = entry.get('summary', '').lower()
                combined_text = entry_text_lower + ' ' + summary_lower + ' ' + ' '.join(entry.get('keywords', [])).lower()

                match_source_year = not year_keywords or any(str(year) in source_lower for year in year_keywords)
                match_source_text = not text_keywords or any(re.search(r'\b' + re.escape(keyword.lower()) + r'\b', source_lower) for keyword in text_keywords_list)

                if match_source_year and match_source_text:
                    total_dynamic_weighted_score = 0
                    keyword_positions = {}
                    for keyword in original_weights.keys():
                        keyword_lower = keyword.lower()
                        for match in re.finditer(r'\b' + re.escape(keyword_lower) + r'\b', combined_text):
                            count = len(re.findall(r'\b' + re.escape(keyword_lower) + r'\b', combined_text))
                            total_dynamic_weighted_score += count * dynamic_weights.get(keyword, 0)
                            keyword_positions[match.start()] = (keyword, original_weights[keyword])

                    if keyword_positions:
                        highest_original_weighted_position = max(keyword_positions.items(), key=lambda x: x[1][1])[0]
                        context_length = 300
                        start_quote = max(0, highest_original_weighted_position - context_length)
                        end_quote = min(len(entry_text_lower), highest_original_weighted_position + context_length)
                        snippet = entry['full_text'][start_quote:end_quote]
                        instances.append({
                            "text_id": entry['text_id'],
                            "source": entry['source'],
                            "summary": entry.get('summary', ''),
                            "key_quote": snippet,
                            "weighted_score": total_dynamic_weighted_score
                        })
        instances.sort(key=lambda x: x['weighted_score'], reverse=True)
        return instances[:top_n]

    def rerank_results(self, user_query, combined_data):
        try:
            combined_data_strs = [
                f"{cd['search_type']}|Text ID: {cd['text_id']}|Summary: {cd['summary']}|Key Quote: {cd['key_quote']}"
                for cd in combined_data if isinstance(cd, dict)
            ]
            reranked_response = self.cohere_client.rerank(
                model='rerank-english-v2.0',
                query=user_query,
                documents=combined_data_strs,
                top_n=10
            )
            full_reranked_results = []
            for idx, result in enumerate(reranked_response.results):
                data_parts = result.document.split("|")
                if len(data_parts) >= 4:
                    full_reranked_results.append({
                        'Rank': idx + 1,
                        'Search Type': data_parts[0].strip(),
                        'Text ID': data_parts[1].strip().replace("Text ID:", "").strip(),
                        'Summary': data_parts[2].strip().replace("Summary:", "").strip(),
                        'Key Quote': data_parts[3].strip().replace("Key Quote:", "").strip(),
                        'Relevance Score': result.relevance_score
                    })
            return full_reranked_results
        except Exception as e:
            raise Exception("Error in reranking: " + str(e))

    def run_rag_process(self, user_query):
        try:
            lincoln_data = self.lincoln_data.to_dict('records')
            response = self.openai_client.chat.completions.create(
                model="ft:gpt-3.5-turbo",
                messages=[{"role": "system", "content": st.session_state['keyword_model_system_prompt']},
                          {"role": "user", "content": user_query}],
                temperature=0,
                max_tokens=500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            initial_answer = json.loads(response.choices[0].message.content)['initial_answer']
            return {"initial_answer": initial_answer}
        except Exception as e:
            raise Exception("Error in RAG process: " + str(e))
