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
from modules.data_logging import DataLogger, log_keyword_search_results, log_semantic_search_results, log_reranking_results, log_nicolay_model_output
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

        # Store the hays_data_logger
        self.hays_data_logger = hays_data_logger
        self.keyword_results_logger = keyword_results_logger

        # Load data using cached functions
        self.lincoln_data = load_lincoln_speech_corpus()
        self.voyant_data = load_voyant_word_counts()
        self.lincoln_index_df = load_lincoln_index_embedded()

    def load_json(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

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
        # Calculate the total number of words for normalization
        total_words = sum(term['rawFreq'] for term in json_data['corpusTerms']['terms'])
        relative_frequencies = {term['term'].lower(): term['rawFreq'] / total_words for term in json_data['corpusTerms']['terms']}

        # Calculate inverse weights based on the relative frequencies
        inverse_weights = {keyword: 1 / relative_frequencies.get(keyword.lower(), 1) for keyword in user_keywords}

        # Normalize weights for dynamic weighting
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

        if text_keywords:
            if isinstance(text_keywords, list):
                text_keywords_list = [keyword.strip().lower() for keyword in text_keywords]
            else:
                text_keywords_list = [keyword.strip().lower() for keyword in text_keywords.split(',')]
        else:
            text_keywords_list = []

        for entry in data:
            if 'full_text' in entry and 'source' in entry:
                entry_text_lower = entry['full_text'].lower()
                source_lower = entry['source'].lower()
                summary_lower = entry.get('summary', '').lower()
                keywords_lower = ' '.join(entry.get('keywords', [])).lower()

                match_source_year = not year_keywords or any(str(year) in source_lower for year in year_keywords)
                match_source_text = not text_keywords or any(re.search(r'\b' + re.escape(keyword.lower()) + r'\b', source_lower) for keyword in text_keywords_list)

                if match_source_year and match_source_text:
                    total_dynamic_weighted_score = 0
                    keyword_counts = {}
                    keyword_positions = {}
                    combined_text = entry_text_lower + ' ' + summary_lower + ' ' + keywords_lower

                    for keyword in original_weights.keys():
                        keyword_lower = keyword.lower()
                        for match in re.finditer(r'\b' + re.escape(keyword_lower) + r'\b', combined_text):
                            count = len(re.findall(r'\b' + re.escape(keyword_lower) + r'\b', combined_text))
                            dynamic_weight = dynamic_weights.get(keyword, 0)
                            if count > 0:
                                keyword_counts[keyword] = count
                                total_dynamic_weighted_score += count * dynamic_weight
                                keyword_index = match.start()
                                original_weight = original_weights[keyword]
                                keyword_positions[keyword_index] = (keyword, original_weight)

                    if keyword_positions:
                        highest_original_weighted_position = max(keyword_positions.items(), key=lambda x: x[1][1])[0]
                        context_length = 300
                        start_quote = max(0, highest_original_weighted_position - context_length)
                        end_quote = min(len(entry_text_lower), highest_original_weighted_position + context_length)
                        snippet = entry['full_text'][start_quote:end_quote]
                        formatted_snippet = snippet.replace("\n", " ") if snippet else 'N/A'
                        instances.append({
                            "text_id": entry['text_id'],
                            "source": entry['source'],
                            "summary": entry.get('summary', ''),
                            "key_quote": formatted_snippet,
                            "weighted_score": total_dynamic_weighted_score,
                            "keyword_counts": keyword_counts
                        })
            else:
                st.write(f"Skipping entry without full_text or source: {entry}")

        instances.sort(key=lambda x: x['weighted_score'], reverse=True)
        return instances[:top_n]

    def search_text(self, df, user_query, n=5):
        user_query_embedding = self.get_embedding(user_query)
        df["similarities"] = df['embedding'].apply(lambda x: self.cosine_similarity(x, user_query_embedding))
        top_n = df.sort_values("similarities", ascending=False).head(n)
        top_n["UserQuery"] = user_query
        return top_n, user_query_embedding

    def remove_duplicates(self, search_results, semantic_matches):
        combined_results = pd.concat([search_results, semantic_matches], ignore_index=True)

        if 'text_id' not in combined_results.columns:
            raise ValueError("The 'text_id' column is missing from combined results.")

        if 'weighted_score' in combined_results.columns:
            combined_results = combined_results.sort_values(by='weighted_score', ascending=False)
        elif 'similarities' in combined_results.columns:
            combined_results = combined_results.sort_values(by='similarities', ascending=False)

        deduplicated_results = combined_results.drop_duplicates(subset='text_id', keep='first').reset_index(drop=True)

        deduplicated_results['key_quote'] = deduplicated_results['key_quote'].fillna('')
        deduplicated_results['summary'] = deduplicated_results['summary'].fillna('')

        return deduplicated_results

    def rerank_results(self, user_query, combined_data):
        try:
            combined_data_strs = [
                f"{cd['search_type']}|Text ID: {cd['text_id']}|Summary: {cd['summary']}|Key Quote: {cd['key_quote']}"
                if isinstance(cd, dict) else cd
                for cd in combined_data
            ]

            unique_combined_data_strs = list(set(combined_data_strs))

            reranked_response = self.cohere_client.rerank(
                model='rerank-english-v2.0',
                query=user_query,
                documents=unique_combined_data_strs,
                top_n=10
            )

            full_reranked_results = []
            for idx, result in enumerate(reranked_response.results):
                combined_data_text = result.document if isinstance(result.document, str) else result.document['text']

                data_parts = combined_data_text.split("|")
                if len(data_parts) >= 4:
                    search_type = data_parts[0].strip()
                    text_id_part = data_parts[1].strip()
                    summary = data_parts[2].strip()
                    quote = data_parts[3].strip()

                    text_id = text_id_part.replace("Text ID:", "").strip()
                    summary = summary.replace("Summary:", "").strip()
                    quote = quote.replace("Key Quote:", "").strip()

                    source = self.lincoln_dict.get(f"Text #: {text_id}", {}).get('source', 'Source information not available')

                    full_reranked_results.append({
                        'Rank': idx + 1,
                        'Search Type': search_type,
                        'Text ID': text_id,
                        'Source': source,
                        'Summary': summary,
                        'Key Quote': quote,
                        'Relevance Score': result.relevance_score
                    })
            return full_reranked_results
        except Exception as e:
            raise Exception("Error in reranking: " + str(e))

    def run_rag_process(self, user_query):
        try:
            start_time = time.time()

            lincoln_data = self.lincoln_data.to_dict('records')
            keyword_data = self.voyant_data
            df = self.lincoln_index_df

            lincoln_dict = {item['text_id']: item for item in lincoln_data}
            self.lincoln_dict = lincoln_dict

            df['embedding'] = df['embedding'].apply(lambda x: list(map(float, x.strip("[]").split(","))))

            df['full_text'] = df['combined'].apply(extract_full_text)

            df['source'], df['summary'] = zip(*df['text_id'].map(lambda text_id: get_source_and_summary(text_id, lincoln_dict)))

            response = self.openai_client.chat.completions.create(
                model="ft:gpt-3.5-turbo-1106:personal::8XtdXKGK",
                messages=[
                    {"role": "system", "content": keyword_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0,
                max_tokens=500
            )

            api_response_data = json.loads(response.choices[0].message.content)

            initial_answer = api_response_data['initial_answer']
            model_weighted_keywords = api_response_data['weighted_keywords']
            model_year_keywords = api_response_data['year_keywords']
            model_text_keywords = api_response_data['text_keywords']

            self.keyword_results_logger.log_keyword_search_results(
                search_results=pd.DataFrame(),  # Placeholder, replace with actual results
                user_query=user_query,
                initial_answer=initial_answer,
                model_weighted_keywords=model_weighted_keywords,
                model_year_keywords=model_year_keywords,
                model_text_keywords=model_text_keywords
            )

            search_results = self.search_with_dynamic_weights_expanded(
                user_keywords=model_weighted_keywords,
                json_data=keyword_data,
                year_keywords=model_year_keywords,
                text_keywords=model_text_keywords,
                top_n_results=5,
                lincoln_data=lincoln_data
            )

            search_results_df = pd.DataFrame(search_results)

            semantic_matches, user_query_embedding = self.search_text(df, user_query + initial_answer, n=5)

            deduplicated_results = self.remove_duplicates(search_results_df, semantic_matches)

            reranked_results = self.rerank_results(user_query, deduplicated_results.to_dict(orient='records'))

            return {
                "initial_answer": initial_answer,
                "search_results": search_results_df,
                "semantic_matches": semantic_matches,
                "reranked_results": pd.DataFrame(reranked_results)
            }
        except Exception as e:
            raise Exception("Error in run_rag_process: " + str(e))

# Helper Functions
def extract_full_text(combined_text):
    markers = ["Full Text:\n", "Full Text: \n", "Full Text:"]
    if isinstance(combined_text, str):
        for marker in markers:
            marker_index = combined_text.find(marker)
            if marker_index != -1:
                return combined_text[marker_index + len(marker):].strip()
        return ""
    return ""

def get_source_and_summary(text_id, lincoln_dict):
    return lincoln_dict.get(text_id, {}).get('source'), lincoln_dict.get(text_id, {}).get('summary')

def load_prompt(file_name):
    with open(file_name, 'r') as file:
        return file.read()

def load_prompts():
    if 'keyword_model_system_prompt' not in st.session_state:
        st.session_state['keyword_model_system_prompt'] = load_prompt('prompts/keyword_model_system_prompt.txt')
    if 'response_model_system_prompt' not in st.session_state:
        st.session_state['response_model_system_prompt'] = load_prompt('prompts/response_model_system_prompt.txt')

load_prompts()
keyword_prompt = st.session_state['keyword_model_system_prompt']
response_prompt = st.session_state['response_model_system_prompt']
