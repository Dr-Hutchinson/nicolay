# rag_module.py

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

class RAGProcess:
    def __init__(self, openai_api_key, cohere_api_key, gcp_service_account):
        # Initialize OpenAI and Cohere clients
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.cohere_client = cohere.Client(api_key=cohere_api_key)

        # Initialize Google Sheets client
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        credentials = service_account.Credentials.from_service_account_info(gcp_service_account, scopes=scope)
        self.gc = pygsheets.authorize(custom_credentials=credentials)

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
                        instances.append({
                            "text_id": entry['text_id'],  # Ensure 'text_id' is included
                            "source": entry['source'],
                            "summary": entry.get('summary', ''),
                            "quote": snippet.replace('\n', ' '),
                            "weighted_score": total_dynamic_weighted_score,
                            "keyword_counts": keyword_counts
                        })
        instances.sort(key=lambda x: x['weighted_score'], reverse=True)
        return instances[:top_n]


    def search_text(self, df, user_query, n=5):
        user_query_embedding = self.get_embedding(user_query)
        df["similarities"] = df['embedding'].apply(lambda x: self.cosine_similarity(x, user_query_embedding))
        top_n = df.sort_values("similarities", ascending=False).head(n)
        top_n["UserQuery"] = user_query  # Add 'UserQuery' column to the DataFrame
        return top_n, user_query_embedding

    def compare_segments_with_query_parallel(self, segments, query_embedding):
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.get_embedding, segment) for segment in segments]
            segment_embeddings = [future.result() for future in futures]
            return [(segments[i], self.cosine_similarity(segment_embeddings[i], query_embedding)) for i in range(len(segments))]

    def remove_duplicates(self, search_results, semantic_matches):
        combined_results = pd.concat([search_results, semantic_matches])
        deduplicated_results = combined_results.drop_duplicates(subset='text_id')
        return deduplicated_results


    def rerank_results(self, user_query, combined_data):
        try:
            reranked_response = self.cohere_client.rerank(
                model='rerank-english-v2.0',
                query=user_query,
                documents=combined_data,
                top_n=10
            )
            full_reranked_results = []
            for idx, result in enumerate(reranked_response.results):  # Access the results attribute of the response
                combined_data = result.document
                data_parts = combined_data.split("|")
                if len(data_parts) >= 4:
                    search_type, text_id_part, summary, quote = data_parts
                    text_id = str(text_id_part.split(":")[-1].strip())
                    summary = summary.strip()
                    quote = quote.strip()
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


    def get_final_model_response(self, user_query, initial_answer, formatted_input_for_model):
        messages_for_second_model = [
            {"role": "system", "content": response_prompt},
            {"role": "user", "content": f"User Query: {user_query}\n\nInitial Answer: {initial_answer}\n\n{formatted_input_for_model}"}
        ]
        response = self.openai_client.chat.completions.create(
            model="ft:gpt-3.5-turbo-1106:personal::8clf6yi4",  # Replace with your fine-tuned model
            messages=messages_for_second_model,
            temperature=0,
            max_tokens=2000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content

    def run_rag_process(self, user_query):
        lincoln_data = self.load_json('data/lincoln_speech_corpus.json')
        keyword_data = self.load_json('data/voyant_word_counts.json')
        df = pd.read_csv("lincoln_index_embedded.csv")

        lincoln_dict = {item['text_id']: item for item in lincoln_data}

        df['full_text'] = df['combined'].apply(extract_full_text)
        df['embedding'] = df['full_text'].apply(lambda x: self.get_embedding(x) if x else np.zeros(1536))
        df['source'], df['summary'] = zip(*df['Unnamed: 0'].apply(lambda text_id: get_source_and_summary(text_id, lincoln_dict)))

        response = self.openai_client.chat.completions.create(
            model="ft:gpt-3.5-turbo-1106:personal::8XtdXKGK",
            messages=[
                {"role": "system", "content": keyword_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        api_response_data = json.loads(response.choices[0].message.content)
        initial_answer = api_response_data['initial_answer']
        model_weighted_keywords = api_response_data['weighted_keywords']
        model_year_keywords = api_response_data['year_keywords']
        model_text_keywords = api_response_data['text_keywords']

        search_results = self.search_with_dynamic_weights_expanded(
            user_keywords=model_weighted_keywords,
            json_data=keyword_data,
            year_keywords=model_year_keywords,
            text_keywords=model_text_keywords,
            top_n_results=5,
            lincoln_data=lincoln_data
        )

        search_results_df = pd.DataFrame(search_results)

        # Debug: Print the columns of search_results_df
        st.write("search_results_df columns:", search_results_df.columns)

        semantic_matches, user_query_embedding = self.search_text(df, user_query + initial_answer, n=5)
        top_segments = []
        for idx, row in semantic_matches.iterrows():
            segments = segment_text(row['full_text'])
            segment_scores = self.compare_segments_with_query_parallel(segments, user_query_embedding)
            top_segment = max(segment_scores, key=lambda x: x[1])
            top_segments.append(top_segment[0])
        semantic_matches["TopSegment"] = top_segments

        # Debug: Print the columns of semantic_matches
        st.write("semantic_matches columns:", semantic_matches.columns)

        deduplicated_results = self.remove_duplicates(search_results_df, semantic_matches)

        all_combined_data = [
            f"Keyword|Text ID: {result['text_id']}|{result['summary']}|{result['quote']}" for _, result in deduplicated_results.iterrows()
        ] + [
            f"Semantic|Text ID: {result['text_id']}|{result['summary']}|{result['TopSegment']}" for _, result in semantic_matches.iterrows()
        ]

        reranked_results = self.rerank_results(user_query, all_combined_data)
        formatted_input_for_model = format_reranked_results_for_model_input(reranked_results)
        final_model_response = self.get_final_model_response(user_query, initial_answer, formatted_input_for_model)

        return final_model_response



# Helper Functions

def extract_full_text(record):
    marker = "Full Text:\n"
    if isinstance(record, str):
        marker_index = record.find(marker)
        if marker_index != -1:
            return record[marker_index + len(marker):].strip()
        else:
            return ""
    else:
        return ""

def get_source_and_summary(text_id, lincoln_dict):
    text_id_str = f"Text #: {text_id}"
    return lincoln_dict.get(text_id_str, {}).get('source'), lincoln_dict.get(text_id_str, {}).get('summary')

def format_reranked_results_for_model_input(reranked_results):
    formatted_results = []
    top_three_results = reranked_results[:3]
    for result in top_three_results:
        formatted_entry = f"Match {result['Rank']}: Search Type - {result['Search Type']}, Text ID - {result['Text ID']}, Source - {result['Source']}, Summary - {result['Summary']}, Key Quote - {result['Key Quote']}, Relevance Score - {result['Relevance Score']:.2f}"
        formatted_results.append(formatted_entry)
    return "\n\n".join(formatted_results)


def segment_text(text, segment_size=100):
    words = text.split()
    return [' '.join(words[i:i+segment_size]) for i in range(0, len(words), segment_size)]

# System prompt
def load_prompt(file_name):
    """Load prompt from a file."""
    with open(file_name, 'r') as file:
        return file.read()

# Function to ensure prompts are loaded into session state
def load_prompts():
    if 'keyword_model_system_prompt' not in st.session_state:
        st.session_state['keyword_model_system_prompt'] = load_prompt('prompts/keyword_model_system_prompt.txt')
    if 'response_model_system_prompt' not in st.session_state:
        st.session_state['response_model_system_prompt'] = load_prompt('prompts/response_model_system_prompt.txt')
    if 'app_into' not in st.session_state:
        st.session_state['app_intro'] = load_prompt('prompts/app_intro.txt')
    if 'keyword_search_explainer' not in st.session_state:
        st.session_state['keyword_search_explainer'] = load_prompt('prompts/keyword_search_explainer.txt')
    if 'semantic_search_explainer' not in st.session_state:
        st.session_state['semantic_search_explainer'] = load_prompt('prompts/semantic_search_explainer.txt')
    if 'relevance_ranking_explainer' not in st.session_state:
        st.session_state['relevance_ranking_explainer'] = load_prompt('prompts/relevance_ranking_explainer.txt')
    if 'nicolay_model_explainer' not in st.session_state:
        st.session_state['nicolay_model_explainer'] = load_prompt('prompts/nicolay_model_explainer.txt')

# Ensure prompts are loaded
load_prompts()

# Now you can use the prompts from session state
keyword_prompt = st.session_state['keyword_model_system_prompt']
response_prompt = st.session_state['response_model_system_prompt']
app_intro = st.session_state['app_intro']
keyword_search_explainer = st.session_state['keyword_search_explainer']
semantic_search_explainer = st.session_state['semantic_search_explainer']
relevance_ranking_explainer = st.session_state['relevance_ranking_explainer']
nicolay_model_explainer = st.session_state['nicolay_model_explainer']