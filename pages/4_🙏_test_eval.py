import streamlit as st
import pandas as pd
import pygsheets
from google.oauth2 import service_account
from datetime import datetime as dt
import json
import time
from openai import OpenAI
import cohere
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import re
from typing import List, Dict, Tuple, Any

# Initialize API keys and Google Sheets authorization
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
client = OpenAI()

os.environ["CO_API_KEY"]= st.secrets["cohere_api_key"]
co = cohere.Client()

scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"], scopes = scope)

gc = pygsheets.authorize(custom_credentials=credentials)

# Data Logger class
class DataLogger:
    def __init__(self, gc, sheet_name):
        self.gc = gc
        self.sheet = self.gc.open(sheet_name).sheet1

    def record_api_outputs(self, data_dict):
        now = dt.now()
        data_dict['Timestamp'] = now  # Add timestamp to the data

        # Convert the data dictionary to a DataFrame
        df = pd.DataFrame([data_dict])

        # Find the next empty row in the sheet to avoid overwriting existing data
        end_row = len(self.sheet.get_all_records()) + 2

        # Append the new data row to the sheet
        self.sheet.set_dataframe(df, (end_row, 1), copy_head=False, extend=True)

# Usage
hays_data_logger = DataLogger(gc, 'hays_data')
keyword_results_logger = DataLogger(gc, 'keyword_search_results')
semantic_results_logger = DataLogger(gc, 'semantic_search_results')
reranking_results_logger = DataLogger(gc, 'reranking_results')
nicolay_data_logger = DataLogger(gc, 'nicolay_data')

def log_keyword_search_results(keyword_results_logger: DataLogger, search_results: pd.DataFrame, user_query: str, initial_answer: str, model_weighted_keywords: Dict[str, float], model_year_keywords: List[str], model_text_keywords: List[str]) -> None:
    now = dt.now()

    for idx, result in search_results.iterrows():
        record = {
            'Timestamp': now,
            'UserQuery': user_query,
            "initial_Answer": initial_answer,
            'Weighted_Keywoords': model_weighted_keywords,
            'Year_Keywords': model_year_keywords,
            'text_keywords': model_text_keywords,
            'TextID': result['text_id'],
            'KeyQuote': result['quote'],
            'WeightedScore': result['weighted_score'],
            'KeywordCounts': json.dumps(result['keyword_counts'])
        }
        keyword_results_logger.record_api_outputs(record)

 def log_semantic_search_results(semantic_results_logger: DataLogger, semantic_matches: pd.DataFrame, initial_answer: str) -> None:
     now = dt.now()

     for idx, row in semantic_matches.iterrows():
         record = {
             'Timestamp': now,
             'UserQuery': row['UserQuery'],
             'HyDE_Query': initial_answer,
             'TextID': row['text_id'], # Corrected key to 'text_id'
             'SimilarityScore': row['similarities'],
             'TopSegment': row['TopSegment']
         }
         semantic_results_logger.record_api_outputs(record)

def log_reranking_results(reranking_results_logger: DataLogger, reranked_df: pd.DataFrame, user_query: str) -> None:
    now = dt.now()

    for idx, row in reranked_df.iterrows():
        record = {
            'Timestamp': now,
            'UserQuery': user_query,
            'Rank': row['Rank'],
            'SearchType': row['Search Type'],
            'TextID': row['Text ID'],
            'KeyQuote': row['Key Quote'],
            'Relevance_Score': row['Relevance Score']
        }
        reranking_results_logger.record_api_outputs(record)

def log_nicolay_model_output(nicolay_data_logger: DataLogger, model_output: Dict[str, Any], user_query: str, initial_answer: str, highlight_success_dict: Dict[str, bool]) -> None:
    final_answer_text = model_output.get("FinalAnswer", {}).get("Text", "No response available")
    references = ", ".join(model_output.get("FinalAnswer", {}).get("References", []))

    query_intent = model_output.get("User Query Analysis", {}).get("Query Intent", "")
    historical_context = model_output.get("User Query Analysis", {}).get("Historical Context", "")

    answer_evaluation = model_output.get("Initial Answer Review", {}).get("Answer Evaluation", "")
    quote_integration = model_output.get("Initial Answer Review", {}).get("Quote Integration Points", "")

    response_effectiveness = model_output.get("Model Feedback", {}).get("Response Effectiveness", "")
    suggestions_for_improvement = model_output.get("Model Feedback", {}).get("Suggestions for Improvement", "")

    match_analysis = model_output.get("Match Analysis", {})
    match_fields = ['Text ID', 'Source', 'Summary', 'Key Quote', 'Historical Context', 'Relevance Assessment']
    match_data = {}
    speech = None  # Initialize 'speech' here

    for match_key, match_details in match_analysis.items():
        match_info = [f"{field}: {match_details.get(field, '')}" for field in match_fields]
        match_data[match_key] = "; ".join(match_info)

        if 'Text ID' in match_details:
            formatted_text_id = f"Text #: {match_details.get('Text ID')}"
            speech = next((item for item in lincoln_data if item['text_id'] == formatted_text_id), None)
            if speech:
              highlight_success_dict[match_key] = highlight_key_quote(speech['full_text'], match_details.get('Key Quote', "")) != speech['full_text']
            else:
                highlight_success_dict[match_key] = False
        else:
            highlight_success_dict[match_key] = False  # Ensure default if Text ID is missing

    meta_strategy = model_output.get("Meta Analysis", {}).get("Strategy for Response Composition", {})
    meta_synthesis = model_output.get("Meta Analysis", {}).get("Synthesis", "")

    record = {
        'Timestamp': dt.now(),
        'UserQuery': user_query,
        'initial_Answer': initial_answer,
        'FinalAnswer': final_answer_text,
        'References': references,
        'QueryIntent': query_intent,
        'HistoricalContext': historical_context,
        'AnswerEvaluation': answer_evaluation,
        'QuoteIntegration': quote_integration,
        **match_data,
        'MetaStrategy': str(meta_strategy),
        'MetaSynthesis': meta_synthesis,
        'ResponseEffectiveness': response_effectiveness,
        'Suggestions': suggestions_for_improvement
    }

    for match_key, success in highlight_success_dict.items():
        record[f'{match_key}_HighlightSuccess'] = success

    nicolay_data_logger.record_api_outputs(record)


# System prompt
def load_prompt(file_name):
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
    if 'llm_eval_prompt' not in st.session_state:
        st.session_state['llm_eval_prompt'] = load_prompt('prompts/llm_eval_prompt.txt')

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
llm_eval_prompt = st.session_state['llm_eval_prompt']

# Helper function to load Google Sheet data
def load_sheet_data(sheet_name: str) -> pd.DataFrame:
    try:
      sheet = gc.open(sheet_name).sheet1
      data = sheet.get_all_records()
      return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading sheet '{sheet_name}': {e}")
        return pd.DataFrame()

# Load data from Google Sheets
questions_df = load_sheet_data('benchmark_questions')
hays_df = load_sheet_data('hays_data')
keyword_results_df = load_sheet_data('keyword_search_results')
semantic_results_df = load_sheet_data('semantic_search_results')
reranking_results_df = load_sheet_data('reranking_results')
nicolay_results_df = load_sheet_data('nicolay_data')

# Define functions
def load_json(file_path: str) -> Dict:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to get the full text from the Lincoln data based on text_id for final display of matching results
def get_full_text_by_id(text_id: str, data: List[Dict]) -> str:
  return next((item['full_text'] for item in data if item['text_id'] == text_id), None)

# Function to highlight truncated quotes for Nicolay model outputs
def highlight_key_quote(text: str, key_quote: str) -> str:
  if not key_quote:
    return text
  parts = key_quote.split("...")
  if len(parts) >= 2:
      pattern = re.escape(parts[0]) + r"\s*.*?\s*" + re.escape(parts[-1]) + r"[.;,]?"
  else:
      pattern = re.escape(key_quote) + r"\s*[.;,]?"
  regex = re.compile(pattern, re.IGNORECASE)
  matches = regex.finditer(text)  # Use finditer to get all matches
  for match in matches:
    text = text[:match.start()] + f"<mark>{match.group(0)}</mark>" + text[match.end():]
  return text


# text splitting - 0.1
def segment_text(text: str, segment_size: int = 500, overlap: int = 100) -> List[str]:
    words = text.split()
    segments = []
    for i in range(0, len(words), segment_size - overlap):
        segment = words[i:i + segment_size]
        segments.append(' '.join(segment))
    return segments

def compare_segments_with_query_parallel(segments: List[str], query_embedding: np.ndarray) -> List[Tuple[str, float]]:
  with ThreadPoolExecutor(max_workers=5) as executor:
      futures = [executor.submit(get_embedding, segment) for segment in segments]
      segment_embeddings = [future.result() for future in futures]
      return [(segments[i], cosine_similarity(segment_embeddings[i], query_embedding)) for i in range(len(segments))]

# function for loading JSON 'text_id' for comparsion for semantic search results
def get_source_and_summary(text_id: str) -> Tuple[str, str]:
  text_id_str = f"Text #: {text_id}"
  return lincoln_dict.get(text_id_str, {}).get('source'), lincoln_dict.get(text_id_str, {}).get('summary')

# keyword text segment - 0.1
def find_instances_expanded_search(dynamic_weights: Dict[str, float], original_weights: Dict[str, float], data: List[Dict], year_keywords: List[str] = None, text_keywords: List[str] = None, top_n: int = 5, context_size: int = 1000) -> List[Dict]:
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
                    start_quote = max(0, highest_original_weighted_position - context_size // 3)
                    end_quote = min(len(entry_text_lower), highest_original_weighted_position + context_size // 3)
                    snippet = entry['full_text'][start_quote:end_quote]
                    instances.append({
                        "text_id": entry['text_id'],
                        "source": entry['source'],
                        "summary": entry.get('summary', ''),
                        "quote": snippet.replace('\n', ' '),
                        "weighted_score": total_dynamic_weighted_score,
                        "keyword_counts": keyword_counts
                    })

    instances.sort(key=lambda x: x['weighted_score'], reverse=True)
    return instances[:top_n]

# Updated main search function to use expanded search
def search_with_dynamic_weights_expanded(user_keywords: Dict[str, float], json_data: Dict, year_keywords: List[str] = None, text_keywords: List[str] = None, top_n_results: int = 5) -> List[Dict]:
    total_words = sum(term['rawFreq'] for term in json_data['corpusTerms']['terms'])
    relative_frequencies = {term['term'].lower(): term['rawFreq'] / total_words for term in json_data['corpusTerms']['terms']}
    inverse_weights = {keyword: 1 / relative_frequencies.get(keyword.lower(), 1) for keyword in user_keywords}
    max_weight = max(inverse_weights.values())
    normalized_weights = {keyword: (weight / max_weight) * 10 for keyword, weight in inverse_weights.items()}
    return find_instances_expanded_search(
        dynamic_weights=normalized_weights,
        original_weights=user_keywords,
        data=lincoln_data,
        year_keywords=year_keywords,
        text_keywords=text_keywords,
        top_n=top_n_results
    )

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> np.ndarray:
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def search_text(df: pd.DataFrame, user_query: str, n: int = 5) -> Tuple[pd.DataFrame, np.ndarray]:
    user_query_embedding = get_embedding(user_query)
    df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, user_query_embedding))
    top_n = df.sort_values("similarities", ascending=False).head(n)
    top_n["UserQuery"] = user_query
    top_n = top_n.rename(columns={'Unnamed: 0': 'text_id'})  # assign new df to top_n
    return top_n, user_query_embedding

def extract_full_text(record: str) -> str:
    marker = "Full Text:\n"
    if isinstance(record, str):
        marker_index = record.find(marker)
        if marker_index != -1:
            return record[marker_index + len(marker):].strip()
        else:
            return ""
    else:
        return ""

def remove_duplicates(search_results: pd.DataFrame, semantic_matches: pd.DataFrame) -> pd.DataFrame:
  combined_results = pd.concat([search_results, semantic_matches])
  deduplicated_results = combined_results.drop_duplicates(subset='text_id')
  return deduplicated_results

def format_reranked_results_for_model_input(reranked_results: List[Dict]) -> str:
  formatted_results = []
  top_three_results = reranked_results[:3]
  for result in top_three_results:
    formatted_entry = f"Match {result['Rank']}: " \
                      f"Search Type - {result['Search Type']}, " \
                      f"Text ID - {result['Text ID']}, " \
                      f"Source - {result['Source']}, " \
                      f"Summary - {result['Summary']}, " \
                      f"Key Quote - {result['Key Quote']}, " \
                      f"Relevance Score - {result['Relevance Score']:.2f}"
    formatted_results.append(formatted_entry)
  return "\n\n".join(formatted_results)

def evaluate_nicolay_output(query: str, initial_answer: str, formatted_matches_for_model: str, final_answer: Dict, system_prompt: str) -> Dict:

      messages = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": f"User Query: {query}\n\nInitial Answer from Hay: {initial_answer}\n\nRelevant Search Results:\n{formatted_matches_for_model}\n\nFinal Analysis from Nicolay: {final_answer}\n\nAssess the factual accuracy, quote integration, and citation accuracy of Nicolay's response."}
      ]

      try:
          response = client.chat.completions.create(
              model="gpt-4",
              messages=messages,
              temperature=0,
              max_tokens=500
          )
          json_response = json.loads(response.choices[0].message.content)
          return json_response
      except Exception as e:
          st.error(f"Error in LLM evaluation: {e}")
          return {"error": str(e), 'factual_accuracy_rating': 'No rating', 'factual_accuracy_rationale': 'No rating', 'quote_integration_rating': 'No rating', 'quote_integration_rationale': 'No rating', 'citation_accuracy_rating': 'No rating', 'citation_accuracy_rationale': 'No rating'}

def track_keyword_success(hays_data: List[Dict], keyword_results: pd.DataFrame, nicolay_data: List[Dict], query: str) -> Tuple[int, int, float, float]:
    hay_entry = next((entry for entry in hays_data if entry['query'] == query), None)
    if not hay_entry:
      return 0, 0, 0, 0

    weighted_keywords = list(hay_entry['weighted_keywords'].keys())
    if not weighted_keywords:
      return 0,0,0,0

    kw_text_ids = set()
    for idx, kw_result in keyword_results.iterrows():
        if kw_result['user_query'] == query:
            kw_text_ids.add(kw_result['text_id'])

    final_text_ids = set()
    for entry in nicolay_data:
        if isinstance(entry, dict) and entry.get('user_query') == query:
            for match_key, match_value in entry.items():
                if match_key.startswith('Match') and isinstance(match_value, dict) and 'Text ID' in match_value:
                    final_text_ids.add(match_value['Text ID'])

    hits = len(kw_text_ids.intersection(final_text_ids))

    if len(kw_text_ids) > 0:
       precision_rate = hits / len(kw_text_ids)
    else:
      precision_rate = 0

    if len(final_text_ids) > 0:
        recall_rate = hits / len(final_text_ids)
    else:
        recall_rate = 0

    return len(weighted_keywords), hits, precision_rate, recall_rate

def track_semantic_success(semantic_results: List[Dict], nicolay_data: List[Dict], query: str) -> Tuple[int, float, float]:
  sem_text_ids = set()
  for sem_result in semantic_results:
    if sem_result['user_query'] == query:
      sem_text_ids.add(str(sem_result['text_id']))

  final_text_ids = set()
  for entry in nicolay_data:
    if isinstance(entry, dict) and entry.get('user_query') == query:
        for match_key, match_value in entry.items():
            if match_key.startswith('Match') and isinstance(match_value, dict) and 'Text ID' in match_value:
                final_text_ids.add(str(match_value['Text ID']))

  hits = len(sem_text_ids.intersection(final_text_ids))
  if len(sem_text_ids) > 0:
      precision_rate = hits / len(sem_text_ids)
  else:
      precision_rate = 0

  if len(final_text_ids) > 0:
      recall_rate = hits / len(final_text_ids)
  else:
      recall_rate = 0

  return hits, precision_rate, recall_rate

def track_rerank_success(rerank_results: List[Dict], query: str, ideal_documents: List[str]) -> Tuple[int, float, float]:
    top_3_ids = []
    for result in rerank_results:
        if result['user_query'] == query and result['result_ranking'] <= 3:
            top_3_ids.append(str(result['text_id']))

    hits = len(set(top_3_ids).intersection(ideal_documents))
    if len(top_3_ids) > 0:
        precision = hits / len(top_3_ids)
    else:
        precision = 0

    average_rank = None
    ranks = []
    for result in rerank_results:
       if result['user_query'] == query and str(result['text_id']) in ideal_documents:
           ranks.append(result['result_ranking'])

    if len(ranks) > 0:
        average_rank = sum(ranks) / len(ranks)
    return hits, precision, average_rank


# Load data
lincoln_speeches_file_path = 'data/lincoln_speech_corpus.json'
keyword_frequency_file_path = 'data/voyant_word_counts.json'
lincoln_speeches_embedded = "lincoln_index_embedded.csv"

lincoln_data = load_json(lincoln_speeches_file_path)
keyword_data = load_json(keyword_frequency_file_path)

# Convert JSON data to a dictionary with 'text_id' as the key for easy access
lincoln_dict = {item['text_id']: item for item in lincoln_data}

def run_rag_process(user_query: str, ideal_documents: List[str], perform_keyword_search: bool = True, perform_semantic_search: bool = True, perform_reranking: bool = True) -> Dict:
  if not perform_keyword_search and not perform_semantic_search:
    return {
      'initial_answer': 'No Search Methods Selected',
      'search_results': pd.DataFrame(),
      'semantic_matches': pd.DataFrame(),
      'full_reranked_results': [],
      'model_output': {},
      'keyword_counts': 0,
      'keyword_hits': 0,
      'keyword_precision': 0,
      'keyword_recall': 0,
      'semantic_hits': 0,
      'semantic_precision': 0,
      'semantic_recall': 0,
      'rerank_hits': 0,
      'rerank_precision': 0,
      'rerank_avg_rank': None,
      'llm_eval': {}
      }

  # Construct the messages for the model
  messages_for_model = [
      {"role": "system", "content": keyword_prompt},
      {"role": "user", "content": user_query}
  ]
  try:
      response = client.chat.completions.create(
          model="ft:gpt-4o-mini-2024-07-18:personal:hays-gpt4o:9tFqrYwI",
          messages=messages_for_model,
          temperature=0,
          max_tokens=500,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
      )
  except Exception as e:
     st.error(f"Error during API call: {e}")
     return {
      'initial_answer': 'Error with API call',
      'search_results': pd.DataFrame(),
      'semantic_matches': pd.DataFrame(),
      'full_reranked_results': [],
      'model_output': {},
      'keyword_counts': 0,
      'keyword_hits': 0,
      'keyword_precision': 0,
      'keyword_recall': 0,
      'semantic_hits': 0,
      'semantic_precision': 0,
      'semantic_recall': 0,
      'rerank_hits': 0,
      'rerank_precision': 0,
      'rerank_avg_rank': None,
      'llm_eval': {}
      }


  msg = response.choices[0].message.content

  try:
    api_response_data = json.loads(msg)
    initial_answer = api_response_data['initial_answer']
    model_weighted_keywords = api_response_data['weighted_keywords']
    model_year_keywords = api_response_data['year_keywords']
    model_text_keywords = api_response_data['text_keywords']
  except (json.JSONDecodeError, KeyError) as e:
        st.error(f"Error parsing or accessing JSON response from Hay: {e}")
        return {
          'initial_answer': 'Error with parsing of Hay model output',
          'search_results': pd.DataFrame(),
          'semantic_matches': pd.DataFrame(),
          'full_reranked_results': [],
          'model_output': {},
          'keyword_counts': 0,
          'keyword_hits': 0,
          'keyword_precision': 0,
          'keyword_recall': 0,
          'semantic_hits': 0,
          'semantic_precision': 0,
          'semantic_recall': 0,
          'rerank_hits': 0,
          'rerank_precision': 0,
          'rerank_avg_rank': None,
           'llm_eval': {}
           }

  hays_data = {
      'query': user_query,
      'initial_answer': initial_answer,
      'weighted_keywords': model_weighted_keywords,
      'year_keywords': model_year_keywords,
      'text_keywords': model_text_keywords,
      'full_output': msg
  }
  hays_data_logger.record_api_outputs(hays_data)

  weighted_keywords = model_weighted_keywords
  year_keywords = model_year_keywords
  text_keywords = model_text_keywords

  search_results = pd.DataFrame()
  if perform_keyword_search:
     search_results = search_with_dynamic_weights_expanded(
          user_keywords=weighted_keywords,
          json_data=keyword_data,
          year_keywords=year_keywords,
          text_keywords=text_keywords,
          top_n_results=5
          )
     if search_results: # the return is a list
       search_results = pd.DataFrame(search_results) # convert to dataframe
       log_keyword_search_results(keyword_results_logger, search_results, user_query, initial_answer, model_weighted_keywords, model_year_keywords, model_text_keywords)


  semantic_matches = pd.DataFrame()
  if perform_semantic_search:
      embedding_size = 1536
      df = pd.read_csv(lincoln_speeches_embedded)
      df['full_text'] = df['combined'].apply(extract_full_text)
      try:
        df['embedding'] = df['full_text'].apply(lambda x: get_embedding(x) if x else np.zeros(embedding_size))
      except Exception as e:
         st.error(f"Error getting embeddings: {e}")
         return {
          'initial_answer': initial_answer,
          'search_results': search_results,
          'semantic_matches': pd.DataFrame(),
          'full_reranked_results': [],
          'model_output': {},
          'keyword_counts': 0,
          'keyword_hits': 0,
          'keyword_precision': 0,
          'keyword_recall': 0,
          'semantic_hits': 0,
          'semantic_precision': 0,
          'semantic_recall': 0,
          'rerank_hits': 0,
          'rerank_precision': 0,
          'rerank_avg_rank': None,
          'llm_eval': {}
           }
      df['source'], df['summary'] = zip(*df['Unnamed: 0'].apply(get_source_and_summary))
      try:
        semantic_matches, user_query_embedding = search_text(df, user_query + initial_answer, n=5)
      except Exception as e:
         st.error(f"Error with semantic search: {e}")
         return {
          'initial_answer': initial_answer,
          'search_results': search_results,
          'semantic_matches': pd.DataFrame(),
          'full_reranked_results': [],
          'model_output': {},
          'keyword_counts': 0,
          'keyword_hits': 0,
          'keyword_precision': 0,
          'keyword_recall': 0,
          'semantic_hits': 0,
          'semantic_precision': 0,
          'semantic_recall': 0,
          'rerank_hits': 0,
          'rerank_precision': 0,
          'rerank_avg_rank': None,
           'llm_eval': {}
           }
      top_segments = []

      for idx, row in semantic_matches.iterrows():
        segments = segment_text(row['full_text'])
        segment_scores = compare_segments_with_query_parallel(segments, user_query_embedding)
        top_segment = max(segment_scores, key=lambda x: x[1])
        top_segments.append(top_segment[0])
      semantic_matches["TopSegment"] = top_segments
      log_semantic_search_results(semantic_results_logger, semantic_matches, initial_answer)


  full_reranked_results = []
  formatted_input_for_model = None  # Initialize to None before the if statement
  if perform_reranking:
    if not search_results.empty or not semantic_matches.empty:

        if isinstance(search_results, list):
            search_results = pd.DataFrame(search_results)

        if not search_results.empty:
            search_results['text_id'] = search_results['text_id'].str.extract('(\d+)').astype(int)
        else:
             search_results = pd.DataFrame(columns=['text_id'])

        semantic_matches.rename(columns={'Unnamed: 0': 'text_id'}, inplace=True)
        semantic_matches['text_id'] = semantic_matches['text_id'].astype(int)


        if search_results.empty:
            deduplicated_results = semantic_matches
        else:
            deduplicated_results = remove_duplicates(search_results, semantic_matches)

        all_combined_data = []

        for index, result in deduplicated_results.iterrows():
            if not search_results.empty and result.text_id in search_results.text_id.values and perform_keyword_search:
                combined_data = f"Keyword|Text ID: {result.text_id}|{result.summary}|{result.quote}"
                all_combined_data.append(combined_data)
            elif not semantic_matches.empty and result.text_id in semantic_matches.text_id.values and perform_semantic_search:
                segments = segment_text(result.full_text)
                segment_scores = compare_segments_with_query_parallel(segments, user_query_embedding)
                top_segment = max(segment_scores, key=lambda x: x[1])
                combined_data = f"Semantic|Text ID: {result.text_id}|{result.summary}|{top_segment[0]}"
                all_combined_data.append(combined_data)
        try:
            if all_combined_data:
                reranked_response = co.rerank(
                    model='rerank-english-v2.0',
                    query=user_query,
                    documents=all_combined_data,
                    top_n=10
                )


                for idx, result in enumerate(reranked_response.results):
                  combined_data = result.document
                  data_parts = combined_data['text'].split("|")
                  if len(data_parts) >= 4:
                      search_type, text_id_part, summary, quote = data_parts
                      text_id = str(text_id_part.split(":")[-1].strip())
                      summary = summary.strip()
                      quote = quote.strip()
                      text_id_str = f"Text #: {text_id}"
                      source = lincoln_dict.get(text_id_str, {}).get('source', 'Source information not available')
                      full_reranked_results.append({
                          'Rank': idx + 1,
                          'Search Type': search_type,
                          'Text ID': text_id,
                          'Source': source,
                          'Summary': summary,
                          'Key Quote': quote,
                          'Relevance Score': result.relevance_score
                      })

            else:
                full_reranked_results = [] # Handle case where no data is present

        except Exception as e:
                st.error(f"Error in reranking: {e}")
                full_reranked_results = [] # handle the error case

        if full_reranked_results:
           reranked_df = pd.DataFrame(full_reranked_results)
           log_reranking_results(reranking_results_logger, reranked_df, user_query)
        else:
            formatted_input_for_model = None # handle the case where there are no reranking results
    else:
      formatted_input_for_model = None
  else:
    formatted_input_for_model = None

  if formatted_input_for_model:
      messages_for_second_model = [
          {"role": "system", "content": response_prompt},
          {"role": "user", "content": f"User Query: {user_query}\n\n"
                                      f"Initial Answer: {initial_answer}\n\n"
                                      f"{formatted_input_for_model}"}
      ]
      try:
            second_model_response = client.chat.completions.create(
                model="ft:gpt-4o-mini-2024-07-18:personal:nicolay-gpt4o:9tG7Cypl",
                messages=messages_for_second_model,
                temperature=0,
                max_tokens=2000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            response_content = second_model_response.choices[0].message.content
            if response_content:
                try:
                    model_output = json.loads(response_content)
                    doc_match_counter = 0
                    highlight_success_dict = {}
                    highlight_style = """
                                  <style>
                                  mark {
                                    background-color: #90ee90;
                                    color: black;
                                  }
                                  </style>
                              """
                    if "Match Analysis" in model_output:
                      speech = None # Ensure 'speech' is initialized before the loop.
                      for match_key, match_info in model_output["Match Analysis"].items():
                          text_id = match_info.get("Text ID")
                          formatted_text_id = f"Text #: {text_id}"
                          key_quote = match_info.get("Key Quote", "")
                          speech = next((item for item in lincoln_data if item['text_id'] == formatted_text_id), None) # speech is redefined
                          doc_match_counter += 1
                          if speech:
                              formatted_full_text = speech['full_text'].replace("\\n", "<br>")
                              if key_quote in speech['full_text']:
                                 formatted_full_text = formatted_full_text.replace(key_quote, f"<mark>{key_quote}</mark>")
                              else:
                                  formatted_full_text = highlight_key_quote(speech['full_text'], key_quote)
                                  formatted_full_text = formatted_full_text.replace("\\n", "<br>")
                          highlight_success_dict[match_key] = (key_quote in formatted_full_text) if speech else False # Only perform check if speech is defined
                    log_nicolay_model_output(nicolay_data_logger, model_output, user_query, initial_answer, highlight_success_dict)
                except (json.JSONDecodeError, KeyError) as e:
                     st.error(f"Error parsing or accessing JSON response from Nicolay: {e}")
                     model_output = {}

            else:
                model_output = {}
      except Exception as e:
         st.error(f"Error with Nicolay Model API Call {e}")
         model_output = {}
  else:
         model_output = {}

  # Track performance metrics here using defined functions
   # Track performance metrics here using defined functions
  keyword_counts, keyword_hits, keyword_precision, keyword_recall = track_keyword_success(hays_data=[hays_data], keyword_results=keyword_results_df, nicolay_data=nicolay_results_df.to_dict(orient='records'), query=user_query)
  semantic_hits, semantic_precision, semantic_recall = track_semantic_success(semantic_results=semantic_results_df.to_dict(orient='records'), nicolay_data=nicolay_results_df.to_dict(orient='records'), query=user_query)
  rerank_hits, rerank_precision, rerank_avg_rank = track_rerank_success(rerank_results=reranking_results_df.to_dict(orient='records'), query=user_query, ideal_documents=ideal_documents)
  #LLM Eval output
  if model_output:
        llm_eval = evaluate_nicolay_output(user_query, initial_answer, formatted_input_for_model, model_output.get('FinalAnswer', {}), llm_eval_prompt)
  else:
        llm_eval = {}


  return {
      'initial_answer': initial_answer,
      'search_results': search_results,
      'semantic_matches': semantic_matches,
      'full_reranked_results': full_reranked_results,
      'model_output': model_output,
      'keyword_counts': keyword_counts,
      'keyword_hits': keyword_hits,
      'keyword_precision': keyword_precision,
      'keyword_recall': keyword_recall,
      'semantic_hits': semantic_hits,
      'semantic_precision': semantic_precision,
      'semantic_recall': semantic_recall,
      'rerank_hits': rerank_hits,
      'rerank_precision': rerank_precision,
      'rerank_avg_rank': rerank_avg_rank,
      'llm_eval': llm_eval
      }

# Streamlit App
st.title("Nicolay RAG Evaluation")

# Question Selection
if questions_df.empty:
  st.error("The 'benchmark_questions' sheet is empty or could not be loaded. Please check your data source.")
else:
    question_titles = questions_df['question'].tolist()
    selected_question_title = st.selectbox("Select a question:", question_titles)
    selected_question_row = questions_df[questions_df['question'] == selected_question_title].iloc[0]
    selected_question = selected_question_row['question']
    selected_ideal_documents = selected_question_row['ideal_documents'].split(',') if pd.notna(selected_question_row['ideal_documents']) else []
    # Button to run RAG
    if st.button("Run RAG Process"):
        with st.spinner("Running RAG..."):
            rag_results = run_rag_process(selected_question, selected_ideal_documents)

            # Display RAG Results
            st.header("RAG Results")

            # Initial Answer from Hay
            st.subheader("Hay's Initial Answer:")
            st.write(rag_results['initial_answer'])

            # Keyword Search Results
            st.subheader("Keyword Search:")
            if not rag_results['search_results'].empty:
                for idx, result in rag_results['search_results'].iterrows():
                    expander_label = f"**Keyword Match {idx+1}**: *{result['source']}* `{result['text_id']}`"
                    with st.expander(expander_label):
                        st.markdown(f"**Source:** {result['source']}")
                        st.markdown(f"**Text ID:** {result['text_id']}")
                        st.markdown(f"**Summary:**\n{result['summary']}")
                        st.markdown(f"**Key Quote:**\n{result['quote']}")
                        st.markdown(f"**Weighted Score:** {result['weighted_score']}")
                        st.markdown("**Keyword Counts:**")
                        st.json(result['keyword_counts'])
            else:
                st.write("No keyword search results.")
            # Semantic Search Results
            st.subheader("Semantic Search:")
            if not rag_results['semantic_matches'].empty:
                for idx, row in rag_results['semantic_matches'].iterrows():
                    semantic_expander_label = f"**Semantic Match {idx+1}**: *{row['source']}* `Text #: {row['Unnamed: 0']}`"
                    with st.expander(semantic_expander_label, expanded=False):
                            # Display 'source', 'text_id', 'summary'
                        st.markdown(f"**Source:** {row['source']}")
                        st.markdown(f"**Text ID:** {row['Unnamed: 0']}")
                        st.markdown(f"**Summary:**\n{row['summary']}")
                        st.markdown(f"**Key Quote:** {row['TopSegment']}")
                        st.markdown(f"**Similarity Score:** {row['similarities']:.2f}")
            else:
                st.write("No semantic search results.")
            # Reranking Results
            st.subheader("Reranking Results")
            if rag_results['full_reranked_results']:
              for idx, result in enumerate(rag_results['full_reranked_results']):
                    # Display only the top 3 results
                      if idx < 3:
                          expander_label = f"**Reranked Match {idx + 1} ({result['Search Type']} Search)**: `Text ID: {result['Text ID']}`"
                          with st.expander(expander_label):
                              st.markdown(f"Text ID: {result['Text ID']}")
                              st.markdown(f"{result['Source']}")
                              st.markdown(f"{result['Summary']}")
                              st.markdown(f"Key Quote:\n{result['Key Quote']}")
                              st.markdown(f"**Relevance Score:** {result['Relevance Score']:.2f}")
            else:
                 st.write("No reranking results.")

            # Nicolay's Output
            st.subheader("Nicolay's Response:")
            if rag_results['model_output']:
                final_answer = rag_results['model_output'].get("FinalAnswer", {})
                st.markdown(f"**Response:**\n{final_answer.get('Text', 'No response available')}")
                if final_answer.get("References"):
                    st.markdown("**References:**")
                    for reference in final_answer["References"]:
                        st.markdown(f"{reference}")
                doc_match_counter = 0
                highlight_style = """
                    <style>
                    mark {
                        background-color: #90ee90;
                        color: black;
                    }
                    </style>
                    """

                if "Match Analysis" in rag_results['model_output']:
                    st.markdown(highlight_style, unsafe_allow_html=True)
                    for match_key, match_info in rag_results['model_output']["Match Analysis"].items():
                        text_id = match_info.get("Text ID")
                        formatted_text_id = f"Text #: {text_id}"
                        key_quote = match_info.get("Key Quote", "")
                        speech = next((item for item in lincoln_data if item['text_id'] == formatted_text_id), None)
                        doc_match_counter += 1
                        if speech:
                             # Use the doc_match_counter in the expander label
                            expander_label = f"**Match {doc_match_counter}**: *{speech['source']}* `{speech['text_id']}`"
                            with st.expander(expander_label, expanded=False):
                                 st.markdown(f"**Source:** {speech['source']}")
                                 st.markdown(f"**Text ID:** {speech['text_id']}")
                                 st.markdown(f"**Summary:**\n{speech['summary']}")
                                 formatted_full_text = speech['full_text'].replace("\\n", "<br>")
                                 if key_quote in speech['full_text']:
                                     formatted_full_text = formatted_full_text.replace(key_quote, f"<mark>{key_quote}</mark>")
                                 else:
                                     formatted_full_text = highlight_key_quote(speech['full_text'], key_quote)
                                     formatted_full_text = formatted_full_text.replace("\\n", "<br>")
                                 st.markdown(f"**Key Quote:**\n{key_quote}")
                                 st.markdown(f"**Full Text with Highlighted Quote:**", unsafe_allow_html=True)
                                 st.markdown(formatted_full_text, unsafe_allow_html=True)
            else:
                st.write("No Nicolay response.")

            # Evaluation Metrics
            st.header("Performance Metrics")
            st.subheader("Keyword Performance")
            st.write(f"**Weighted Keywords Count:** {rag_results['keyword_counts']}")
            st.write(f"**Keyword Search Hits:** {rag_results['keyword_hits']}")
            st.write(f"**Keyword Precision Rate:** {rag_results['keyword_precision']:.2f}")
            st.write(f"**Keyword Recall Rate:** {rag_results['keyword_recall']:.2f}")

            st.subheader("Semantic Performance")
            st.write(f"**Semantic Search Hits:** {rag_results['semantic_hits']}")
            st.write(f"**Semantic Precision Rate:** {rag_results['semantic_precision']:.2f}")
            st.write(f"**Semantic Recall Rate:** {rag_results['semantic_recall']:.2f}")

            st.subheader("Rerank Performance")
            st.write(f"**Rerank Hits (Top 3):** {rag_results['rerank_hits']}")
            st.write(f"**Rerank Precision (Top 3):** {rag_results['rerank_precision']:.2f}")
            if rag_results['rerank_avg_rank']:
                st.write(f"**Rerank Average Rank:** {rag_results['rerank_avg_rank']:.2f}")
            else:
                st.write("**No Rerank Results**")

            st.subheader("LLM Judge Performance")
            if rag_results['llm_eval']:
                st.write(f"**Factual Accuracy Rating:** {rag_results['llm_eval'].get('factual_accuracy_rating', 'No rating')}")
                st.write(f"**Factual Accuracy Rationale:** {rag_results['llm_eval'].get('factual_accuracy_rationale', 'No rationale')}")
                st.write(f"**Quote Integration Rating:** {rag_results['llm_eval'].get('quote_integration_rating', 'No rating')}")
                st.write(f"**Quote Integration Rationale:** {rag_results['llm_eval'].get('quote_integration_rationale', 'No rationale')}")
                st.write(f"**Citation Accuracy Rating:** {rag_results['llm_eval'].get('citation_accuracy_rating', 'No rating')}")
                st.write(f"**Citation Accuracy Rationale:** {rag_results['llm_eval'].get('citation_accuracy_rationale', 'No rationale')}")
            else:
                st.write("LLM judge output not available.")
