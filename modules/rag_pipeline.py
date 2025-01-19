# run_rag_pipeline.py

import streamlit as st
import json
import pygsheets
from google.oauth2 import service_account
import re
from openai import OpenAI
import cohere
import os
import pandas as pd
import numpy as np
from datetime import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor
from modules.data_logging import DataLogger, log_keyword_search_results, log_semantic_search_results, log_reranking_results, log_nicolay_model_output
from modules.reranker import rerank_results, format_reranked_results_for_model_input

# Version 0.4 - Comprehensive RAG Pipeline with Modularized Components and Enhanced Error Handling

st.set_page_config(
    page_title="Nicolay: Exploring the Speeches of Abraham Lincoln with AI (version 0.4)",
    layout='wide',
    page_icon='🔍'
)

# Initialize API Keys from Streamlit Secrets
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
openai_client = OpenAI()

os.environ["CO_API_KEY"] = st.secrets["cohere_api_key"]
cohere_api_key = st.secrets["cohere_api_key"]
co = cohere.Client(cohere_api_key)

# Google Sheets Authorization
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"], scopes=scope)

gc = pygsheets.authorize(custom_credentials=credentials)

# Initialize DataLoggers
hays_data_logger = DataLogger(gc, 'hays_data')
keyword_results_logger = DataLogger(gc, 'keyword_search_results')
semantic_results_logger = DataLogger(gc, 'semantic_search_results')
reranking_results_logger = DataLogger(gc, 'reranking_results')
nicolay_data_logger = DataLogger(gc, 'nicolay_data')

# System prompt loading function
def load_prompt(file_name):
    """Load prompt from a file."""
    with open(file_name, 'r') as file:
        return file.read()

# Function to ensure prompts are loaded into session state
def load_prompts():
    prompts = {
        'keyword_model_system_prompt': 'prompts/keyword_model_system_prompt.txt',
        'response_model_system_prompt': 'prompts/response_model_system_prompt.txt',
        'app_intro': 'prompts/app_intro.txt',
        'keyword_search_explainer': 'prompts/keyword_search_explainer.txt',
        'semantic_search_explainer': 'prompts/semantic_search_explainer.txt',
        'relevance_ranking_explainer': 'prompts/relevance_ranking_explainer.txt',
        'nicolay_model_explainer': 'prompts/nicolay_model_explainer.txt'
    }

    for key, path in prompts.items():
        if key not in st.session_state:
            st.session_state[key] = load_prompt(path)

# Ensure prompts are loaded
load_prompts()

# Assign prompts to variables
keyword_prompt = st.session_state['keyword_model_system_prompt']
response_prompt = st.session_state['response_model_system_prompt']
app_intro = st.session_state['app_intro']
keyword_search_explainer = st.session_state['keyword_search_explainer']
semantic_search_explainer = st.session_state['semantic_search_explainer']
relevance_ranking_explainer = st.session_state['relevance_ranking_explainer']
nicolay_model_explainer = st.session_state['nicolay_model_explainer']

# Streamlit Interface
st.title("Exploring RAG with Nicolay and Hay")

image_url = 'http://danielhutchinson.org/wp-content/uploads/2024/01/nicolay_hay.png'
st.image(image_url, width=600)

st.subheader("**Navigating this App:**")
st.write("Expand the **How It Works?** box below for a walkthrough of the app. Continue to the search interface below to begin exploring Lincoln's speeches.")

with st.expander("**How It Works - Exploring RAG with Hay and Nicolay**"):
    st.write(app_intro)

# Query input form
with st.form("Search Interface"):

    st.markdown("Enter your query below:")
    user_query = st.text_input("Query")

    st.write("**Search Options**:")
    st.write("Note that at least one search method must be selected to perform Response and Analysis.")
    perform_keyword_search = st.toggle("Weighted Keyword Search", value=True)
    perform_semantic_search = st.toggle("Semantic Search", value=True)
    # Always display the reranking toggle
    perform_reranking = st.toggle("Response and Analysis", value=True, key="reranking")

    # Display a warning message if reranking is selected without any search methods
    if perform_reranking and not (perform_keyword_search or perform_semantic_search):
        st.warning("Response & Analysis requires at least one of the search methods (keyword or semantic).")

    with st.expander("Additional Search Options (In Development)"):

        st.markdown("The Hay model will suggest keywords based on your query, but you can select your own criteria for more focused keyword search using the interface below.")
        st.markdown("Weighted Keywords")
        default_values = [1.0, 1.0, 1.0, 1.0, 1.0]  # Default weights as floats
        user_weighted_keywords = {}

        for i in range(1, 6):
            col1, col2 = st.columns(2)
            with col1:
                keyword = st.text_input(f"Keyword {i}", key=f"keyword_{i}")
            with col2:
                weight = st.number_input(f"Weight for Keyword {i}", min_value=0.0, value=default_values[i-1], step=0.1, key=f"weight_{i}")
            if keyword:
                user_weighted_keywords[keyword] = weight

        # User input for year and text keywords
        st.header("Year and Text Filters")
        user_year_keywords = st.text_input("Year Keywords (comma-separated - example: 1861, 1862, 1863)")
        user_text_keywords = st.multiselect("Text Selection:", [
            'At Peoria, Illinois', 'A House Divided', 'Eulogy on Henry Clay', 'Farewell Address',
            'Cooper Union Address', 'First Inaugural Address', 'Second Inaugural Address',
            'July 4th Message to Congress', 'First Annual Message', 'Second Annual Message',
            'Third Annual Message', 'Fourth Annual Message', 'Emancipation Proclamation',
            'Public Letter to James Conkling', 'Gettysburg Address'
        ])

    submitted = st.form_submit_button("Submit")

    if submitted:
        valid_search_condition = perform_keyword_search or perform_semantic_search

        if valid_search_condition:

            st.subheader("Starting RAG Process: (takes about 30-60 seconds in total)")

            # Load data
            lincoln_speeches_file_path = 'data/lincoln_speech_corpus.json'
            keyword_frequency_file_path = 'data/voyant_word_counts.json'
            lincoln_speeches_embedded = "data/lincoln_index_embedded.csv"

            # Define functions
            def load_json(file_path):
                with open(file_path, 'r') as file:
                    data = json.load(file)
                return data

            lincoln_data = load_json(lincoln_speeches_file_path)
            keyword_data = load_json(keyword_frequency_file_path)

            # Convert JSON data to a dictionary with 'text_id' as the key for easy access
            lincoln_dict = {item['text_id']: item for item in lincoln_data}

            # Function for loading JSON 'text_id' for comparison for semantic search results
            def get_source_and_summary(text_id):
                # Ensure text_id is in the correct format
                text_id_str = text_id.strip()
                return lincoln_dict.get(text_id_str, {}).get('source', 'Unknown Source'), lincoln_dict.get(text_id_str, {}).get('summary', '')

            # Function to find instances with expanded search
            def find_instances_expanded_search(dynamic_weights, original_weights, data, year_keywords=None, text_keywords=None, top_n=5, context_size=1000):
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

                        match_source_year = not year_keywords or any(str(year).strip() in source_lower for year in year_keywords)
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
                                end_quote = min(len(entry['full_text']), highest_original_weighted_position + context_size // 3)
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

            # Function to get embeddings
            def get_embedding(text, model="text-embedding-ada-002"):
                text = text.replace("\n", " ")
                response = openai_client.embeddings.create(input=[text], model=model)
                return np.array(response.data[0].embedding)

            # Function to calculate cosine similarity
            def cosine_similarity(vec1, vec2):
                dot_product = np.dot(vec1, vec2)
                norm_vec1 = np.linalg.norm(vec1)
                norm_vec2 = np.linalg.norm(vec2)
                return dot_product / (norm_vec1 * norm_vec2)

            # Function to perform semantic search
            def search_text(df, user_query, n=5):
                user_query_embedding = get_embedding(user_query)
                df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, user_query_embedding))
                top_n = df.sort_values("similarities", ascending=False).head(n)
                top_n["UserQuery"] = user_query  # Add 'UserQuery' column to the DataFrame
                return top_n, user_query_embedding

            # Function to segment text
            def segment_text(text, segment_size=500, overlap=100):
                words = text.split()
                segments = []
                for i in range(0, len(words), segment_size - overlap):
                    segment = words[i:i + segment_size]
                    segments.append(' '.join(segment))
                return segments

            # Function to compare segments with query embedding in parallel
            def compare_segments_with_query_parallel(segments, query_embedding):
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(get_embedding, segment) for segment in segments]
                    segment_embeddings = [future.result() for future in futures]
                    return [(segments[i], cosine_similarity(segment_embeddings[i], query_embedding)) for i in range(len(segments))]

            # Function to extract full text from records
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

            # Function to remove duplicates based on 'text_id'
            def remove_duplicates(search_results, semantic_matches):
                combined_results = pd.concat([search_results, semantic_matches])
                deduplicated_results = combined_results.drop_duplicates(subset='text_id')
                return deduplicated_results

            # Function to highlight key quotes in text
            def highlight_key_quote(text, key_quote):
                parts = key_quote.split("...")
                if len(parts) >= 2:
                    pattern = re.escape(parts[0]) + r"\s*.*?\s*" + re.escape(parts[-1]) + r"[.;,]?"
                else:
                    pattern = re.escape(key_quote) + r"\s*[.;,]?"
                regex = re.compile(pattern, re.IGNORECASE)
                matches = regex.findall(text)
                for match in matches:
                    text = text.replace(match, f"<mark>{match}</mark>")
                return text

            # Function to get full text by 'text_id'
            def get_full_text_by_id(text_id, data):
                return next((item['full_text'] for item in data if item['text_id'] == text_id), None)

            # Function to record API outputs (Deprecated as per data_logging.py)
            # def record_api_outputs(): ...

            if user_query:

                # Construct the messages for the model
                messages_for_model = [
                    {"role": "system", "content": keyword_prompt},
                    {"role": "user", "content": user_query}
                ]

                # Send the messages to the fine-tuned model
                response = openai_client.chat.completions.create(
                    model="ft:gpt-4o-mini-2024-07-18:personal:hays-gpt4o:9tFqrYwI",  # Hays finetuned model, GPT-4O
                    messages=messages_for_model,
                    temperature=0,
                    max_tokens=500,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )

                msg = response.choices[0].message.content

                # Parse the response to extract generated keywords
                try:
                    api_response_data = json.loads(msg)
                    initial_answer = api_response_data.get('initial_answer', 'No initial answer available.')
                    model_weighted_keywords = api_response_data.get('weighted_keywords', {})
                    model_year_keywords = api_response_data.get('year_keywords', [])
                    model_text_keywords = api_response_data.get('text_keywords', [])
                except json.JSONDecodeError:
                    st.error("Failed to parse the model's response. Ensure the model returns valid JSON.")
                    initial_answer = 'No initial answer available.'
                    model_weighted_keywords = {}
                    model_year_keywords = []
                    model_text_keywords = {}

                # Log Hays data
                hays_data = {
                    'query': user_query,
                    'initial_answer': initial_answer,
                    'weighted_keywords': model_weighted_keywords,
                    'year_keywords': model_year_keywords,
                    'text_keywords': model_text_keywords,
                    'full_output': msg
                }

                hays_data_logger.record_api_outputs(hays_data)

                # Check if user provided any custom weighted keywords
                if user_weighted_keywords:
                    # Use user-provided keywords
                    weighted_keywords = user_weighted_keywords
                    year_keywords = [year.strip() for year in user_year_keywords.split(',')] if user_year_keywords else []
                    text_keywords = user_text_keywords if user_text_keywords else []
                else:
                    # Use model-generated keywords
                    weighted_keywords = model_weighted_keywords
                    year_keywords = model_year_keywords
                    text_keywords = model_text_keywords

                with st.expander("**Hay's Response**", expanded=True):
                    st.markdown(initial_answer)
                    st.write("**How Does This Work?**")
                    st.write("The Initial Response based on the user query is given by Hay, a finetuned large language model. This response helps Hay steer in the search process by guiding the selection of weighted keywords and informing the semantic search over the Lincoln speech corpus. Compare the Hay's Response Answer with Nicolay's Response and Analysis and the end of the RAG process to see how AI techniques can be used for historical sources.")

                # Use st.columns to create two columns
                col1, col2 = st.columns(2)

                # Display keyword search results in the first column
                with col1:

                    # Perform the dynamically weighted search
                    if perform_keyword_search:
                        search_results = find_instances_expanded_search(
                            dynamic_weights=weighted_keywords,
                            original_weights=weighted_keywords,  # Assuming original_weights is the same as dynamic_weights
                            data=lincoln_data,
                            year_keywords=year_keywords,
                            text_keywords=text_keywords,
                            top_n=5  # You can adjust the number of results
                        )

                        # Convert search_results to DataFrame if it's a list
                        if isinstance(search_results, list):
                            search_results = pd.DataFrame(search_results)

                        # Check if keyword search results are empty
                        if search_results.empty:

                            st.markdown("### Keyword Search Results")

                            with st.expander("**How Does This Work?: Dynamically Weighted Keyword Search**"):
                                st.write(keyword_search_explainer)

                            # Display message for no results found
                            with st.expander("**No keyword search results found.**"):
                                st.write("No keyword search results found based on your query and Hay's outputs. Try again or modify your query. You can also use the Additional Search Options box above to search for specific terms, speeches, and years.")
                        else:
                            st.markdown("### Keyword Search Results")

                            with st.expander("**How Does This Work?: Dynamically Weighted Keyword Search**"):
                                st.write(keyword_search_explainer)

                            for idx, result in search_results.iterrows():
                                expander_label = f"**Keyword Match {idx + 1}**: *{result['source']}* `{result['text_id']}`"
                                with st.expander(expander_label):
                                    st.markdown(f"**Source:** {result['source']}")
                                    st.markdown(f"**Text ID:** {result['text_id']}")
                                    st.markdown(f"**Summary:**\n{result['summary']}")
                                    st.markdown(f"**Key Quote:**\n{result['quote']}")  # Display the full expanded quote
                                    st.markdown(f"**Weighted Score:** {result['weighted_score']}")
                                    st.markdown("**Keyword Counts:**")
                                    st.json(result['keyword_counts'])

                        # Display "Keyword Search Metadata" expander
                        with st.expander("**Keyword Search Metadata**"):
                            st.write("**Keyword Search Metadata**")
                            st.write("**User Query:**")
                            st.write(user_query)
                            st.write("**Model Response:**")
                            st.write(initial_answer)
                            st.write("**Weighted Keywords:**")
                            st.json(weighted_keywords)  # Display the weighted keywords
                            st.write("**Year Keywords:**")
                            st.json(year_keywords)
                            st.write("**Text Keywords:**")
                            st.json(text_keywords)
                            st.write("**Raw Search Results**")
                            st.dataframe(search_results)
                            st.write("**Full Model Output**")
                            st.write(msg)

                            # Log keyword search results
                            log_keyword_search_results(
                                keyword_results_logger=keyword_results_logger,
                                search_results=search_results,
                                user_query=user_query,
                                initial_answer=initial_answer,
                                model_weighted_keywords=weighted_keywords,
                                model_year_keywords=year_keywords,
                                model_text_keywords=text_keywords
                            )

                # Display semantic search results in the second column
                with col2:
                    if perform_semantic_search:
                        embedding_size = 1536
                        st.markdown("### Semantic Search Results")

                        with st.expander("**How Does This Work?: Semantic Search with HyDE**"):
                            st.write(semantic_search_explainer)

                        # Before starting the semantic search
                        progress_text = "Semantic search in progress."
                        my_bar = st.progress(0, text=progress_text)

                        # Initialize the match counter
                        match_counter = 1

                        df = pd.read_csv(lincoln_speeches_embedded)
                        df['full_text'] = df['combined'].apply(extract_full_text)
                        df['embedding'] = df['full_text'].apply(lambda x: get_embedding(x) if x else np.zeros(embedding_size))

                        # After calculating embeddings for the dataset
                        my_bar.progress(20, text=progress_text)  # Update to 20% after embeddings

                        df['source'], df['summary'] = zip(*df['text_id'].apply(get_source_and_summary))

                        # Perform initial semantic search, using HyDE approach
                        semantic_matches, user_query_embedding = search_text(df, user_query + initial_answer, n=5)

                        # After performing the initial semantic search
                        my_bar.progress(50, text=progress_text)  # Update to 50% after initial search

                        top_segments = []  # List to store top segments for each match

                        # Loop for top semantic matches
                        for idx, row in semantic_matches.iterrows():
                            # Update progress bar based on the index
                            progress_update = 50 + ((idx + 1) / len(semantic_matches)) * 40
                            progress_update = min(progress_update, 100)  # Ensure it doesn't exceed 100
                            my_bar.progress(progress_update / 100, text=progress_text)  # Divide by 100 if using float scale

                            semantic_expander_label = f"**Semantic Match {match_counter}**: *{row['source']}* `Text #: {row['text_id']}`"
                            with st.expander(semantic_expander_label, expanded=False):
                                # Display 'source', 'text_id', 'summary'
                                st.markdown(f"**Source:** {row['source']}")
                                st.markdown(f"**Text ID:** {row['text_id']}")
                                st.markdown(f"**Summary:**\n{row['summary']}")

                                # Process for finding key quotes remains the same
                                segments = segment_text(row['full_text'])  # Use 'full_text' for segmenting

                                segment_scores = compare_segments_with_query_parallel(segments, user_query_embedding)
                                if segment_scores:
                                    top_segment = max(segment_scores, key=lambda x: x[1])
                                    top_segments.append(top_segment[0])  # Store top segment for logging
                                    st.markdown(f"**Key Quote:** {top_segment[0]}")
                                    st.markdown(f"**Similarity Score:** {top_segment[1]:.2f}")
                                else:
                                    top_segments.append("No relevant segment found.")
                                    st.markdown("**Key Quote:** No relevant segment found.")
                                    st.markdown("**Similarity Score:** 0.00")

                            # Increment the match counter
                            match_counter += 1

                        semantic_matches["TopSegment"] = top_segments  # Add TopSegment column to semantic_matches DataFrame

                        my_bar.progress(100, text="Semantic search completed.")
                        time.sleep(1)
                        my_bar.empty()  # Remove the progress bar

                        with st.expander("**Semantic Search Metadata**"):
                            st.write("**Semantic Search Metadata**")
                            st.dataframe(semantic_matches)

                        # Log semantic search results
                        log_semantic_search_results(
                            semantic_results_logger=semantic_results_logger,
                            semantic_matches=semantic_matches
                        )

                # Reranking results with Cohere's Reranker API Endpoint
                if perform_reranking:

                    # Prepare documents for reranking
                    quotes_for_cohere = []
                    text_id_mapping = {}  # To map back reranked quotes to their metadata

                    # Collect documents from keyword search
                    if perform_keyword_search and not search_results.empty:
                        for _, row in search_results.iterrows():
                            search_type = "Keyword"
                            text_id_str = row['text_id']
                            summary_str = row['summary']
                            quote_str = row['quote'] if pd.notna(row['quote']) else "No key quote available."
                            cleaned_quote = ' '.join(quote_str.split())
                            quotes_for_cohere.append(cleaned_quote)
                            text_id_mapping[cleaned_quote] = {
                                "Search Type": search_type,
                                "Text ID": text_id_str,
                                "Source": row['source'],
                                "Summary": summary_str,
                                "Key Quote": quote_str
                            }

                    # Collect documents from semantic search
                    if perform_semantic_search and not semantic_matches.empty:
                        for _, row in semantic_matches.iterrows():
                            search_type = "Semantic"
                            text_id_str = row['text_id']
                            summary_str = row['summary']
                            quote_str = row['TopSegment'] if pd.notna(row['TopSegment']) else "No key quote available."
                            cleaned_quote = ' '.join(quote_str.split())
                            quotes_for_cohere.append(cleaned_quote)
                            text_id_mapping[cleaned_quote] = {
                                "Search Type": search_type,
                                "Text ID": text_id_str,
                                "Source": row['source'],
                                "Summary": summary_str,
                                "Key Quote": quote_str
                            }

                    # Ensure all documents are strings
                    quotes_for_cohere = [str(doc) for doc in quotes_for_cohere]

                    # Validate documents
                    invalid_docs = [doc for doc in quotes_for_cohere if not isinstance(doc, str) or "|" not in doc]
                    if invalid_docs:
                        st.error(f"Invalid documents for reranking: {invalid_docs}")
                        reranked_df = pd.DataFrame()
                    else:
                        # Rerank the documents
                        reranked_results = rerank_results(
                            query=user_query,
                            documents=quotes_for_cohere,
                            api_key=cohere_api_key,
                            top_n=10
                        )

                        if not reranked_results:
                            st.error("No reranked results returned from Cohere.")
                            reranked_df = pd.DataFrame()
                        else:
                            # Map reranked quotes back to metadata
                            reranked_data = []
                            for i, item in enumerate(reranked_results):
                                reranked_quote = item.document if isinstance(item.document, str) else item.document.get('text', '')
                                relevance_score = item.relevance_score

                                # Retrieve the metadata using the quote
                                metadata = text_id_mapping.get(reranked_quote, {})

                                reranked_data.append({
                                    "Rank": i + 1,
                                    "Search Type": metadata.get("Search Type", "N/A"),
                                    "Text ID": metadata.get("Text ID", "N/A"),
                                    "Source": metadata.get("Source", "Unknown Source"),
                                    "Summary": metadata.get("Summary", "N/A"),
                                    "Key Quote": metadata.get("Key Quote", "No key quote available."),
                                    "Relevance Score": relevance_score
                                })

                            reranked_df = pd.DataFrame(reranked_data)

                            # Log reranking results
                            log_reranking_results(
                                reranking_results_logger=reranking_results_logger,
                                reranked_df=reranked_df
                            )

                            # Format reranked results for Nicolay model input
                            formatted_input_for_model = format_reranked_results_for_model_input(reranked_df)

                            # Display full reranked results in an expander
                            with st.expander("**Result Reranking Metadata**"):
                                st.dataframe(reranked_df)
                                st.write("**Formatted Results:**")
                                st.write(formatted_input_for_model)

                            # Proceed with Nicolay Model Call
                            if formatted_input_for_model:
                                # Construct the message for the Nicolay model
                                messages_for_second_model = [
                                    {"role": "system", "content": response_prompt},
                                    {"role": "user", "content": f"User Query: {user_query}\n\n"
                                                                f"Initial Answer: {initial_answer}\n\n"
                                                                f"{formatted_input_for_model}"}
                                ]

                                # Send the messages to the Nicolay finetuned model
                                second_model_response = openai_client.chat.completions.create(
                                    model="ft:gpt-4o-mini-2024-07-18:personal:nicolay-gpt4o:9tG7Cypl",  # Nicolay finetuned model, GPT-4O
                                    messages=messages_for_second_model,
                                    temperature=0,
                                    max_tokens=2000,
                                    top_p=1,
                                    frequency_penalty=0,
                                    presence_penalty=0
                                )

                                # Process and display the model's response
                                response_content = second_model_response.choices[0].message.content

                                if response_content:  # Assuming 'response_content' is the output from the second model
                                    try:
                                        model_output = json.loads(response_content)
                                    except json.JSONDecodeError:
                                        st.error("Nicolay model output was not valid JSON.")
                                        model_output = {}

                                    if model_output:
                                        # Displaying the Final Answer
                                        st.header("Nicolay's Response & Analysis:")

                                        with st.expander("**How Does This Work?: Nicolay's Response and Analysis**"):
                                            st.write(nicolay_model_explainer)
                                        with st.expander("**Nicolay's Response**", expanded=True):
                                            final_answer = model_output.get("FinalAnswer", {})
                                            st.markdown(f"**Response:**\n{final_answer.get('Text', 'No response available')}")
                                            if final_answer.get("References"):
                                                st.markdown("**References:**")
                                                for reference in final_answer["References"]:
                                                    st.markdown(f"{reference}")

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
                                            st.markdown(highlight_style, unsafe_allow_html=True)
                                            for match_key, match_info in model_output["Match Analysis"].items():
                                                text_id = match_info.get("Text ID")
                                                formatted_text_id = f"Text #: {text_id}"
                                                key_quote = match_info.get("Key Quote", "")

                                                speech = lincoln_dict.get(formatted_text_id, None)

                                                # Increment the counter for each match
                                                doc_match_counter += 1

                                                # Initialize highlight_success for each iteration
                                                highlight_success = False  # Flag to track highlighting success

                                                if speech:
                                                    expander_label = f"**Match {doc_match_counter}**: *{speech['source']}* `{speech['text_id']}`"
                                                    with st.expander(expander_label, expanded=False):
                                                        st.markdown(f"**Source:** {speech['source']}")
                                                        st.markdown(f"**Text ID:** {speech['text_id']}")
                                                        st.markdown(f"**Summary:**\n{speech['summary']}")

                                                        # Replace line breaks for HTML display
                                                        formatted_full_text = speech['full_text'].replace("\\n", "<br>")

                                                        # Attempt direct highlighting
                                                        if key_quote and key_quote in speech['full_text']:
                                                            formatted_full_text = formatted_full_text.replace(key_quote, f"<mark>{key_quote}</mark>")
                                                            highlight_success = True
                                                        else:
                                                            # If direct highlighting fails, use regex-based approach
                                                            formatted_full_text = highlight_key_quote(speech['full_text'], key_quote)
                                                            formatted_full_text = formatted_full_text.replace("\\n", "<br>")
                                                            # Check if highlighting was successful with regex approach
                                                            highlight_success = key_quote in formatted_full_text

                                                        st.markdown(f"**Key Quote:**\n{key_quote}")
                                                        st.markdown(f"**Full Text with Highlighted Quote:**", unsafe_allow_html=True)
                                                        st.markdown(formatted_full_text, unsafe_allow_html=True)

                                                        # Update highlight_success_dict for the current match
                                                        highlight_success_dict[match_key] = highlight_success
                                                else:
                                                    with st.expander(f"**Match {doc_match_counter}**: Not Found", expanded=False):
                                                        st.markdown("Full text not found.")
                                                        highlight_success_dict[match_key] = False  # Indicate failure as text not found

                                        # Displaying the Analysis Metadata
                                        with st.expander("**Analysis Metadata**"):
                                            # Displaying User Query Analysis
                                            if "User Query Analysis" in model_output:
                                                st.markdown("**User Query Analysis:**")
                                                for key, value in model_output["User Query Analysis"].items():
                                                    st.markdown(f"- **{key}:** {value}")

                                            # Displaying Initial Answer Review
                                            if "Initial Answer Review" in model_output:
                                                st.markdown("**Initial Answer Review:**")
                                                for key, value in model_output["Initial Answer Review"].items():
                                                    st.markdown(f"- **{key}:** {value}")

                                            # Displaying Match Analysis
                                            if "Match Analysis" in model_output:
                                                st.markdown("**Match Analysis:**")
                                                for match_key, match_info in model_output["Match Analysis"].items():
                                                    st.markdown(f"- **{match_key}:**")
                                                    for key, value in match_info.items():
                                                        st.markdown(f"  - {key}: {value}")

                                            # Displaying Meta Analysis
                                            if "Meta Analysis" in model_output:
                                                st.markdown("**Meta Analysis:**")
                                                for key, value in model_output["Meta Analysis"].items():
                                                    st.markdown(f"- **{key}:** {value}")

                                            # Displaying Model Feedback
                                            if "Model Feedback" in model_output:
                                                st.markdown("**Model Feedback:**")
                                                for key, value in model_output["Model Feedback"].items():
                                                    st.markdown(f"- **{key}:** {value}")

                                            st.write("**Full Model Output:**")
                                            st.write(response_content)

                                        # Log Nicolay model output
                                        log_nicolay_model_output(
                                            nicolay_data_logger=nicolay_data_logger,
                                            model_output=model_output,
                                            user_query=user_query,
                                            highlight_success_dict=highlight_success_dict
                                        )
                                else:
                                    st.error("Nicolay model did not return any content.")

                else:
                    st.error("Search halted: Invalid search condition. Please ensure at least one search method is selected.")
