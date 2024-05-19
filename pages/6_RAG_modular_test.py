import streamlit as st
import json
import pygsheets
from google.oauth2 import service_account
import cohere
import os
import pandas as pd
import numpy as np
from datetime import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import cohere
import pygsheets
from google.oauth2 import service_account

# Importing custom helper modules
from modules.prompt_loader import load_prompts
from modules.keyword_search import load_json, keyword_search
from modules.semantic_search import semantic_search, search_text, get_embedding, segment_text, compare_segments_with_query_parallel
from modules.reranking import rerank_results, format_reranked_results_for_model_input
from modules.data_logging import DataLogger, log_keyword_search_results, log_semantic_search_results, log_reranking_results, log_nicolay_model_output
from modules.misc_helpers import get_source_and_summary, extract_full_text, remove_duplicates, get_full_text_by_id, highlight_key_quote

# version 0.3 - Experiment for making sequential API calls for semantic search.

st.set_page_config(
    page_title="Nicolay: Exploring the Speeches of Abraham Lincoln with AI (version 0.2)",
    layout='wide',
    page_icon='🔍'
)

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

os.environ["CO_API_KEY"] = st.secrets["cohere_api_key"]
co = cohere.Client(api_key=os.environ["CO_API_KEY"])

scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"], scopes=scope)

gc = pygsheets.authorize(custom_credentials=credentials)

api_sheet = gc.open('api_outputs')
api_outputs = api_sheet.sheet1

# Load prompts
load_prompts()

# Now you can use the prompts from session state
keyword_prompt = st.session_state['keyword_model_system_prompt']
response_prompt = st.session_state['response_model_system_prompt']
app_intro = st.session_state['app_intro']
keyword_search_explainer = st.session_state['keyword_search_explainer']
semantic_search_explainer = st.session_state['semantic_search_explainer']
relevance_ranking_explainer = st.session_state['relevance_ranking_explainer']
nicolay_model_explainer = st.session_state['nicolay_model_explainer']

# Initialize DataLoggers
hays_data_logger = DataLogger(gc, 'hays_data')
keyword_results_logger = DataLogger(gc, 'keyword_search_results')
semantic_results_logger = DataLogger(gc, 'semantic_search_results')
reranking_results_logger = DataLogger(gc, 'reranking_results')
nicolay_data_logger = DataLogger(gc, 'nicolay_data')

# Streamlit interface
st.title("Exploring RAG with Nicolay and Hay")

image_url = 'http://danielhutchinson.org/wp-content/uploads/2024/01/nicolay_hay.png'
st.image(image_url, width=600)

st.subheader("**Navigating this App:**")
st.write("Expand the **How It Works?** box below for a walkthrough of the app. Continue to the search interface below to begin exploring Lincoln's speeches.")

with st.expander("**How It Works - Exploring RAG with Hay and Nicolay**"):
    st.write(app_intro)

# Query input
with st.form("Search Interface"):
    st.markdown("Enter your query below:")
    user_query = st.text_input("Query")

    st.write("**Search Options**:")
    st.write("Note that at least one search method must be selected to perform Response and Analysis.")
    perform_keyword_search = st.toggle("Weighted Keyword Search", value=True)
    perform_semantic_search = st.toggle("Semantic Search", value=True)
    perform_reranking = st.toggle("Response and Analysis", value=False, key="reranking")

    if perform_reranking and not (perform_keyword_search or perform_semantic_search):
        st.warning("Response & Analysis requires at least one of the search methods (keyword or semantic).")

    with st.expander("Additional Search Options (In Development)"):
        st.markdown("The Hay model will suggest keywords based on your query, but you can select your own criteria for more focused keyword search using the interface below.")
        st.markdown("Weighted Keywords")
        default_values = [1.0, 1.0, 1.0, 1.0, 1.0]
        user_weighted_keywords = {}

        for i in range(1, 6):
            col1, col2 = st.columns(2)
            with col1:
                keyword = st.text_input(f"Keyword {i}", key=f"keyword_{i}")
            with col2:
                weight = st.number_input(f"Weight for Keyword {i}", min_value=0.0, value=default_values[i-1], step=0.1, key=f"weight_{i}")
            if keyword:
                user_weighted_keywords[keyword] = weight

        st.header("Year and Text Filters")
        user_year_keywords = st.text_input("Year Keywords (comma-separated - example: 1861, 1862, 1863)")
        user_text_keywords = st.multiselect("Text Selection:", ['At Peoria, Illinois', 'A House Divided', 'Eulogy on Henry Clay', 'Farewell Address', 'Cooper Union Address', 'First Inaugural Address', 'Second Inaugural Address', 'July 4th Message to Congress', 'First Annual Message', 'Second Annual Message', 'Third Annual Message', 'Fourth Annual Message', 'Emancipation Proclamation', 'Public Letter to James Conkling', 'Gettysburg Address'])

    submitted = st.form_submit_button("Submit")

    if submitted:
        valid_search_condition = perform_keyword_search or perform_semantic_search

        if valid_search_condition:
            st.subheader("Starting RAG Process: (takes about 30-60 seconds in total)")

            lincoln_speeches_file_path = 'data/lincoln_speech_corpus.json'
            keyword_frequency_file_path = 'data/voyant_word_counts.json'
            lincoln_speeches_embedded = "lincoln_index_embedded.csv"

            lincoln_data = load_json(lincoln_speeches_file_path)
            keyword_data = load_json(keyword_frequency_file_path)

            lincoln_dict = {item['text_id']: item for item in lincoln_data}

            if user_query:
                messages_for_model = [
                    {"role": "system", "content": keyword_prompt},
                    {"role": "user", "content": user_query}
                ]

                response = client.chat_completions.create(
                    model="ft:gpt-3.5-turbo-1106:personal::8XtdXKGK",
                    messages=messages_for_model,
                    temperature=0,
                    max_tokens=500,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )

                msg = response.choices[0].message.content

                api_response_data = json.loads(msg)
                initial_answer = api_response_data['initial_answer']
                model_weighted_keywords = api_response_data['weighted_keywords']
                model_year_keywords = api_response_data['year_keywords']
                model_text_keywords = api_response_data['text_keywords']

                hays_data = {
                    'query': user_query,
                    'initial_answer': initial_answer,
                    'weighted_keywords': model_weighted_keywords,
                    'year_keywords': model_year_keywords,
                    'text_keywords': model_text_keywords,
                    'full_output': msg
                }

                hays_data_logger.record_api_outputs(hays_data)

                if user_weighted_keywords:
                    weighted_keywords = user_weighted_keywords
                    year_keywords = user_year_keywords.split(',') if user_year_keywords else []
                    text_keywords = user_text_keywords if user_text_keywords else []
                else:
                    weighted_keywords = model_weighted_keywords
                    year_keywords = model_year_keywords
                    text_keywords = model_text_keywords

                with st.expander("**Hay's Response**", expanded=True):
                    st.markdown(initial_answer)
                    st.write("**How Does This Work?**")
                    st.write("The Initial Response based on the user query is given by Hay, a finetuned large language model. This response helps Hay steer in the search process by guiding the selection of weighted keywords and informing the semantic search over the Lincoln speech corpus. Compare the Hay's Response Answer with Nicolay's Response and Analysis and the end of the RAG process to see how AI techniques can be used for historical sources.")

                col1, col2 = st.columns(2)

                with col1:
                    if perform_keyword_search:
                        search_results = keyword_search(weighted_keywords, keyword_frequency_file_path, year_keywords, text_keywords, top_n=5)

                        if not search_results:
                            st.markdown("### Keyword Search Results")
                            with st.expander("**How Does This Work?: Dynamically Weighted Keyword Search**"):
                                st.write(keyword_search_explainer)
                            search_results = pd.DataFrame()
                            with st.expander("**No keyword search results found.**"):
                                st.write("No keyword search results found based on your query and Hay's outputs. Try again or modify your query. You can also use the Additional Search Options box above to search for specific terms, speeches, and years.")
                        else:
                            st.markdown("### Keyword Search Results")
                            with st.expander("**How Does This Work?: Dynamically Weighted Keyword Search**"):
                                st.write(keyword_search_explainer)

                            for idx, result in enumerate(search_results, start=1):
                                expander_label = f"**Keyword Match {idx}**: *{result['source']}* `{result['text_id']}`"
                                with st.expander(expander_label):
                                    st.markdown(f"{result['source']}")
                                    st.markdown(f"{result['text_id']}")
                                    st.markdown(f"{result['summary']}")
                                    st.markdown(f"**Key Quote:**\n{result['quote']}")
                                    st.markdown(f"**Weighted Score:** {result['weighted_score']}")
                                    st.markdown("**Keyword Counts:**")
                                    st.json(result['keyword_counts'])

                        with st.expander("**Keyword Search Metadata**"):
                            st.write("**Keyword Search Metadata**")
                            st.write("**User Query:**")
                            st.write(user_query)
                            st.write("**Model Response:**")
                            st.write(initial_answer)
                            st.write("**Weighted Keywords:**")
                            st.json(weighted_keywords)
                            st.write("**Year Keywords:**")
                            st.json(year_keywords)
                            st.write("**Text Keywords:**")
                            st.json(text_keywords)
                            st.write("**Raw Search Results**")
                            st.dataframe(search_results)
                            st.write("**Full Model Output**")
                            st.write(msg)

                            search_results_df = pd.DataFrame(search_results)
                            log_keyword_search_results(keyword_results_logger, search_results_df, user_query, initial_answer, model_weighted_keywords, model_year_keywords, model_text_keywords)

                with col2:
                    if perform_semantic_search:
                        embedding_size = 1536
                        st.markdown("### Semantic Search Results")

                        with st.expander("**How Does This Work?: Semantic Search with HyDE**"):
                            st.write(semantic_search_explainer)

                        progress_text = "Semantic search in progress."
                        my_bar = st.progress(0, text=progress_text)

                        match_counter = 1

                        df = pd.read_csv(lincoln_speeches_embedded)
                        df['full_text'] = df['combined'].apply(extract_full_text)
                        df['embedding'] = df['full_text'].apply(lambda x: get_embedding(x, client) if x else np.zeros(embedding_size))

                        my_bar.progress(20, text=progress_text)

                        df['source'], df['summary'] = zip(*df['Unnamed: 0'].apply(lambda x: get_source_and_summary(x, lincoln_dict)))

                        semantic_matches, user_query_embedding = search_text(df, user_query + initial_answer, client, n=5)

                        my_bar.progress(50, text=progress_text)

                        top_segments = []

                        for idx, row in semantic_matches.iterrows():
                            progress_update = 50 + ((idx + 1) / len(semantic_matches)) * 40
                            progress_update = min(progress_update, 100)
                            my_bar.progress(progress_update / 100, text=progress_text)
                            if match_counter > 5:
                                break

                            semantic_expander_label = f"**Semantic Match {match_counter}**: *{row['source']}* `Text #: {row['Unnamed: 0']}`"
                            with st.expander(semantic_expander_label, expanded=False):
                                st.markdown(f"**Source:** {row['source']}")
                                st.markdown(f"**Text ID:** {row['Unnamed: 0']}")
                                st.markdown(f"**Summary:**\n{row['summary']}")

                                segments = segment_text(row['full_text'])

                                segment_scores = compare_segments_with_query_parallel(segments, user_query_embedding, client)
                                top_segment = max(segment_scores, key=lambda x: x[1])
                                top_segments.append(top_segment[0])
                                st.markdown(f"**Key Quote:** {top_segment[0]}")
                                st.markdown(f"**Similarity Score:** {top_segment[1]:.2f}")

                            match_counter += 1

                        semantic_matches["TopSegment"] = top_segments

                        my_bar.progress(100, text="Semantic search completed.")
                        time.sleep(1)
                        my_bar.empty()

                        with st.expander("**Semantic Search Metadata**"):
                            st.write("**Semantic Search Metadata**")
                            st.dataframe(semantic_matches)

                        log_semantic_search_results(semantic_results_logger, semantic_matches, initial_answer)

                if perform_reranking:
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
                        if result.text_id in search_results.text_id.values and perform_keyword_search:
                            combined_data = f"Keyword|Text ID: {result.text_id}|{result.summary}|{result.quote}"
                            all_combined_data.append(combined_data)
                        elif result.text_id in semantic_matches.text_id.values and perform_semantic_search:
                            segments = segment_text(result.full_text)
                            segment_scores = compare_segments_with_query_parallel(segments, user_query_embedding, client)
                            top_segment = max(segment_scores, key=lambda x: x[1])
                            combined_data = f"Semantic|Text ID: {result.text_id}|{result.summary}|{top_segment[0]}"
                            all_combined_data.append(combined_data)

                    if all_combined_data:
                        st.markdown("### Ranked Search Results")
                        try:
                            reranked_response = rerank_results(
                                query=user_query,
                                documents=all_combined_data,
                                api_key=os.environ["CO_API_KEY"]
                            )
                            with st.expander("**How Does This Work?: Relevance Ranking with Cohere's Rerank**"):
                                st.write(relevance_ranking_explainer)

                            full_reranked_results = []
                            for idx, result in enumerate(reranked_response):
                                combined_data = result.document
                                data_parts = combined_data.split("|")
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
                                    if idx < 3:
                                        expander_label = f"**Reranked Match {idx + 1} ({search_type} Search)**: `Text ID: {text_id}`"
                                        with st.expander(expander_label):
                                            st.markdown(f"Text ID: {text_id}")
                                            st.markdown(f"{source}")
                                            st.markdown(f"{summary}")
                                            st.markdown(f"Key Quote:\n{quote}")
                                            st.markdown(f"**Relevance Score:** {result.relevance_score:.2f}")
                        except Exception as e:
                            st.error("Error in reranking: " + str(e))

                    formatted_input_for_model = format_reranked_results_for_model_input(full_reranked_results)

                    with st.expander("**Result Reranking Metadata**"):
                        reranked_df = pd.DataFrame(full_reranked_results)
                        st.dataframe(reranked_df)
                        st.write("**Formatted Results:**")
                        st.write(formatted_input_for_model)

                    log_reranking_results(reranking_results_logger, reranked_df, user_query)

                    if formatted_input_for_model:
                        messages_for_second_model = [
                            {"role": "system", "content": response_prompt},
                            {"role": "user", "content": f"User Query: {user_query}\n\n"
                                                        f"Initial Answer: {initial_answer}\n\n"
                                                        f"{formatted_input_for_model}"}
                        ]

                        second_model_response = client.chat_completions.create(
                            model="ft:gpt-3.5-turbo-1106:personal::8clf6yi4",
                            messages=messages_for_second_model,
                            temperature=0,
                            max_tokens=2000,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                        )

                        response_content = second_model_response.choices[0].message.content

                        if response_content:
                            model_output = json.loads(response_content)

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

                                    speech = next((item for item in lincoln_data if item['text_id'] == formatted_text_id), None)

                                    doc_match_counter += 1

                                    highlight_success = False

                                    if speech:
                                        expander_label = f"**Match {doc_match_counter}**: *{speech['source']}* `{speech['text_id']}`"
                                        with st.expander(expander_label, expanded=False):
                                            st.markdown(f"**Source:** {speech['source']}")
                                            st.markdown(f"**Text ID:** {speech['text_id']}")
                                            st.markdown(f"**Summary:**\n{speech['summary']}")

                                            formatted_full_text = speech['full_text'].replace("\\n", "<br>")

                                            if key_quote in speech['full_text']:
                                                formatted_full_text = formatted_full_text.replace(key_quote, f"<mark>{key_quote}</mark>")
                                                highlight_success = True
                                            else:
                                                formatted_full_text = highlight_key_quote(speech['full_text'], key_quote)
                                                formatted_full_text = formatted_full_text.replace("\\n", "<br>")
                                                highlight_success = key_quote in formatted_full_text

                                            st.markdown(f"**Key Quote:**\n{key_quote}")
                                            st.markdown(f"**Full Text with Highlighted Quote:**", unsafe_allow_html=True)
                                            st.markdown(formatted_full_text, unsafe_allow_html=True)

                                            highlight_success_dict[match_key] = highlight_success
                                    else:
                                        with st.expander(f"**Match {doc_match_counter}**: Not Found", expanded=False):
                                            st.markdown("Full text not found.")
                                            highlight_success_dict[match_key] = False

                            with st.expander("**Analysis Metadata**"):
                                if "User Query Analysis" in model_output:
                                    st.markdown("**User Query Analysis:**")
                                    for key, value in model_output["User Query Analysis"].items():
                                        st.markdown(f"- **{key}:** {value}")

                                if "Initial Answer Review" in model_output:
                                    st.markdown("**Initial Answer Review:**")
                                    for key, value in model_output["Initial Answer Review"].items():
                                        st.markdown(f"- **{key}:** {value}")

                                if "Match Analysis" in model_output:
                                    st.markdown("**Match Analysis:**")
                                    for match_key, match_info in model_output["Match Analysis"].items():
                                        st.markdown(f"- **{match_key}:**")
                                        for key, value in match_info.items():
                                            st.markdown(f"  - {key}: {value}")

                                if "Meta Analysis" in model_output:
                                    st.markdown("**Meta Analysis:**")
                                    for key, value in model_output["Meta Analysis"].items():
                                        st.markdown(f"- **{key}:** {value}")

                                if "Model Feedback" in model_output:
                                    st.markdown("**Model Feedback:**")
                                    for key, value in model_output["Model Feedback"].items():
                                        st.markdown(f"- **{key}:** {value}")

                                st.write("**Full Model Output:**")
                                st.write(response_content)

                            log_nicolay_model_output(nicolay_data_logger, model_output, user_query, highlight_success_dict)

        else:
            st.error("Search halted: Invalid search condition. Please ensure at least one search method is selected.")
