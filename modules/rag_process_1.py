import streamlit as st
import pandas as pd
import json
from datetime import datetime as dt
from modules.rag_process_1 import RAGProcess
from modules.data_logging import DataLogger
from modules.semantic_search import semantic_search
from modules.keyword_search import search_with_dynamic_weights_expanded
from modules.reranking import rerank_results
from modules.prompt_loader import load_prompts
import pygsheets
from google.oauth2 import service_account
import logging

# Suppress debug messages globally
logging.basicConfig(level=logging.WARNING)

# Streamlit App Initialization
st.set_page_config(page_title="RAG Benchmarking", layout="wide")
st.title("RAG Benchmarking and Evaluation Module")

# Google Sheets Client Initialization
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"], scopes=["https://www.googleapis.com/auth/drive"]
)
gc = pygsheets.authorize(custom_credentials=credentials)

# Logger Instances
hays_data_logger = DataLogger(gc=gc, sheet_name="hays_data")
keyword_results_logger = DataLogger(gc=gc, sheet_name="keyword_search_results")
nicolay_data_logger = DataLogger(gc=gc, sheet_name="nicolay_data")
reranking_results_logger = DataLogger(gc=gc, sheet_name="reranking_results")
semantic_results_logger = DataLogger(gc=gc, sheet_name="semantic_search_results")

# Load Prompts
load_prompts()

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state['messages'] = []  # Initialize as an empty list or the expected default value
if 'response_model_system_prompt' not in st.session_state:
    st.session_state['response_model_system_prompt'] = load_prompts()['response_prompt']  # Ensure prompt is loaded

# Load Benchmark Questions from Google Sheets
try:
    benchmark_sheet = gc.open("benchmark_questions").sheet1
    benchmark_data = pd.DataFrame(benchmark_sheet.get_all_records())
    st.success("Benchmark questions loaded successfully.")
except Exception as e:
    st.error(f"Error loading benchmark questions: {e}")

# Preprocess the 'ideal_documents' column to handle comma-separated values
benchmark_data["ideal_documents"] = benchmark_data["ideal_documents"].apply(
    lambda x: [doc.strip() for doc in x.split(",")] if isinstance(x, str) else []
)

# Add dropdown to select a benchmark question
selected_question_index = st.selectbox(
    "Select a Benchmark Question:",
    options=range(len(benchmark_data)),
    format_func=lambda idx: f"{idx + 1}: {benchmark_data.iloc[idx]['question']}"
)

# Add a button to process the selected question
if st.button("Run Benchmark Question"):
    # Get the selected question
    row = benchmark_data.iloc[selected_question_index]
    user_query = row["question"]
    expected_documents = row["ideal_documents"]  # Use preprocessed list

    # Initialize RAG Process
    rag = RAGProcess(
        openai_api_key=st.secrets["openai_api_key"],
        cohere_api_key=st.secrets["cohere_api_key"],
        gcp_service_account=st.secrets["gcp_service_account"],
        hays_data_logger=hays_data_logger,
        keyword_results_logger=keyword_results_logger
    )

    try:
        st.subheader(f"Benchmark Question {selected_question_index + 1}")
        st.write(f"Processing query: {user_query}")

        # Execute RAG Process
        rag_result = rag.run_rag_process(user_query)

        # Display Keyword Search Results
        st.write("### Keyword Search Results")
        search_results = rag_result["search_results"]
        st.dataframe(search_results)

        # Display Semantic Search Results
        st.write("### Semantic Search Results")
        semantic_matches = rag_result["semantic_matches"]
        st.dataframe(semantic_matches)

        # Display Reranked Results
        st.write("### Reranked Results")
        reranked_results = rag_result["reranked_results"]
        st.dataframe(reranked_results)

        # Compare to Benchmark
        st.write("### Benchmark Analysis")
        top_reranked_ids = reranked_results["Text ID"].head(3).tolist()
        matching_expected = len(set(expected_documents) & set(top_reranked_ids))
        st.write(f"Expected documents matched in top 3: {matching_expected}/{len(expected_documents)}")

        # Display Final Response
        #st.write("### Nicolay's Final Response")
        final_response = rag_result["response"]
        #st.markdown(final_response)

        # Evaluate Nicolay Response with LLM
        evaluation_prompt = st.session_state['response_model_system_prompt']

        st.write("### LLM Evaluation of Nicolay's Response")
        st.json(llm_evaluation)

        # Fix logging for proper Google Sheet

        # Log Results
        #nicolay_data_logger.record_api_outputs({
        #    'Benchmark Question': user_query,
        #    'Ideal Documents': expected_documents,
        #    'Matched Documents': top_reranked_ids,
        #    'LLM Evaluation': llm_evaluation
        #})

    except Exception as e:
        st.error(f"Error processing query {selected_question_index + 1}: {e}")

# Summary and Visualization (Future Implementation)
#st.write("### Summary and Visualization")
# Add additional charts or summary metrics for visualization
