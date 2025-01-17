import streamlit as st
import pandas as pd
import json
import pygsheets
from modules.rag_process import RAGProcess
from modules.data_logging import DataLogger
from modules.semantic_search import semantic_search
from modules.keyword_search import search_with_dynamic_weights_expanded
from modules.reranking import rerank_results
from modules.prompt_loader import load_prompts

# Initialize app
st.set_page_config(page_title="RAG Benchmarking", layout="wide")

# Load prompts
load_prompts()

#scope = ['https://spreadsheets.google.com/feeds',
#             'https://www.googleapis.com/auth/drive']
#credentials = service_account.Credentials.from_service_account_info(
#                    st.secrets["gcp_service_account"], scopes = scope)

#gc = pygsheets.authorize(custom_credentials=credentials)

# Google Sheets setup
st.write("Loading Google Sheets data...")
try:
    # Load benchmark questions from Google Sheet
    credentials = st.secrets["gcp_service_account"]
    sheet_name = "benchmark_questions"
    benchmark_data = DataLogger(gc=None, sheet_name=sheet_name).sheet.get_as_df()
    st.write("Benchmark questions loaded successfully.")
except Exception as e:
    st.error(f"Error loading Google Sheets data: {e}")

# Initialize RAG process
rag = RAGProcess(
    openai_api_key=st.secrets["openai_api_key"],
    cohere_api_key=st.secrets["cohere_api_key"],
    gcp_service_account=st.secrets["gcp_service_account"],
    hays_data_logger=DataLogger(gc=None, sheet_name='hays_data'),
    keyword_results_logger=DataLogger(gc=None, sheet_name='keyword_search_results')
)

# Process benchmark questions
for idx, row in benchmark_data.iterrows():
    st.subheader(f"Benchmark Question {idx + 1}")
    user_query = row["question"]
    expected_documents = json.loads(row["ideal_documents"])

    # Run RAG process
    try:
        st.write(f"Processing query: {user_query}")
        rag_result = rag.run_rag_process(user_query)

        # Log and display search results
        search_results = rag_result["search_results"]
        semantic_matches = rag_result["semantic_matches"]
        reranked_results = rag_result["reranked_results"]

        st.write("### Keyword Search Results")
        st.dataframe(search_results)

        st.write("### Semantic Search Results")
        st.dataframe(semantic_matches)

        st.write("### Reranked Results")
        st.dataframe(reranked_results)

        # Compare results with expected documents
        st.write("### Benchmark Analysis")
        top_reranked_ids = reranked_results["Text ID"].head(3).tolist()
        matching_expected = len(set(expected_documents) & set(top_reranked_ids))
        st.write(f"Expected documents matched in top 3: {matching_expected}/{len(expected_documents)}")

        # Evaluate Nicolay's response
        final_response = rag_result["response"]
        st.write("### Nicolay's Final Response")
        st.markdown(final_response)

        # LLM evaluation of Nicolay's response
        evaluation_prompt = st.session_state['response_model_system_prompt']
        evaluation = rag.get_final_model_response(user_query, rag_result["initial_answer"], final_response)
        st.write("### LLM Evaluation of Nicolay's Response")
        st.markdown(evaluation)

    except Exception as e:
        st.error(f"Error processing query {idx + 1}: {e}")

# Summary and visualization
st.write("### Benchmark Summary")
# Add further analysis and summary visualization logic here (e.g., charts for effectiveness).
