import streamlit as st
from modules.rag_process import RAGProcess
from modules.data_logging import DataLogger, log_keyword_search_results, log_semantic_search_results, log_reranking_results, log_nicolay_model_output
import json
import os
from openai import OpenAI
import cohere
import pygsheets
from google.oauth2 import service_account

# chatbot development - 0.1 - basic UI for RAG search and data logging


st.set_page_config(
    page_title="Nicolay: Exploring the Speeches of Abraham Lincoln with AI (version 0.2)",
    layout='wide',
    page_icon='🎩'
)

# Set environment variables and initialize API clients
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
client = OpenAI()
openai_api_key = st.secrets["openai_api_key"]

os.environ["CO_API_KEY"] = st.secrets["cohere_api_key"]
co = cohere.Client()
cohere_api_key = st.secrets["cohere_api_key"]

# Extract Google Cloud service account details
gcp_service_account = st.secrets["gcp_service_account"]

# Initialize Google Sheets client
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
credentials = service_account.Credentials.from_service_account_info(gcp_service_account, scopes=scope)
gc = pygsheets.authorize(custom_credentials=credentials)

# Initialize DataLogger objects for each type of data to log
hays_data_logger = DataLogger(gc, 'hays_data')
keyword_results_logger = DataLogger(gc, 'keyword_search_results')
semantic_results_logger = DataLogger(gc, 'semantic_search_results')
reranking_results_logger = DataLogger(gc, 'reranking_results')
nicolay_data_logger = DataLogger(gc, 'nicolay_data')

# Initialize the RAG Process
rag = RAGProcess(openai_api_key, cohere_api_key, gcp_service_account, hays_data_logger)

# Streamlit Chatbot Interface
st.title("Chat with Hays and Nicolay - in development")

user_query = st.text_input("Ask me anything about Abraham Lincoln's speeches:")

st.title("Chat with Hays and Nicolay - in development")

user_query = st.text_input("Ask me anything about Abraham Lincoln's speeches:")

st.title("Chat with Hays and Nicolay - in development")

# Use unique keys for the text_input widget
user_query_hays = st.text_input("Ask me anything about Abraham Lincoln's speeches:", key="hays")

if st.button("Submit", key="submit_hays"):
    if user_query_hays:
        try:
            st.write("Processing your query...")
            results = rag.run_rag_process(user_query_hays)

            # Unpack the results
            initial_answer = results["initial_answer"]
            final_response = json.loads(results["response"])
            search_results = results["search_results"]
            semantic_matches = results["semantic_matches"]
            reranked_results = results["reranked_results"]
            model_weighted_keywords = results["model_weighted_keywords"]
            model_year_keywords = results["model_year_keywords"]
            model_text_keywords = results["model_text_keywords"]

            st.markdown("### Initial Answer")
            st.write(initial_answer, key="initial_hays")

            st.markdown("### Final Answer")
            st.write(final_response['FinalAnswer']['Text'], key="final_hays")

            with st.expander("Search Metadata", key="metadata_hays"):
                st.json(final_response)

            # Log the data
            log_keyword_search_results(keyword_results_logger, search_results, user_query_hays, initial_answer, model_weighted_keywords, model_year_keywords, model_text_keywords)
            log_semantic_search_results(semantic_results_logger, semantic_matches, initial_answer)
            log_reranking_results(reranking_results_logger, reranked_results, user_query_hays)
            log_nicolay_model_output(nicolay_data_logger, final_response, user_query_hays, initial_answer, {})

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a query.")
