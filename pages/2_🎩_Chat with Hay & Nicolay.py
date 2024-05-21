mport streamlit as st
from modules.rag_process import RAGProcess
from modules.data_logging import DataLogger, log_keyword_search_results, log_semantic_search_results, log_reranking_results, log_nicolay_model_output
import json
import os
from openai import OpenAI
import cohere
import pygsheets
from google.oauth2 import service_account

# chatbot development - 0.0 - basic UI for RAG search and data logging


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

user_query = st.text_input("Chat interface is in development. The output below comes from the RAG process:")

if st.button("Submit"):
    if user_query:
        try:
            st.write("Processing your query...")
            results = rag.run_rag_process(user_query)

            # Unpack the results
            response = results["response"]
            search_results = results["search_results"]
            semantic_matches = results["semantic_matches"]
            reranked_results = results["reranked_results"]
            initial_answer = results["initial_answer"]
            model_weighted_keywords = results["model_weighted_keywords"]
            model_year_keywords = results["model_year_keywords"]
            model_text_keywords = results["model_text_keywords"]

            st.markdown("### Response")
            st.write(response)

            # Log the data
            log_keyword_search_results(keyword_results_logger, search_results, user_query, initial_answer, model_weighted_keywords, model_year_keywords, model_text_keywords)
            log_semantic_search_results(semantic_results_logger, semantic_matches, initial_answer)
            log_reranking_results(reranking_results_logger, reranked_results, user_query)

            # Ensure response is in the correct format for logging
            log_nicolay_model_output(nicolay_data_logger, json.loads(response), user_query, initial_answer, {})

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a query.")
