# chatbot.py

import streamlit as st
from modules.rag_process import RAGProcess
from modules.data_logging import DataLogger, log_keyword_search_results, log_semantic_search_results, log_reranking_results, log_nicolay_model_output
import json
import os
from openai import OpenAI
import cohere
import pygsheets
from google.oauth2 import service_account

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

# Initialize the RAG Process
rag = RAGProcess(openai_api_key, cohere_api_key, gcp_service_account)

# Streamlit Chatbot Interface
st.title("Abraham Lincoln Speeches Chatbot")

user_query = st.text_input("Ask me anything about Abraham Lincoln's speeches:")

if st.button("Submit"):
    if user_query:
        st.write("Processing your query...")
        response = rag.run_rag_process(user_query)

        # Log keyword search results
        log_keyword_search_results(keyword_results_logger, search_results, user_query, initial_answer, model_weighted_keywords, model_year_keywords, model_text_keywords)

        # Log semantic search results
        log_semantic_search_results(semantic_results_logger, semantic_matches, initial_answer)

        # Log reranking results
        log_reranking_results(reranking_results_logger, reranked_results, user_query)

        # Log final model output
        log_nicolay_model_output(nicolay_data_logger, model_output, user_query, initial_answer, highlight_success_dict)
        
        st.markdown("### Response")
        st.write(response)
    else:
        st.error("Please enter a query.")
