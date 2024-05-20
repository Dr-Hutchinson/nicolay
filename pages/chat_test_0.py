# chatbot.py

import streamlit as st
from modules.rag_process import RAGProcess
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
        st.markdown("### Response")
        st.write(response)
    else:
        st.error("Please enter a query.")
