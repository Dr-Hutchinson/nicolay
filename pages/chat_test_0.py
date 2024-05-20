# chatbot.py

import streamlit as st
from modules.rag_process import RAGProcess
import json

# Load secrets from a JSON file or Streamlit secrets
with open('secrets.json') as f:
    secrets = json.load(f)

openai_api_key = secrets['openai_api_key']
cohere_api_key = secrets['cohere_api_key']
gcp_service_account = secrets['gcp_service_account']

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
