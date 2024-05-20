# chatbot.py

import streamlit as st
from modules.rag_process import RAGProcess
import json
import os
from openai import OpenAI
import cohere
import pygsheets
from google.oauth2 import service_account

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
client = OpenAI()
openai_api_key = st.secrets["openai_api_key"]

os.environ["CO_API_KEY"]= st.secrets["cohere_api_key"]
co = cohere.Client()

scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"], scopes = scope)

gc = pygsheets.authorize(custom_credentials=credentials)

#openai_api_key = secrets['openai_api_key']
#cohere_api_key = secrets['cohere_api_key']
#gcp_service_account = secrets['gcp_service_account']

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
