import streamlit as st
from modules.rag_process import RAGProcess
from modules.data_logging import DataLogger, log_keyword_search_results, log_semantic_search_results, log_reranking_results, log_nicolay_model_output
import json
import os
from openai import OpenAI
import cohere
import pygsheets
from google.oauth2 import service_account
import llamaindex as li  # Import LlamaIndex

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

# Initialize LlamaIndex
index = li.Index()
index.load('llamaindex_memory.json')  # Load existing memory or create new one

# Initialize chat messages history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "What would you like to know about Abraham Lincoln's speeches?"}
    ]

# Function to add messages to the chat history
def add_to_message_history(role: str, content: str) -> None:
    message = {"role": role, "content": str(content)}
    st.session_state.messages.append(message)

# Display the chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_query = st.text_input("Ask me anything about Abraham Lincoln's speeches:")

if st.button("Submit"):
    if user_query:
        try:
            add_to_message_history("user", user_query)
            with st.chat_message("user"):
                st.write(user_query)

            st.write("Processing your query...")

            # Use LlamaIndex to handle memory and interaction
            conversation_context = index.get_context(user_query)
            full_query = f"{conversation_context}\nUser: {user_query}"

            # Run the RAG process
            results = rag.run_rag_process(full_query)

            # Unpack the results
            initial_answer = results["initial_answer"]
            final_response = results["response"]
            search_results = results["search_results"]
            semantic_matches = results["semantic_matches"]
            reranked_results = results["reranked_results"]
            model_weighted_keywords = results["model_weighted_keywords"]
            model_year_keywords = results["model_year_keywords"]
            model_text_keywords = results["model_text_keywords"]

            # Display initial_answer from the Hays model
            add_to_message_history("assistant", initial_answer)
            with st.chat_message("assistant"):
                st.write(initial_answer)

            # Display final_response from the Nicolay model
            add_to_message_history("assistant", final_response)
            with st.chat_message("assistant"):
                st.write(final_response)

            # Log the data
            log_keyword_search_results(keyword_results_logger, search_results, user_query, initial_answer, model_weighted_keywords, model_year_keywords, model_text_keywords)
            log_semantic_search_results(semantic_results_logger, semantic_matches, initial_answer)
            log_reranking_results(reranking_results_logger, reranked_results, user_query)

            # Ensure response is in the correct format for logging
            log_nicolay_model_output(nicolay_data_logger, json.loads(final_response), user_query, initial_answer, {})

            # Update LlamaIndex with the new interaction
            index.update_memory(user_query, final_response)
            index.save('llamaindex_memory.json')  # Save updated memory

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a query.")
