import streamlit as st
from modules.rag_process import RAGProcess
from modules.data_logging import DataLogger, log_keyword_search_results, log_semantic_search_results, log_reranking_results, log_nicolay_model_output
import json
import os
from openai import OpenAI
import cohere
import pygsheets
from google.oauth2 import service_account
import re
import pandas as pd

# Function to highlight key quotes using regex
def highlight_key_quote(text, quote):
    escaped_quote = re.escape(quote)
    pattern = re.compile(escaped_quote, re.IGNORECASE)
    highlighted_text = pattern.sub(f"<mark>{quote}</mark>", text)
    return highlighted_text

# Streamlit app setup
st.set_page_config(
    page_title="Nicolay: Exploring the Speeches of Abraham Lincoln with AI (version 0.2)",
    layout='wide',
    page_icon='🎩'
)

# Initialize API clients
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
openai_api_key = st.secrets["openai_api_key"]
client = OpenAI(api_key=openai_api_key)

os.environ["CO_API_KEY"] = st.secrets["cohere_api_key"]
cohere_api_key = st.secrets["cohere_api_key"]
co = cohere.Client(api_key=cohere_api_key)

# Extract Google Cloud service account details
gcp_service_account = st.secrets["gcp_service_account"]
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
credentials = service_account.Credentials.from_service_account_info(gcp_service_account, scopes=scope)
gc = pygsheets.authorize(custom_credentials=credentials)

# Initialize DataLogger objects
hays_data_logger = DataLogger(gc, 'hays_data')
keyword_results_logger = DataLogger(gc, 'keyword_search_results')
semantic_results_logger = DataLogger(gc, 'semantic_search_results')
reranking_results_logger = DataLogger(gc, 'reranking_results')
nicolay_data_logger = DataLogger(gc, 'nicolay_data')

# Load Lincoln speeches data
#lincoln_data_file_path = 'data/lincoln_speech_corpus.json'
#with open(lincoln_data_file_path, 'r') as file:
#    lincoln_data = json.load(file)

#def load_data_into_session_state():
#    if 'lincoln_data' not in st.session_state:
#        with open('data/lincoln_speech_corpus.json', 'r') as file:
#            st.session_state.lincoln_data = json.load(file)
#    if 'keyword_data' not in st.session_state:
#        with open('data/voyant_word_counts.json', 'r') as file:
#            st.session_state.keyword_data = json.load(file)
#    if 'df' not in st.session_state:
#        st.session_state.df = pd.read_csv("lincoln_index_embedded.csv")

@st.cache_data(persist="disk")
def load_and_prepare_data():
    if 'lincoln_data' not in st.session_state:
        with open('data/lincoln_speech_corpus.json', 'r') as file:
            st.session_state.lincoln_data = json.load(file)
    if 'keyword_data' not in st.session_state:
        with open('data/voyant_word_counts.json', 'r') as file:
            st.session_state.keyword_data = json.load(file)
    if 'df' not in st.session_state:
        st.session_state.df = pd.read_csv("lincoln_index_embedded.csv")
    # Return the variables from session state
    return st.session_state.lincoln_data, st.session_state.keyword_data, st.session_state.df

# Load data using the cached function
lincoln_data, keyword_data, df = load_and_prepare_data()

# Call this function at the beginning of the Streamlit script
#load_data_into_session_state()


# Initialize the RAG Process
rag = RAGProcess(openai_api_key, cohere_api_key, gcp_service_account, hays_data_logger)

# Streamlit Chatbot Interface
st.title("Chat with Hays and Nicolay - in development")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about Abraham Lincoln's speeches:"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        # Process the user query
        st.write("Processing your query...")
        results = rag.run_rag_process(prompt)

        response_json = json.loads(results["response"])  # Parse the response as JSON
        initial_answer = results["initial_answer"]
        final_answer_text = response_json.get("FinalAnswer", {}).get("Text", "Final answer not available.")
        references = response_json.get("FinalAnswer", {}).get("References", [])

        final_answer = f"{final_answer_text}\n\n**References:**\n" + "\n".join([f"- {ref}" for ref in references])

        #with st.chat_message("assistant"):
        #    st.markdown(f"Hays' Response: {initial_answer}")
        #st.session_state.messages.append({"role": "assistant", "content": f"Initial Answer: {initial_answer}"})

        with st.chat_message("assistant"):
            st.markdown(f"Nicolay's Response: {final_answer}")
        st.session_state.messages.append({"role": "assistant", "content": f"Nicolay's response: {final_answer}"})

        with st.expander("**Analysis Metadata**"):
            if "User Query Analysis" in response_json:
                st.markdown("**User Query Analysis:**")
                for key, value in response_json["User Query Analysis"].items():
                    st.markdown(f"- **{key}:** {value}")

            if "Initial Answer Review" in response_json:
                st.markdown("**Initial Answer Review:**")
                for key, value in response_json["Initial Answer Review"].items():
                    st.markdown(f"- **{key}:** {value}")

            if "Match Analysis" in response_json:
                st.markdown("**Match Analysis:**")
                for match_key, match_info in response_json["Match Analysis"].items():
                    st.markdown(f"- **{match_key}:**")
                    for key, value in match_info.items():
                        st.markdown(f"  - {key}: {value}")

            if "Meta Analysis" in response_json:
                st.markdown("**Meta Analysis:**")
                for key, value in response_json["Meta Analysis"].items():
                    st.markdown(f"- **{key}:** {value}")

            if "Model Feedback" in response_json:
                st.markdown("**Model Feedback:**")
                for key, value in response_json["Model Feedback"].items():
                    st.markdown(f"- **{key}:** {value}")

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

        if "Match Analysis" in response_json:
            st.markdown(highlight_style, unsafe_allow_html=True)
            for match_key, match_info in response_json["Match Analysis"].items():
                text_id = match_info.get("Text ID")
                formatted_text_id = f"Text #: {text_id}"
                key_quote = match_info.get("Key Quote", "")

                speech = next((item for item in st.session_state.lincoln_data if item['text_id'] == formatted_text_id), None)

                doc_match_counter += 1
                highlight_success = False

                if speech:
                    expander_label = f"**Match {doc_match_counter}**: *{speech['source']}* `{speech['text_id']}`"
                    with st.expander(expander_label, expanded=False):
                        st.markdown(f"**Source:** {speech['source']}")
                        st.markdown(f"**Text ID:** {speech['text_id']}")
                        st.markdown(f"**Summary:**\n{speech['summary']}")

                        formatted_full_text = speech['full_text'].replace("\n", "<br>")
                        if key_quote in speech['full_text']:
                            formatted_full_text = formatted_full_text.replace(key_quote, f"<mark>{key_quote}</mark>")
                            highlight_success = True
                        else:
                            formatted_full_text = highlight_key_quote(speech['full_text'], key_quote)
                            formatted_full_text = formatted_full_text.replace("\n", "<br>")
                            highlight_success = key_quote in formatted_full_text

                        st.markdown(f"**Key Quote:**\n{key_quote}")
                        st.markdown(f"**Full Text with Highlighted Quote:**", unsafe_allow_html=True)
                        st.markdown(formatted_full_text, unsafe_allow_html=True)

                        highlight_success_dict[match_key] = highlight_success
                else:
                    with st.expander(f"**Match {doc_match_counter}**: Not Found", expanded=False):
                        st.markdown("Full text not found.")
                        highlight_success_dict[match_key] = False

        search_results = results["search_results"]
        semantic_matches = results["semantic_matches"]
        reranked_results = results["reranked_results"]
        model_weighted_keywords = results["model_weighted_keywords"]
        model_year_keywords = results["model_year_keywords"]
        model_text_keywords = results["model_text_keywords"]

        # Ensure this block is only called once
        log_keyword_search_results(keyword_results_logger, search_results, prompt, initial_answer, model_weighted_keywords, model_year_keywords, model_text_keywords)
        log_semantic_search_results(semantic_results_logger, semantic_matches, initial_answer)
        log_reranking_results(reranking_results_logger, reranked_results, prompt)
        log_nicolay_model_output(nicolay_data_logger, response_json, prompt, initial_answer, {})

    except Exception as e:
        st.error(f"An error occurred: {e}")
