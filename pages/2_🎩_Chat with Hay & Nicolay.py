import streamlit as st
from modules.rag_process import RAGProcess
from modules.data_logging import DataLogger, log_keyword_search_results, log_semantic_search_results, log_reranking_results, log_nicolay_model_output
from modules.data_utils import load_lincoln_speech_corpus, load_voyant_word_counts, load_lincoln_index_embedded
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

if 'lincoln_data' not in st.session_state:
    st.session_state.lincoln_data = load_lincoln_speech_corpus()
if 'voyant_data' not in st.session_state:
    st.session_state.voyant_data = load_voyant_word_counts()
if 'lincoln_index_df' not in st.session_state:
    st.session_state.lincoln_index_df = load_lincoln_index_embedded()

lincoln_data = st.session_state.lincoln_data
voyant_data = st.session_state.voyant_data
lincoln_index_df = st.session_state.lincoln_index_df

# Initialize the RAG Process
rag = RAGProcess(openai_api_key, cohere_api_key, gcp_service_account, hays_data_logger, keyword_results_logger)

# Streamlit Chatbot Interface
st.title("Chat with Hays and Nicolay - in development")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything about Abraham Lincoln's speeches:"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        # Process the user query
        st.write("Processing your query...")
        results = rag.run_rag_process(prompt)

        # Print the raw response for debugging
        #st.write("Raw response content:", results["response"])

        # Try to parse the JSON response
        try:
            response_json = json.loads(results["response"])  # Parse the response as JSON
        except json.JSONDecodeError as e:
            st.error(f"JSON decoding error: {e}")
            st.error(f"Raw JSON response: {results['response']}")
            raise e

        initial_answer = results["initial_answer"]
        final_answer_text = response_json.get("FinalAnswer", {}).get("Text", "Final answer not available.")
        references = response_json.get("FinalAnswer", {}).get("References", [])

        # Combine final answer text with references
        final_answer = f"{final_answer_text}\n\n**References:**\n" + "\n".join([f"- {ref}" for ref in references])

        # Display final answer with references
        with st.chat_message("assistant"):
            st.markdown(f"Nicolay's Response: {final_answer}")
        st.session_state.messages.append({"role": "assistant", "content": f"Nicolay's Response: {final_answer}"})

        # Displaying the Search Results
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

        # Ensure lincoln_data is converted to a list of dictionaries if it's a DataFrame
        if isinstance(lincoln_data, pd.DataFrame):
            lincoln_data_list = lincoln_data.to_dict('records')
        else:
            lincoln_data_list = lincoln_data

        if "Match Analysis" in response_json:
            st.markdown(highlight_style, unsafe_allow_html=True)
            for match_key, match_info in response_json["Match Analysis"].items():
                try:
                    if isinstance(match_info, dict):
                        # Ensure all necessary keys are present
                        required_keys = ["Text ID", "Key Quote"]
                        for key in required_keys:
                            if key not in match_info:
                                continue  # Skip to the next match if a required key is missing

                        text_id = match_info.get("Text ID")
                        formatted_text_id = f"Text #: {text_id}"
                        key_quote = match_info.get("Key Quote", "")

                        # Ensure 'text_id' is found in lincoln_data_list
                        speech = next((item for item in lincoln_data_list if item['text_id'] == formatted_text_id), None)

                        # Increment the counter for each match
                        doc_match_counter += 1

                        # Initialize highlight_success for each iteration
                        highlight_success = False  # Flag to track highlighting success

                        if speech:
                            # Use the doc_match_counter in the expander label
                            expander_label = f"**Match {doc_match_counter}**: *{speech['source']}* `{speech['text_id']}`"
                            with st.expander(expander_label, expanded=False):
                                st.markdown(f"**Source:** {speech['source']}")
                                st.markdown(f"**Text ID:** {speech['text_id']}")
                                st.markdown(f"**Summary:**\n{speech.get('summary', '')}")

                                # Replace line breaks for HTML display
                                formatted_full_text = speech['full_text'].replace("\n", "<br>")

                                # Attempt direct highlighting
                                if key_quote and key_quote in speech['full_text']:
                                    formatted_full_text = formatted_full_text.replace(key_quote, f"<mark>{key_quote}</mark>")
                                    highlight_success = True
                                elif key_quote:
                                    # If direct highlighting fails, use regex-based approach
                                    formatted_full_text = highlight_key_quote(speech['full_text'], key_quote)
                                    formatted_full_text = formatted_full_text.replace("\n", "<br>")
                                    # Check if highlighting was successful with regex approach
                                    highlight_success = key_quote in formatted_full_text

                                st.markdown(f"**Key Quote:**\n{key_quote}")
                                st.markdown(f"**Full Text with Highlighted Quote:**", unsafe_allow_html=True)
                                st.markdown(formatted_full_text, unsafe_allow_html=True)

                                # Update highlight_success_dict for the current match
                                highlight_success_dict[match_key] = highlight_success
                        else:
                            with st.expander(f"**Match {doc_match_counter}**: Not Found", expanded=False):
                                st.markdown("Full text not found.")
                                highlight_success_dict[match_key] = False  # Indicate failure as text not found
                    else:
                        continue
                except Exception as e:
                    st.write(f"Error processing match {match_key}: {e}")

        # Log the data
        search_results = results["search_results"]
        semantic_matches = results["semantic_matches"]
        reranked_results = results["reranked_results"]
        model_weighted_keywords = results["model_weighted_keywords"]
        model_year_keywords = results["model_year_keywords"]
        model_text_keywords = results["model_text_keywords"]

        log_keyword_search_results(keyword_results_logger, search_results, prompt, initial_answer, model_weighted_keywords, model_year_keywords, model_text_keywords)
        log_semantic_search_results(semantic_results_logger, semantic_matches, initial_answer)
        log_reranking_results(reranking_results_logger, reranked_results, prompt)
        log_nicolay_model_output(nicolay_data_logger, response_json, prompt, initial_answer, {})

        # Displaying the Analysis Metadata in an expander
        with st.expander("**Analysis Metadata**"):
            # Displaying User Query Analysis
            if "User Query Analysis" in response_json:
                st.markdown("**User Query Analysis:**")
                for key, value in response_json["User Query Analysis"].items():
                    st.markdown(f"- **{key}:** {value}")

            # Displaying Initial Answer Review
            if "Initial Answer Review" in response_json:
                st.markdown("**Initial Answer Review:**")
                for key, value in response_json["Initial Answer Review"].items():
                    st.markdown(f"- **{key}:** {value}")

            # Displaying Match Analysis
            if "Match Analysis" in response_json:
                st.markdown("**Match Analysis:**")
                for match_key, match_info in response_json["Match Analysis"].items():
                    st.markdown(f"- **{match_key}:**")
                    if isinstance(match_info, dict):
                        for key, value in match_info.items():
                            st.markdown(f"  - {key}: {value}")

            # Displaying Meta Analysis
            if "Meta Analysis" in response_json:
                st.markdown("**Meta Analysis:**")
                for key, value in response_json["Meta Analysis"].items():
                    st.markdown(f"- **{key}:** {value}")

            # Displaying Model Feedback
            if "Model Feedback" in response_json:
                st.markdown("**Model Feedback:**")
                for key, value in response_json["Model Feedback"].items():
                    st.markdown(f"- **{key}:** {value}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
