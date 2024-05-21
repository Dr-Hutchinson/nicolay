import streamlit as st
import openai
import cohere
import pygsheets
from google.oauth2 import service_account

# Set page config first
st.set_page_config(page_title="Nicolay: Exploring the Speeches of Abraham Lincoln with AI (version 0.2)", layout='wide', page_icon='🎩')

# Access secrets
try:
    openai_api_key = st.secrets["openai_key"]
    cohere_api_key = st.secrets["cohere_api_key"]
    gcp_service_account = st.secrets["gcp_service_account"]
except KeyError as e:
    st.error(f"Missing secret: {e}")
    st.stop()

# Initialize OpenAI API key
openai.api_key = openai_api_key

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

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Lincoln speeches – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()

        # Define LLM with a fine-tuned model
        llm = LlamaOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0.5)
        service_context = ServiceContext.from_defaults(llm=llm)

        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        st.write("VectorStoreIndex successfully created")
        return index

index = load_data()

# Initialize chat engine
if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
    st.write("Chat engine initialized")

# Initialize chat messages history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Abraham Lincoln's speeches!"}
    ]

# Prompt for user input and save to chat history
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If the last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history

            # Now call the RAG process to enhance the response
            try:
                results = rag.run_rag_process(prompt)

                # Unpack the results
                initial_answer = results["initial_answer"]
                final_response = json.loads(results["response"])
                search_results = results["search_results"]
                semantic_matches = results["semantic_matches"]
                reranked_results = results["reranked_results"]
                model_weighted_keywords = results["model_weighted_keywords"]
                model_year_keywords = results["model_year_keywords"]
                model_text_keywords = results["model_text_keywords"]

                # Display the RAG process results
                st.markdown("### Initial Answer")
                st.write(initial_answer)

                st.markdown("### Final Answer")
                st.write(final_response['FinalAnswer']['Text'])

                with st.expander("Search Metadata"):
                    st.json(final_response)

                # Log the data
                log_keyword_search_results(keyword_results_logger, search_results, prompt, initial_answer, model_weighted_keywords, model_year_keywords, model_text_keywords)
                log_semantic_search_results(semantic_results_logger, semantic_matches, initial_answer)
                log_reranking_results(reranking_results_logger, reranked_results, prompt)
                log_nicolay_model_output(nicolay_data_logger, final_response, prompt, initial_answer, {})

            except Exception as e:
                st.error(f"An error occurred: {e}")
