import streamlit as st
import pandas as pd
import json
from datetime import datetime as dt
import pygsheets
from google.oauth2 import service_account
import logging
import nltk
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Download NLTK data at app startup
try:
    nltk.download('punkt_tab')
    nltk.download('stopwords')
except Exception as e:
    st.warning(f"NLTK resource download failed: {str(e)}. Some features may be limited.")

# --- Our pipeline orchestrator ---
from modules.rag_pipeline import run_rag_pipeline

# Logging Modules
from modules.data_logging import DataLogger, log_benchmark_results

# Import additional modules
from modules.semantic_search import semantic_search
from modules.keyword_search import search_with_dynamic_weights_expanded
from modules.reranking import rerank_results
from modules.prompt_loader import load_prompts
from modules.rag_evaluator import RAGEvaluator, add_evaluator_to_benchmark
from modules.llm_evaluator import LLMEvaluator

# Import DataStax ColBERT implementation (new module)
from modules.colbert_search import ColBERTSearcher

from modules.data_utils import (
    load_lincoln_speech_corpus,
    load_voyant_word_counts,
    load_lincoln_index_embedded
)

# Check for Astra DB credentials in Streamlit secrets or environment variables
astra_db_id = st.secrets.get("ASTRA_DB_ID") if "ASTRA_DB_ID" in st.secrets else os.getenv("ASTRA_DB_ID")
astra_db_token = st.secrets.get("ASTRA_DB_APPLICATION_TOKEN") if "ASTRA_DB_APPLICATION_TOKEN" in st.secrets else os.getenv("ASTRA_DB_APPLICATION_TOKEN")

if not astra_db_id or not astra_db_token:
    st.warning("""
        DataStax Astra DB credentials not found. Please add them to your Streamlit secrets or environment variables:
        - ASTRA_DB_ID: Your Astra DB ID
        - ASTRA_DB_APPLICATION_TOKEN: Your Astra DB application token
    """)

# Streamlit App Initialization
st.set_page_config(page_title="RAG Benchmarking", layout="wide")
st.title("RAG Benchmarking and Evaluation Module")

# Google Sheets Client Initialization
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=["https://www.googleapis.com/auth/drive"]
)
gc = pygsheets.authorize(custom_credentials=credentials)

# Logger Instances
hays_data_logger = DataLogger(gc=gc, sheet_name="hays_data")
keyword_results_logger = DataLogger(gc=gc, sheet_name="keyword_search_results")
nicolay_data_logger = DataLogger(gc=gc, sheet_name="nicolay_data")
reranking_results_logger = DataLogger(gc=gc, sheet_name="reranking_results")
semantic_results_logger = DataLogger(gc=gc, sheet_name="semantic_search_results")
benchmark_logger = DataLogger(gc=gc, sheet_name="benchmark_results")

# Load Prompts into session_state
load_prompts()

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'response_model_system_prompt' not in st.session_state:
    st.session_state['response_model_system_prompt'] = st.session_state.get('response_model_system_prompt', "")
if 'datastax_colbert_initialized' not in st.session_state:
    st.session_state['datastax_colbert_initialized'] = False
if 'corpus_ingested' not in st.session_state:
    st.session_state['corpus_ingested'] = False

# Sidebar for DataStax ColBERT Config
with st.sidebar:
    st.header("DataStax ColBERT Configuration")

    # Display status
    if not astra_db_id or not astra_db_token:
        st.warning("Astra DB credentials not found in Streamlit secrets or environment variables")
    else:
        st.success(f"Astra DB credentials configured: {astra_db_id[:6]}...{astra_db_id[-4:]}")

    # Initialize DataStax ColBERT
    if not st.session_state.datastax_colbert_initialized and astra_db_id and astra_db_token:
        if st.button("Initialize DataStax ColBERT"):
            try:
                # Initialize with custom stopwords
                custom_stopwords = {'civil', 'war', 'union', 'confederate'}

                # Load Lincoln corpus for initialization
                lincoln_data_df = load_lincoln_speech_corpus()
                lincoln_data = lincoln_data_df.to_dict("records")
                lincoln_dict = {item["text_id"]: item for item in lincoln_data}

                # Initialize the searcher - Streamlit secrets handled inside the class
                datastax_colbert_searcher = ColBERTSearcher(
                    lincoln_dict=lincoln_dict,
                    custom_stopwords=custom_stopwords
                )

                # Store in session state
                st.session_state.datastax_colbert_searcher = datastax_colbert_searcher
                st.session_state.datastax_colbert_initialized = True

                st.success("DataStax ColBERT initialized successfully")
            except Exception as e:
                st.error(f"Error initializing DataStax ColBERT: {str(e)}")

    # Ingest corpus if initialized but not ingested
    if st.session_state.datastax_colbert_initialized and not st.session_state.corpus_ingested:
        if st.button("Ingest Lincoln Corpus"):
            try:
                with st.spinner("Ingesting Lincoln corpus into DataStax ColBERT..."):
                    # Load Lincoln corpus
                    lincoln_data_df = load_lincoln_speech_corpus()

                    # Extract texts and IDs
                    corpus_texts = lincoln_data_df["speech_text"].tolist()
                    doc_ids = lincoln_data_df["text_id"].tolist()

                    # Ingest corpus
                    success = st.session_state.datastax_colbert_searcher.ingest_corpus(
                        corpus_texts=corpus_texts,
                        doc_ids=doc_ids
                    )

                    if success:
                        st.session_state.corpus_ingested = True
                        st.success("Lincoln corpus ingested successfully")
                    else:
                        st.error("Failed to ingest Lincoln corpus")
            except Exception as e:
                st.error(f"Error ingesting corpus: {str(e)}")

    if st.session_state.datastax_colbert_initialized and st.session_state.corpus_ingested:
        st.success("DataStax ColBERT ready for search")

# Add query method selection
query_method = st.radio(
    "Select Query Input Method:",
    ["Benchmark Questions", "Custom Query"]
)

# Variable to store query and expected documents
user_query = None
expected_documents = None
question_category = None

# Query Input Section
if query_method == "Benchmark Questions":
    try:
        benchmark_sheet = gc.open("benchmark_questions").sheet1
        benchmark_data = pd.DataFrame(benchmark_sheet.get_all_records())

        # Ensure required columns exist
        required_columns = ['question', 'ideal_documents', 'category']
        missing_columns = [col for col in required_columns if col not in benchmark_data.columns]
        if missing_columns:
            st.error(f"Missing required columns in benchmark data: {missing_columns}")
            benchmark_data = pd.DataFrame()
        else:
            st.success("Benchmark questions loaded successfully.")
    except Exception as e:
        st.error(f"Error loading benchmark questions: {e}")
        benchmark_data = pd.DataFrame()

    # Preprocess the benchmark data
    if not benchmark_data.empty and "ideal_documents" in benchmark_data.columns:
        benchmark_data["ideal_documents"] = benchmark_data["ideal_documents"].apply(
            lambda x: [doc.strip() for doc in str(x).split(",") if doc.strip()]
        )

        # Ensure category is lowercase and standardized
        benchmark_data["category"] = benchmark_data["category"].str.lower().str.replace(" ", "_")

        # Validate categories
        valid_categories = ['factual_retrieval', 'analysis', 'comparative_analysis', 'synthesis']
        invalid_categories = benchmark_data[~benchmark_data["category"].isin(valid_categories)]
        if not invalid_categories.empty:
            st.warning(f"Found invalid categories: {invalid_categories['category'].unique()}")

        # Add dropdown to select a benchmark question
        selected_question_index = st.selectbox(
            "Select a Benchmark Question:",
            options=range(len(benchmark_data)),
            format_func=lambda idx: f"{idx + 1} [{benchmark_data.iloc[idx]['category']}]: {benchmark_data.iloc[idx]['question']}"
        )
        if selected_question_index is not None:
            user_query = benchmark_data.iloc[selected_question_index]["question"]
            expected_documents = benchmark_data.iloc[selected_question_index]["ideal_documents"]
            question_category = benchmark_data.iloc[selected_question_index]["category"]

            # Display selected question details
            st.write(f"**Selected Question Category:** {question_category}")
    else:
        st.warning("No benchmark data available.")

else:
    # Custom query input
    user_query = st.text_input("Enter your query:", "")
    expected_docs_input = st.text_input(
        "Enter expected document IDs (comma-separated):",
        ""
    )
    if expected_docs_input:
        expected_documents = [doc.strip() for doc in expected_docs_input.split(",")]
    else:
        expected_documents = []

    # Add category selection for custom queries
    question_category = st.selectbox(
        "Select Question Category:",
        ['factual_retrieval', 'analysis', 'comparative_analysis', 'synthesis']
    )

# Select ColBERT Implementation
#colbert_impl = st.radio(
#    "Select ColBERT Implementation:",
#    ["DataStax ColBERT", "Local ColBERT"],
#    disabled=not st.session_state.get('datastax_colbert_initialized', False)
#)

st.info("Using Astra DB ColBERT implementation")
colbert_impl = "Astra DB ColBERT"  # Set a default value for use in later code

#if colbert_impl == "DataStax ColBERT" and not st.session_state.get('datastax_colbert_initialized', False):
#    st.warning("DataStax ColBERT is not initialized. Please initialize it in the sidebar.")

# Initialize ColBERT based on selected implementation
# Use the initialized ColBERT searcher
if st.session_state.get('datastax_colbert_initialized', False):
    colbert_searcher = st.session_state.datastax_colbert_searcher
else:
    st.error("ColBERT searcher not initialized. Please initialize it in the sidebar first.")
    colbert_searcher = None

# Evaluation Method Selection
st.subheader("Evaluation Options")
eval_methods = st.multiselect(
    "Select evaluation methods:",
    ["BLEU/ROUGE Evaluation", "LLM-based Evaluation"],
    default=["BLEU/ROUGE Evaluation"]
)

# Process button
if user_query and st.button("Run Evaluation"):
    st.subheader("Processing Query")
    st.write(f"Query: {user_query}")

    try:
        # Load necessary data
        lincoln_data_df = load_lincoln_speech_corpus()
        lincoln_data = lincoln_data_df.to_dict("records")
        lincoln_dict = {item["text_id"]: item for item in lincoln_data}

        # Initialize ColBERT based on selected implementation
        if colbert_impl == "DataStax ColBERT" and st.session_state.get('datastax_colbert_initialized', False):
            colbert_searcher = st.session_state.datastax_colbert_searcher
        else:
            # Use original ColBERT implementation (imported as needed)
            from modules.colbert_search import ColBERTSearcher
            custom_stopwords = {'civil', 'war', 'union', 'confederate'}
            colbert_searcher = ColBERTSearcher(
                lincoln_dict=lincoln_dict,
                custom_stopwords=custom_stopwords
            )

        # --- 1. Execute the RAG Pipeline ---
        pipeline_results = run_rag_pipeline(
            user_query=user_query,
            perform_keyword_search=True,
            perform_semantic_search=True,
            perform_colbert_search=True,
            perform_reranking=True,
            hays_data_logger=hays_data_logger,
            keyword_results_logger=keyword_results_logger,
            nicolay_data_logger=nicolay_data_logger,
            reranking_results_logger=reranking_results_logger,
            semantic_results_logger=semantic_results_logger,
            colbert_searcher=colbert_searcher  # Pass the selected searcher
        )

        # --- 2. Unpack the pipeline results ---
        hay_output = pipeline_results.get("hay_output", {})
        search_results = pipeline_results.get("search_results", pd.DataFrame())
        semantic_matches = pipeline_results.get("semantic_results", pd.DataFrame())
        reranked_results = pipeline_results.get("reranked_results", pd.DataFrame())
        nicolay_output = pipeline_results.get("nicolay_output", {})

        # Extract items from hay_output
        initial_answer = hay_output.get("initial_answer", "")
        weighted_keywords = hay_output.get("weighted_keywords", {})
        year_keywords = hay_output.get("year_keywords", [])
        text_keywords = hay_output.get("text_keywords", [])

        # --- 3. Display the Hay Initial Answer ---
        st.write("### Hay's Initial Answer")
        st.markdown(initial_answer)
        st.write("**Keywords from Hay**:", weighted_keywords)
        st.write("**Year Keywords**:", year_keywords)
        st.write("**Text Keywords**:", text_keywords)

        # --- 4. Display the RAG Search Results ---
        st.write("### Keyword Search Results")
        if not search_results.empty:
            st.dataframe(search_results)
        else:
            st.write("No keyword search results found.")

        st.write("### Semantic Search Results")
        if not semantic_matches.empty:
            st.dataframe(semantic_matches)
        else:
            st.write("No semantic search results found.")

        st.write(f"### {colbert_impl} Search Results")
        colbert_results = pipeline_results.get("colbert_results", pd.DataFrame())
        if not colbert_results.empty:
            st.dataframe(colbert_results)
        else:
            st.write(f"No {colbert_impl} search results found.")

        st.write("### Reranked Results")
        if not reranked_results.empty:
            st.dataframe(reranked_results)
        else:
            st.write("No reranked results found.")

        # Normalized comparison function
        def normalize_doc_id(doc_id):
            """Normalize document IDs by extracting just the numeric portion."""
            if isinstance(doc_id, str) and "Text #:" in doc_id:
                return doc_id.split("Text #:")[1].strip()
            return str(doc_id)

        # --- 6. Display Nicolay's Final Response ---
        final_answer_dict = nicolay_output.get("FinalAnswer", {})
        final_answer_text = final_answer_dict.get("Text", "")

        st.write("### Nicolay's Final Response")
        st.markdown(final_answer_text)

        # If references exist, display them
        references = final_answer_dict.get("References", [])
        if references:
            st.write("**References:**")
            for ref in references:
                st.write(f"- {ref}")

        # --- 5. Compare to Benchmark ---
        st.write("### Benchmark Analysis")
        top_reranked_ids = reranked_results["Text ID"].head(3).tolist() if not reranked_results.empty else []

        # Normalize document IDs before comparison
        normalized_expected = [normalize_doc_id(doc) for doc in expected_documents]
        normalized_reranked = [normalize_doc_id(doc) for doc in top_reranked_ids]

        matching_expected = len(set(normalized_expected) & set(normalized_reranked))
        st.write(f"Expected documents matched in top 3: {matching_expected}/{len(expected_documents)}")

        # --- 7. Run Selected Evaluations ---
        evaluation_results = None
        eval_results = None

        if "BLEU/ROUGE Evaluation" in eval_methods:
            st.subheader("BLEU/ROUGE Evaluation Results")
            evaluator = RAGEvaluator()
            evaluation_results = evaluator.evaluate_rag_response(
                reranked_results=reranked_results,
                generated_response=final_answer_text,
                ideal_documents=expected_documents
            )
            st.markdown(add_evaluator_to_benchmark(evaluation_results))

        if "LLM-based Evaluation" in eval_methods:
            st.subheader("LLM Evaluation Results")
            llm_evaluator = LLMEvaluator()
            eval_results = llm_evaluator.evaluate_response(
                query=user_query,
                response=final_answer_text,
                source_texts=reranked_results['Key Quote'].tolist(),
                ideal_docs=expected_documents,
                category=question_category
            )
            if eval_results:
                st.markdown(llm_evaluator.format_evaluation_results(eval_results))
            else:
                st.error("Unable to generate LLM evaluation results")

        # Log benchmark results only if we have evaluation results
        if evaluation_results or eval_results:
            log_benchmark_results(
                benchmark_logger=benchmark_logger,
                user_query=user_query,
                expected_documents=expected_documents,
                bleu_rouge_results=evaluation_results or {},
                llm_results=eval_results or {},
                reranked_results=reranked_results
            )

    except Exception as e:
        st.error(f"Error processing query: {e}")
        st.exception(e)  # This will show the full traceback

# Summary/Visualization section
st.write("### Summary and Visualization")
st.write("Additional charts or summary metrics can go here.")

# Add instructions for configuring DataStax
st.sidebar.markdown("""
### DataStax Configuration Guide

To use DataStax ColBERT, you need to:

1. Create an Astra DB account at [datastax.com](https://astra.datastax.com/signup)
2. Create a database and generate an application token
3. Add the following to your Streamlit secrets:
   - In local dev: Create `.streamlit/secrets.toml` file with:
     ```toml
     ASTRA_DB_ID = "your-database-id"
     ASTRA_DB_APPLICATION_TOKEN = "your-application-token"
     ```
   - In Streamlit Cloud: Add these values in the app settings
4. Click "Initialize DataStax ColBERT" to connect to Astra DB
5. Click "Ingest Lincoln Corpus" to upload your documents
""")
