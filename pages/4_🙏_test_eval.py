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

# Import DataStax ColBERT implementation
from modules.colbert_search import ColBERTSearcher

from modules.data_utils import (
    load_lincoln_speech_corpus,
    load_voyant_word_counts,
    load_lincoln_index_embedded
)

# Function for DataStax ColBERT initialization
def initialize_datastax_colbert(collection_name="lincoln_corpus"):
    """Initialize DataStax ColBERT connection and return initialization status."""
    try:
        # Check for Astra DB credentials in Streamlit secrets or environment variables
        astra_db_id = st.secrets.get("ASTRA_DB_ID") if "ASTRA_DB_ID" in st.secrets else os.getenv("ASTRA_DB_ID")
        astra_db_token = st.secrets.get("ASTRA_DB_APPLICATION_TOKEN") if "ASTRA_DB_APPLICATION_TOKEN" in st.secrets else os.getenv("ASTRA_DB_APPLICATION_TOKEN")

        if not astra_db_id or not astra_db_token:
            return False, "Astra DB credentials not found in secrets or environment variables"

        # Load Lincoln metadata for result enrichment
        lincoln_data_df = load_lincoln_speech_corpus()
        lincoln_data = lincoln_data_df.to_dict("records")
        lincoln_dict = {item["text_id"]: item for item in lincoln_data}

        # Initialize ColBERT searcher for pre-existing corpus
        datastax_colbert_searcher = ColBERTSearcher(
            lincoln_dict=lincoln_dict,
            astra_db_id=astra_db_id,
            astra_db_token=astra_db_token,
            collection_name=collection_name
        )

        # Store in session state
        st.session_state.datastax_colbert_searcher = datastax_colbert_searcher
        st.session_state.datastax_colbert_initialized = True
        st.session_state.corpus_ingested = True  # Always true for pre-existing corpus
        st.session_state.astra_db_id = astra_db_id
        st.session_state.astra_db_token = astra_db_token

        return True, None
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return False, str(e)

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
if 'collection_name' not in st.session_state:
    st.session_state['collection_name'] = "lincoln_corpus"

# Check for Astra DB credentials in Streamlit secrets or environment variables
astra_db_id = st.secrets.get("ASTRA_DB_ID") if "ASTRA_DB_ID" in st.secrets else os.getenv("ASTRA_DB_ID")
astra_db_token = st.secrets.get("ASTRA_DB_APPLICATION_TOKEN") if "ASTRA_DB_APPLICATION_TOKEN" in st.secrets else os.getenv("ASTRA_DB_APPLICATION_TOKEN")

# Sidebar for status and debugging
with st.sidebar:
    st.header("DataStax ColBERT Status")

    # Collection name input (with stored value persistence)
    collection_name = st.text_input(
        "Collection Name",
        value=st.session_state.get('collection_name', "lincoln_corpus"),
        key="collection_name_input"
    )
    st.session_state.collection_name = collection_name

    # Show connection status
    if st.session_state.get('datastax_colbert_initialized', False):
        st.success("✅ Connected to Astra DB")

        # Display collection details
        if hasattr(st.session_state, 'datastax_colbert_searcher'):
            try:
                status = st.session_state.datastax_colbert_searcher.get_status()
                st.info(f"Using collection: {status.get('corpus_collection', 'unknown')}")
            except:
                st.warning("Could not retrieve collection status")

        # Test search functionality
        with st.expander("Debug Tools"):
            if st.button("Run Test Search"):
                try:
                    test_query = "Lincoln Civil War"
                    with st.spinner(f"Searching for: '{test_query}'"):
                        test_results = st.session_state.datastax_colbert_searcher.search(
                            query=test_query, k=2
                        )

                    if not test_results.empty:
                        st.success(f"Found {len(test_results)} results")
                        st.dataframe(test_results[["text_id", "colbert_score", "Key Quote"]])
                    else:
                        st.warning("No results found")
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")

            # Show credentials (masked)
            st.write("**Astra DB Credentials:**")
            db_id = st.session_state.get('astra_db_id', astra_db_id)
            db_token = st.session_state.get('astra_db_token', astra_db_token)
            if db_id:
                st.write(f"DB ID: {db_id[:4]}...{db_id[-4:]}")
            if db_token:
                st.write(f"Token: {db_token[:4]}...{db_token[-4:]}")
    else:
        st.error("❌ Not connected to Astra DB")

        # Show credentials status
        if not astra_db_id or not astra_db_token:
            st.warning("""
                DataStax Astra DB credentials not found. Please add them to your Streamlit secrets or environment variables:
                - ASTRA_DB_ID: Your Astra DB ID
                - ASTRA_DB_APPLICATION_TOKEN: Your Astra DB application token
            """)
        else:
            st.info(f"Credentials found but not connected")

        # Manual connection button
        if st.button("Connect to Astra DB"):
            with st.spinner("Connecting to Astra DB..."):
                success, error = initialize_datastax_colbert(collection_name)
                if success:
                    st.success("✅ Connected successfully")
                else:
                    st.error(f"❌ Connection failed: {error}")

        # Show help for troubleshooting
        with st.expander("Troubleshooting"):
            st.markdown("""
            **Common issues:**
            - Check that Astra DB credentials are correct
            - Verify that the collection name exists
            - Ensure your Astra DB instance is active
            - Check for network connectivity issues
            """)

    # Add configuration guide
    with st.expander("DataStax Configuration Guide"):
        st.markdown("""
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
        4. Ensure your collection name is correct
        """)

# Automatically initialize DataStax ColBERT if not already initialized
if not st.session_state.datastax_colbert_initialized and astra_db_id and astra_db_token:
    with st.spinner("Connecting to DataStax Astra DB..."):
        success, error = initialize_datastax_colbert(collection_name)
        if success:
            st.success("✅ Connected to DataStax Astra DB successfully")
        else:
            st.warning(f"⚠️ Could not connect automatically: {error}. You can retry manually using the sidebar.")

# Main content area
st.write("## RAG Query Evaluation")

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

# Display information about the search method
st.info("Using Astra DB ColBERT implementation for enhanced retrieval accuracy")

# Evaluation Method Selection
st.subheader("Evaluation Options")
eval_methods = st.multiselect(
    "Select evaluation methods:",
    ["BLEU/ROUGE Evaluation", "LLM-based Evaluation"],
    default=["BLEU/ROUGE Evaluation"]
)

# Process button
if user_query and st.button("Run Evaluation"):
    # Check if DataStax ColBERT is initialized - if not, try to initialize on-demand
    if not st.session_state.get('datastax_colbert_initialized', False):
        st.info("DataStax ColBERT not initialized. Attempting to connect...")
        success, error = initialize_datastax_colbert(collection_name)
        if not success:
            st.error(f"❌ Could not initialize DataStax ColBERT: {error}")
            st.stop()  # Stop execution if initialization fails

    st.subheader("Processing Query")
    st.write(f"Query: '{user_query}'")

    try:
        # Start evaluation process - show a spinner
        with st.spinner("Processing query through the RAG pipeline..."):
            # Get the colbert searcher from session state
            colbert_searcher = st.session_state.datastax_colbert_searcher

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
                colbert_searcher=colbert_searcher  # Pass the DataStax searcher
            )

        # --- 2. Unpack the pipeline results ---
        hay_output = pipeline_results.get("hay_output", {})
        search_results = pipeline_results.get("search_results", pd.DataFrame())
        semantic_matches = pipeline_results.get("semantic_results", pd.DataFrame())
        colbert_results = pipeline_results.get("colbert_results", pd.DataFrame())
        reranked_results = pipeline_results.get("reranked_results", pd.DataFrame())
        nicolay_output = pipeline_results.get("nicolay_output", {})

        # Extract items from hay_output
        initial_answer = hay_output.get("initial_answer", "")
        weighted_keywords = hay_output.get("weighted_keywords", {})
        year_keywords = hay_output.get("year_keywords", [])
        text_keywords = hay_output.get("text_keywords", [])

        # --- 3. Display the results in tabs for better organization ---
        tabs = st.tabs(["Initial Answer", "Search Results", "Final Response", "Benchmark Analysis"])

        with tabs[0]:
            st.write("### Hay's Initial Analysis")
            st.markdown(initial_answer)

            # Display keywords in columns
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Keywords from Hay**")
                st.json(weighted_keywords)
            with col2:
                st.write("**Year Keywords**")
                st.write(year_keywords)
                st.write("**Text Keywords**")
                st.write(text_keywords)

        with tabs[1]:
            st.write("### Search Results")

            # Create subtabs for different search types
            search_tabs = st.tabs(["Keyword Search", "Semantic Search", "ColBERT Search", "Reranked Results"])

            with search_tabs[0]:
                if not search_results.empty:
                    st.dataframe(search_results)
                else:
                    st.write("No keyword search results found.")

            with search_tabs[1]:
                if not semantic_matches.empty:
                    st.dataframe(semantic_matches)
                else:
                    st.write("No semantic search results found.")

            with search_tabs[2]:
                if not colbert_results.empty:
                    st.dataframe(colbert_results)

                    # Display a sample of the top result in an expander
                    if len(colbert_results) > 0:
                        with st.expander("View Top ColBERT Result Text"):
                            top_result = colbert_results.iloc[0]
                            st.markdown(f"**Document ID:** {top_result['text_id']}")
                            st.markdown(f"**Score:** {top_result['colbert_score']:.4f}")
                            st.markdown("**Text Content:**")
                            st.markdown(top_result['TopSegment'])
                else:
                    st.write("No ColBERT search results found.")

            with search_tabs[3]:
                if not reranked_results.empty:
                    st.dataframe(reranked_results)
                else:
                    st.write("No reranked results found.")

        with tabs[2]:
            # --- 4. Display Nicolay's Final Response ---
            final_answer_dict = nicolay_output.get("FinalAnswer", {})
            final_answer_text = final_answer_dict.get("Text", "")

            st.write("### Nicolay's Final Response")
            st.markdown(final_answer_text)

            # If references exist, display them
            references = final_answer_dict.get("References", [])
            if references:
                with st.expander("View References"):
                    for ref in references:
                        st.write(f"- {ref}")

        with tabs[3]:
            # --- 5. Compare to Benchmark ---
            st.write("### Benchmark Analysis")

            # Show expected documents
            st.write("**Expected Documents:**")
            st.write(expected_documents)

            # Get top reranked IDs
            top_reranked_ids = reranked_results["text_id"].head(3).tolist() if not reranked_results.empty else []

            # Normalize document IDs before comparison
            def normalize_doc_id(doc_id):
                """Normalize document IDs by extracting just the numeric portion."""
                if isinstance(doc_id, str) and "Text #:" in doc_id:
                    return doc_id.split("Text #:")[1].strip()
                return str(doc_id)

            normalized_expected = [normalize_doc_id(doc) for doc in expected_documents]
            normalized_reranked = [normalize_doc_id(doc) for doc in top_reranked_ids]

            # Calculate matching metrics
            matching_expected = len(set(normalized_expected) & set(normalized_reranked))
            matching_percentage = matching_expected / len(expected_documents) * 100 if expected_documents else 0

            # Display metrics with color coding
            if matching_percentage >= 75:
                st.success(f"✅ Document match: {matching_expected}/{len(expected_documents)} expected documents found ({matching_percentage:.1f}%)")
            elif matching_percentage >= 50:
                st.warning(f"⚠️ Document match: {matching_expected}/{len(expected_documents)} expected documents found ({matching_percentage:.1f}%)")
            else:
                st.error(f"❌ Document match: {matching_expected}/{len(expected_documents)} expected documents found ({matching_percentage:.1f}%)")

            # Display retrieved vs expected in columns
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Top Retrieved Documents:**")
                for i, doc_id in enumerate(top_reranked_ids):
                    st.write(f"{i+1}. {doc_id}")

            with col2:
                st.write("**Expected Documents:**")
                for i, doc_id in enumerate(expected_documents):
                    st.write(f"{i+1}. {doc_id}")

        # --- 6. Run Selected Evaluations ---
        st.subheader("Detailed Evaluation")
        eval_tabs = st.tabs(["BLEU/ROUGE Evaluation", "LLM-based Evaluation"])

        # BLEU/ROUGE Evaluation
        with eval_tabs[0]:
            if "BLEU/ROUGE Evaluation" in eval_methods:
                with st.spinner("Running BLEU/ROUGE evaluation..."):
                    evaluator = RAGEvaluator()
                    evaluation_results = evaluator.evaluate_rag_response(
                        reranked_results=reranked_results,
                        generated_response=final_answer_text,
                        ideal_documents=expected_documents
                    )
                    st.markdown(add_evaluator_to_benchmark(evaluation_results))
            else:
                st.info("BLEU/ROUGE evaluation not selected")

        # LLM-based Evaluation
        with eval_tabs[1]:
            if "LLM-based Evaluation" in eval_methods:
                with st.spinner("Running LLM-based evaluation..."):
                    llm_evaluator = LLMEvaluator()
                    eval_results = llm_evaluator.evaluate_response(
                        query=user_query,
                        response=final_answer_text,
                        source_texts=reranked_results['Key Quote'].tolist() if not reranked_results.empty else [],
                        ideal_docs=expected_documents,
                        category=question_category
                    )
                    if eval_results:
                        st.markdown(llm_evaluator.format_evaluation_results(eval_results))
                    else:
                        st.error("Unable to generate LLM evaluation results")
            else:
                st.info("LLM-based evaluation not selected")

        # Log benchmark results
        if "BLEU/ROUGE Evaluation" in eval_methods or "LLM-based Evaluation" in eval_methods:
            log_benchmark_results(
                benchmark_logger=benchmark_logger,
                user_query=user_query,
                expected_documents=expected_documents,
                bleu_rouge_results=evaluation_results if "BLEU/ROUGE Evaluation" in eval_methods else {},
                llm_results=eval_results if "LLM-based Evaluation" in eval_methods else {},
                reranked_results=reranked_results
            )
            st.success("Benchmark results logged successfully")

    except Exception as e:
        st.error(f"Error processing query: {e}")
        st.exception(e)  # This will show the full traceback

# Add about section
st.markdown("---")
st.subheader("About Astra DB ColBERT Implementation")
st.markdown("""
This evaluation module uses DataStax's Astra DB ColBERT implementation for enhanced retrieval accuracy.

**How it works:**
- Unlike traditional vector search that creates one vector per document, ColBERT creates vectors for each token in a document
- This allows for more accurate retrieval, especially for unusual terms and proper names
- All processing happens on DataStax's cloud service, minimizing local resource usage
- The implementation follows DataStax's recommended best practices for RAG applications

**Benefits of ColBERT over traditional vector search:**
- Better handling of proper names and rare terms
- Less sensitive to chunking strategies
- Higher accuracy for specific queries
- Better contextual understanding

**Learn more:** [Highly Accurate Retrieval with ColBERT and Astra DB](https://www.datastax.com/blog/highly-accurate-retrieval-for-your-rag-application-with-colbert-and-astra-db)
""")
