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

# Import the Astra connection handler
from modules.astra_connection import validate_astra_credentials

# Streamlit App Initialization
st.set_page_config(page_title="RAG Benchmarking", layout="wide")
st.title("RAG Benchmarking and Evaluation Module")

# Initialize status container at the top of the app
status_container = st.container()

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

# Check Astra DB environment first
is_valid_env, env_message = validate_astra_credentials()
if not is_valid_env:
    with status_container:
        st.error(f"⚠️ {env_message}")
        st.info("""
        To use this application with Astra DB:
        1. Create an account at datastax.com
        2. Create a database and generate an application token
        3. Add the following to your environment or .streamlit/secrets.toml:
           - ASTRA_DB_ID
           - ASTRA_DB_APPLICATION_TOKEN
        """)
    st.stop()  # Stop execution if credentials are missing

# Initialize ColBERT connection only (skip ingestion)
with status_container:
    with st.spinner("Connecting to Astra DB ColBERT service..."):
        if 'colbert_searcher' not in st.session_state:
            try:
                # Load Lincoln corpus for reference data only
                # (we don't need this for ingestion, just for metadata lookup)
                lincoln_data_df = load_lincoln_speech_corpus()
                lincoln_data = lincoln_data_df.to_dict("records")
                lincoln_dict = {item["text_id"]: item for item in lincoln_data}

                # Initialize with custom stopwords
                #custom_stopwords = {'civil', 'war', 'union', 'confederate'}
                custom_stopwords = {
                # general English function words
                'a', 'an', 'the', 'in', 'on', 'at', 'for', 'to', 'of', 'with', 'by',
                'and', 'or', 'but', 'if', 'then', 'else', 'when', 'up', 'down', 'out',
                'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
                'this', 'that', 'these', 'those', 'there', 'here',

                # Lincoln‑specific
                'abraham', 'lincoln', 'abe', 'president', 'mr', 'mrs',
                'presidential', 'presidency', 'white', 'house'
                }


                # Initialize the searcher WITHOUT ingestion
                colbert_searcher = ColBERTSearcher(
                    lincoln_dict=lincoln_dict,
                    custom_stopwords=custom_stopwords
                )

                # Store in session state
                st.session_state.colbert_searcher = colbert_searcher
                st.session_state.colbert_initialized = True

                # Skip ingestion - assume corpus is already in Astra DB
                st.session_state.corpus_ingested = True

                st.success("✅ Connected to Astra DB ColBERT service successfully")

                # Verify connection with a test query
                with st.spinner("Verifying connection with test query..."):
                    success, message, test_results = colbert_searcher.test_connection("Lincoln")
                    if success:
                        st.success("✅ Test query successful - system ready")
                    else:
                        st.warning(f"⚠️ {message}. Will proceed anyway.")

            except Exception as e:
                import traceback
                st.error(f"❌ Failed to connect to Astra DB: {str(e)}")
                st.error(traceback.format_exc())
                st.session_state.colbert_initialized = False
                st.stop()  # Stop execution if initialization fails
        else:
            colbert_searcher = st.session_state.colbert_searcher
            st.success("✅ Astra DB connection ready")

# Clean, minimal sidebar with system status
with st.sidebar:
    st.header("System Status")

    # Display Astra DB connection status
    if st.session_state.get('colbert_initialized', False):
        st.success("✅ Astra DB ColBERT: Connected")

        # Add a collapsed advanced panel
        # Add a collapsed advanced panel
        with st.expander("Advanced Options", expanded=False):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("Test Connection"):
                    try:
                        with st.spinner("Testing connection..."):
                            success, message, test_results = st.session_state.colbert_searcher.test_connection(
                                "Lincoln presidency"
                            )

                        if success:
                            st.success(f"Connection test successful: {message}")
                            st.dataframe(test_results)
                        else:
                            st.warning(f"Connection test issue: {message}")
                    except Exception as e:
                        st.error(f"Connection test failed: {str(e)}")

            with col2:
                if st.button("Inspect Collection"):
                    try:
                        with st.spinner("Inspecting collection..."):
                            # Define a simple inspection function that works with existing attributes
                            def inspect_collection(searcher):
                                """Inspect the collection using existing attributes"""
                                try:
                                    # Create a simple info object with what we can safely access
                                    info = {
                                        "database_type": type(searcher.database).__name__,
                                        "vector_store_type": type(searcher.vector_store).__name__
                                    }

                                    # Try to get a sample document to understand structure
                                    sample_doc = None
                                    try:
                                        docs = searcher.vector_store.similarity_search("lincoln", k=1)
                                        if docs:
                                            sample_doc = docs[0]
                                    except Exception as e:
                                        info["sample_error"] = str(e)

                                    # If we got a sample document, extract its structure
                                    if sample_doc:
                                        info["document_structure"] = {
                                            "type": type(sample_doc).__name__,
                                            "has_metadata": hasattr(sample_doc, "metadata"),
                                            "metadata_keys": list(sample_doc.metadata.keys()) if hasattr(sample_doc, "metadata") and sample_doc.metadata else [],
                                            "content_preview": sample_doc.page_content[:100] + "..." if hasattr(sample_doc, "page_content") else "No content attribute"
                                        }

                                        # Include sample metadata if available
                                        if hasattr(sample_doc, "metadata") and sample_doc.metadata:
                                            info["metadata_sample"] = sample_doc.metadata

                                    return info
                                except Exception as e:
                                    return {"error": str(e)}

                            # Call the inspection function directly
                            collection_info = inspect_collection(st.session_state.colbert_searcher)
                            st.json(collection_info)  # Display as JSON for easy analysis
                    except Exception as e:
                        st.error(f"Collection inspection failed: {str(e)}")

            with col3:
                if st.button("Check API Parameters"):
                        try:
                            # Check if the API supports multi-field search
                            import inspect
                            method_info = inspect.signature(st.session_state.colbert_searcher.vector_store.similarity_search)
                            method_doc = st.session_state.colbert_searcher.vector_store.similarity_search.__doc__

                            st.write("**Method Signature:**")
                            st.code(str(method_info))

                            st.write("**Method Documentation:**")
                            st.write(method_doc)
                        except Exception as e:
                            st.error(f"Error checking API parameters: {str(e)}")

            # Add this right before using the Test Multi-Field Search button
            if 'search_with_fields' not in dir(st.session_state.colbert_searcher):
                # Define the method dynamically if it doesn't exist
                def search_with_fields_method(self, query, k=5):
                    """Attempt to search across multiple fields"""
                    try:
                        # Preprocess query
                        processed_query = self.preprocess_query(query)

                        # Try different approaches
                        try:
                            # Approach 1: Try fields parameter
                            docs = self.vector_store.similarity_search(
                                query=processed_query,
                                k=k,
                                fields=["full_text", "summary", "keywords"]
                            )
                        except Exception as e1:
                            try:
                                # Approach 2: Try search_parameters
                                docs = self.vector_store.similarity_search(
                                    query=processed_query,
                                    k=k,
                                    search_parameters={"include_fields": ["full_text", "summary", "keywords"]}
                                )
                            except Exception as e2:
                                # Fall back to standard search
                                docs = self.vector_store.similarity_search(
                                    query=processed_query,
                                    k=k
                                )

                        # Process results
                        if not docs:
                            return pd.DataFrame()

                        return self._process_search_results(docs)

                    except Exception as e:
                        import traceback
                        st.error(f"Multi-field search error: {str(e)}")
                        st.error(f"Traceback: {traceback.format_exc()}")
                        return pd.DataFrame()

                # Add the method to the class instance
                import types
                st.session_state.colbert_searcher.search_with_fields = types.MethodType(
                    search_with_fields_method, st.session_state.colbert_searcher
                )



            with col4:
                # Instead of calling search_with_fields
                if st.button("Test Multi-Field Search"):
                    with st.spinner("Testing multi-field search..."):
                        # Try using existing search method with different parameters
                        try:
                            # Try with fields parameter directly
                            processed_query = st.session_state.colbert_searcher.preprocess_query("emancipation")

                            # First try with explicit fields parameter
                            try:
                                docs = st.session_state.colbert_searcher.vector_store.similarity_search(
                                    query=processed_query,
                                    k=3,
                                    fields=["full_text", "summary", "keywords"]
                                )
                                results_df = st.session_state.colbert_searcher._process_search_results(docs)
                                st.success("Fields parameter worked!")
                            except Exception as e1:
                                # Then try with search_parameters
                                try:
                                    docs = st.session_state.colbert_searcher.vector_store.similarity_search(
                                        query=processed_query,
                                        k=3,
                                        search_parameters={"include_fields": ["full_text", "summary", "keywords"]}
                                    )
                                    results_df = st.session_state.colbert_searcher._process_search_results(docs)
                                    st.success("Search parameters worked!")
                                except Exception as e2:
                                    # Fall back to standard search
                                    docs = st.session_state.colbert_searcher.vector_store.similarity_search(
                                        query=processed_query,
                                        k=3
                                    )
                                    results_df = st.session_state.colbert_searcher._process_search_results(docs)
                                    st.info("Used fallback standard search")

                            st.dataframe(results_df)
                        except Exception as e:
                            st.error(f"Multi-field search test failed: {str(e)}")

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

# Set ColBERT implementation info
st.info("Using Astra DB ColBERT implementation")
colbert_impl = "Astra DB ColBERT"  # Set a default value for use in later code

# Use the initialized ColBERT searcher from session state
if st.session_state.get('colbert_initialized', False):
    colbert_searcher = st.session_state.colbert_searcher
else:
    st.error("ColBERT searcher not initialized. Please check the system status.")
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
