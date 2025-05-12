"""
Simplified ColBERT search implementation for pre-existing corpus on DataStax Astra DB.
This assumes the Lincoln corpus is already stored on DataStax's servers.
"""

import os
import time
from typing import List, Dict, Any, Set, Optional, Union
import pandas as pd
import streamlit as st

# DataStax RAGStack imports
from ragstack_colbert import CassandraDatabase, ColbertEmbeddingModel
from ragstack_langchain.colbert import ColbertVectorStore as LangchainColbertVectorStore

class ColBERTSearcher:
    """
    Implements ColBERT searching using DataStax's Astra DB and RAGStack.
    Assumes the corpus is already stored on DataStax's servers.
    """

    def __init__(
        self,
        lincoln_dict: Optional[Dict[str, Any]] = None,
        custom_stopwords: Optional[Set[str]] = None,
        astra_db_id: Optional[str] = None,
        astra_db_token: Optional[str] = None,
        keyspace: str = "default_keyspace",
        collection_name: str = "lincoln_corpus"  # Name of pre-existing collection
    ):
        """
        Initialize DataStax ColBERT searcher for pre-existing corpus.

        Args:
            lincoln_dict: Dictionary of Lincoln corpus documents for result enrichment
            custom_stopwords: Not used in this implementation
            astra_db_id: Astra DB ID
            astra_db_token: Astra DB application token
            keyspace: Astra DB keyspace to use
            collection_name: Name of the pre-existing collection containing the Lincoln corpus
        """
        # Get Astra DB credentials
        self.astra_db_id = astra_db_id or st.secrets.get("ASTRA_DB_ID") or os.getenv("ASTRA_DB_ID")
        self.astra_db_token = astra_db_token or st.secrets.get("ASTRA_DB_APPLICATION_TOKEN") or os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.keyspace = keyspace
        self.collection_name = collection_name

        if not self.astra_db_id or not self.astra_db_token:
            raise ValueError(
                "Astra DB credentials not found. Please provide them via st.secrets, "
                "environment variables, or directly in the constructor."
            )

        # Store lincoln_dict for metadata enrichment
        self.lincoln_dict = lincoln_dict or {}

        # Initialize DataStax components
        self._initialize_datastax_components()

    def _initialize_datastax_components(self) -> None:
        """
        Initialize DataStax components for ColBERT search on pre-existing corpus.
        """
        try:
            st.info(f"Connecting to Astra DB (ID: {self.astra_db_id[:4]}...{self.astra_db_id[-4:]})")

            # Initialize the embedding model (lightweight operation)
            self.embedding_model = ColbertEmbeddingModel()

            # Connect to Astra DB
            self.database = CassandraDatabase.from_astra(
                astra_token=self.astra_db_token,
                database_id=self.astra_db_id,
                keyspace=self.keyspace
            )

            # Create vector store pointing to existing collection
            self.vector_store = LangchainColbertVectorStore(
                database=self.database,
                embedding_model=self.embedding_model,
                collection_name=self.collection_name  # Point to existing collection
            )

            st.success(f"Successfully connected to DataStax Astra DB ColBERT service")
            st.info(f"Using pre-existing corpus collection: '{self.collection_name}'")

            # Test connection with simple query
            try:
                test_docs = self.vector_store.similarity_search("test", k=1)
                st.success(f"Connection test successful - corpus is accessible")
            except Exception as e:
                st.warning(f"Connection established but couldn't access collection: {str(e)}")

        except Exception as e:
            st.error(f"Failed to initialize DataStax components: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            raise

    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Perform ColBERT search on pre-existing corpus using DataStax's API.

        Args:
            query: The search query string
            k: Number of results to return
            filter_metadata: Optional filtering criteria for results

        Returns:
            DataFrame containing search results
        """
        try:
            st.info(f"Executing ColBERT search via Astra DB: '{query}'")

            # Let DataStax handle all the search operations on their server
            docs = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_metadata
            )

            st.success(f"Found {len(docs)} results")

            # Process results into DataFrame format
            return self._process_search_results(docs)

        except Exception as e:
            st.error(f"DataStax ColBERT search error: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def _process_search_results(self, docs) -> pd.DataFrame:
        """
        Process search results from DataStax ColBERT into DataFrame.
        Enriches with metadata from lincoln_dict if available.

        Args:
            docs: List of Document objects from LangChain

        Returns:
            DataFrame with processed results
        """
        processed_results = []

        for i, doc in enumerate(docs):
            try:
                # Extract document ID and score from metadata
                doc_id = doc.metadata.get("source", "unknown")
                score = doc.metadata.get("score", 0.0)

                # Default values
                source = doc.metadata.get("original_source", "")
                summary = doc.metadata.get("summary", "")

                # Try to enrich with lincoln_dict if available
                if doc_id in self.lincoln_dict:
                    lincoln_data = self.lincoln_dict[doc_id]
                    if not source and "source" in lincoln_data:
                        source = lincoln_data["source"]
                    if not summary and "summary" in lincoln_data:
                        summary = lincoln_data["summary"]

                # Extract a key quote from the content
                key_quote = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content

                processed_results.append({
                    "text_id": doc_id,
                    "search_rank": i + 1,
                    "colbert_score": float(score),
                    "raw_score": float(score),
                    "TopSegment": doc.page_content,
                    "Key Quote": key_quote,
                    "source": source,
                    "summary": summary,
                    "search_type": "Astra_ColBERT",
                    "timestamp": pd.Timestamp.now()
                })
            except Exception as e:
                st.warning(f"Error processing result {i+1}: {str(e)}")
                continue

        # Create DataFrame and sort by score
        results_df = pd.DataFrame(processed_results)
        if not results_df.empty:
            results_df = results_df.sort_values(by="colbert_score", ascending=False)

        return results_df

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the ColBERT searcher.

        Returns:
            Dictionary with status information
        """
        return {
            "connected": hasattr(self, "vector_store"),
            "corpus_ingested": True,  # Always true as we're using a pre-existing corpus
            "corpus_collection": self.collection_name,
            "astra_db_id": f"{self.astra_db_id[:4]}...{self.astra_db_id[-4:]}",
        }
