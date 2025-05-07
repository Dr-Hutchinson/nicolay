from typing import List, Set, Optional, Dict, Any
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from nltk.corpus import stopwords
import nltk

# DataStax RAGStack imports
from ragstack_colbert import CassandraDatabase, ColbertEmbeddingModel
from ragstack_langchain.colbert import ColbertVectorStore as LangchainColbertVectorStore

# Load environment variables for Astra DB credentials
load_dotenv()

class DataStaxColBERTSearcher:
    """
    Implements ColBERT searching using DataStax's Astra DB and RAGStack.
    This implementation replaces the local ColBERT index with a cloud-based solution.
    """

    def __init__(self,
                 lincoln_dict: Optional[dict] = None,
                 custom_stopwords: Optional[Set[str]] = None,
                 astra_db_id: Optional[str] = None,
                 astra_db_token: Optional[str] = None,
                 keyspace: str = "default_keyspace"):
        """
        Initialize DataStax ColBERT searcher.

        Args:
            lincoln_dict: Dictionary of Lincoln corpus documents
            custom_stopwords: Set of custom stopwords to add to defaults
            astra_db_id: Astra DB ID (if None, will check st.secrets then env vars)
            astra_db_token: Astra DB application token (if None, will check st.secrets then env vars)
            keyspace: Astra DB keyspace to use
        """
        # Try to get Astra DB credentials from Streamlit secrets first
        if astra_db_id is None and "ASTRA_DB_ID" in st.secrets:
            astra_db_id = st.secrets["ASTRA_DB_ID"]

        if astra_db_token is None and "ASTRA_DB_APPLICATION_TOKEN" in st.secrets:
            astra_db_token = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]

        # Fall back to environment variables if not in secrets
        self.astra_db_id = astra_db_id or os.getenv("ASTRA_DB_ID")
        self.astra_db_token = astra_db_token or os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.keyspace = keyspace

        if not self.astra_db_id or not self.astra_db_token:
            raise ValueError(
                "Astra DB credentials not found. Please provide them via st.secrets, "
                "environment variables, or directly in the constructor."
            )

        # Initialize Lincoln dictionary
        self._initialize_lincoln_dict(lincoln_dict)

        # Initialize stopwords
        self._initialize_stopwords(custom_stopwords)

        # Initialize DataStax components
        self._initialize_datastax_components()

    def _initialize_lincoln_dict(self, lincoln_dict: Optional[Dict[str, Any]]) -> None:
        """
        Initialize lincoln_dict with error handling and validation.

        Args:
            lincoln_dict: Dictionary of Lincoln corpus documents

        Raises:
            RuntimeError: If lincoln_dict initialization fails
            ValueError: If provided lincoln_dict is invalid
        """
        if lincoln_dict is None:
            try:
                from modules.data_utils import load_lincoln_speech_corpus
                lincoln_data_df = load_lincoln_speech_corpus()
                lincoln_data = lincoln_data_df.to_dict("records")
                self.lincoln_dict = {item['text_id']: item for item in lincoln_data}
            except Exception as e:
                raise RuntimeError(f"Failed to initialize lincoln_dict: {str(e)}")
        else:
            # Validate the provided dictionary
            if not all('text_id' in item for item in lincoln_dict.values()):
                raise ValueError("Provided lincoln_dict items must contain 'text_id' field")
            self.lincoln_dict = lincoln_dict

    def _initialize_stopwords(self, custom_stopwords: Optional[Set[str]]) -> None:
        """
        Initialize stopwords for query preprocessing.

        Args:
            custom_stopwords: Additional custom stopwords to include
        """
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            st.warning("NLTK resources not available. Using basic stopwords set.")
            # Fallback to basic stopwords set
            self.stopwords = {
                'a', 'an', 'the', 'in', 'on', 'at', 'for', 'to', 'of', 'with', 'by',
                'and', 'or', 'but', 'if', 'then', 'else', 'when', 'up', 'down', 'out',
                'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
                'this', 'that', 'these', 'those', 'there', 'here'
            }

        # Add Lincoln-specific stopwords
        lincoln_specific_stopwords = {
            'abraham', 'lincoln', 'abe', 'president', 'mr', 'mrs',
            'presidential', 'presidency', 'white', 'house'
        }

        # Add custom stopwords if provided
        if custom_stopwords:
            lincoln_specific_stopwords.update(custom_stopwords)

        self.stopwords.update(lincoln_specific_stopwords)

    def _initialize_datastax_components(self) -> None:
        """
        Initialize DataStax components for ColBERT search.
        """
        try:
            # Connect to Astra DB
            self.database = CassandraDatabase.from_astra(
                astra_token=self.astra_db_token,
                database_id=self.astra_db_id,
                keyspace=self.keyspace
            )

            # Initialize embedding model
            self.embedding_model = ColbertEmbeddingModel()

            # Create vector store
            self.vector_store = LangchainColbertVectorStore(
                database=self.database,
                embedding_model=self.embedding_model,
            )

            st.success("Successfully connected to DataStax Astra DB ColBERT service")
        except Exception as e:
            st.error(f"Failed to initialize DataStax components: {str(e)}")
            raise

    def preprocess_query(self, query: str) -> str:
        """
        Preprocess query by removing stopwords and normalizing text.

        Args:
            query: The search query string

        Returns:
            Preprocessed query string
        """
        try:
            # Simple tokenization by splitting on whitespace if word_tokenize fails
            try:
                tokens = nltk.word_tokenize(query.lower())
            except Exception as e:
                st.write(f"word_tokenize failed: {str(e)}. Using basic split.")
                tokens = query.lower().split()

            # Remove stopwords and normalize
            filtered_tokens = [
                token for token in tokens
                if token not in self.stopwords and token.isalnum()
            ]

            # If the filtered query would be empty, return original query
            if not filtered_tokens:
                return query

            return ' '.join(filtered_tokens)
        except Exception as e:
            st.warning(f"Query preprocessing failed: {str(e)}. Using original query.")
            return query

    def ingest_corpus(self, corpus_texts: List[str], doc_ids: List[str] = None) -> bool:
        """
        Ingest corpus texts into DataStax ColBERT index.

        Args:
            corpus_texts: List of text documents to ingest
            doc_ids: Optional list of document IDs to associate with texts

        Returns:
            True if ingestion was successful, False otherwise
        """
        try:
            if doc_ids is None:
                # Generate sequential IDs if not provided
                doc_ids = [f"doc_{i}" for i in range(len(corpus_texts))]

            if len(corpus_texts) != len(doc_ids):
                raise ValueError("corpus_texts and doc_ids must have the same length")

            # Ingest texts batch by batch to avoid timeouts
            batch_size = 10
            for i in range(0, len(corpus_texts), batch_size):
                batch_texts = corpus_texts[i:i+batch_size]
                batch_ids = doc_ids[i:i+batch_size]

                for text, doc_id in zip(batch_texts, batch_ids):
                    # Add text to vector store with document ID as metadata
                    self.vector_store.add_texts(
                        texts=[text],
                        metadatas=[{"source": doc_id}],
                        doc_id=doc_id
                    )

                st.write(f"Indexed batch {i//batch_size + 1}/{(len(corpus_texts) + batch_size - 1)//batch_size}")

            return True
        except Exception as e:
            st.error(f"Error ingesting corpus: {str(e)}")
            return False

    def search(self, query: str, k: int = 5,
               skip_preprocessing: bool = False) -> pd.DataFrame:
        """
        Perform ColBERT search with processed query.

        Args:
            query: The search query string
            k: Number of results to return
            skip_preprocessing: If True, skip query preprocessing

        Returns:
            DataFrame containing search results
        """
        try:
            # Preprocess query unless explicitly skipped
            processed_query = query if skip_preprocessing else self.preprocess_query(query)

            # Log original and processed queries for debugging
            st.write(f"Original query: {query}")
            st.write(f"Processed query: {processed_query}")

            # Perform search with DataStax ColBERT
            docs = self.vector_store.similarity_search(
                query=processed_query,
                k=k
            )

            # Process results into expected format
            return self._process_search_results(docs)

        except Exception as e:
            st.error(f"DataStax ColBERT search error: {str(e)}")
            return pd.DataFrame()

    def _process_search_results(self, docs) -> pd.DataFrame:
        """
        Process search results from DataStax ColBERT into DataFrame.

        Args:
            docs: List of Document objects from LangChain

        Returns:
            DataFrame with processed results
        """
        processed_results = []

        for doc in docs:
            try:
                # Extract document ID and score from metadata
                doc_id = doc.metadata.get("source", "unknown")
                score = doc.metadata.get("score", 0.0)

                # Retrieve Lincoln data if available
                lincoln_data = self.lincoln_dict.get(doc_id, {})

                processed_results.append({
                    "text_id": doc_id,
                    "colbert_score": float(score),
                    "raw_score": float(score),  # Use same score for consistency
                    "TopSegment": doc.page_content,
                    "source": lincoln_data.get("source", ""),
                    "summary": lincoln_data.get("summary", ""),
                    "search_type": "DataStax_ColBERT",
                    "timestamp": pd.Timestamp.now()
                })
            except Exception as e:
                st.warning(f"Error processing result {doc}: {str(e)}")
                continue

        return pd.DataFrame(processed_results)

    def add_stopwords(self, new_stopwords: Set[str]) -> None:
        """
        Add new stopwords to the existing set.

        Args:
            new_stopwords: Set of new stopwords to add
        """
        self.stopwords.update(new_stopwords)

    def remove_stopwords(self, words_to_remove: Set[str]) -> None:
        """
        Remove words from the stopwords set.

        Args:
            words_to_remove: Set of words to remove from stopwords
        """
        self.stopwords.difference_update(words_to_remove)

    def get_stopwords(self) -> Set[str]:
        """
        Get the current set of stopwords.

        Returns:
            Set of current stopwords
        """
        return self.stopwords.copy()
