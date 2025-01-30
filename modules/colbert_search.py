from ragatouille import RAGPretrainedModel
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import re
from typing import List, Set, Optional, Dict, Any
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import streamlit as st
from modules.data_utils import load_lincoln_speech_corpus

class ColBERTSearcher:
    def __init__(self,
                 index_path: str = "data/LincolnCorpus_1",
                 lincoln_dict: Optional[dict] = None,
                 custom_stopwords: Optional[Set[str]] = None):
        """
        Initialize ColBERT searcher with custom stopwords support.

        Args:
            index_path: Path to the ColBERT index
            lincoln_dict: Dictionary of Lincoln corpus documents
            custom_stopwords: Set of custom stopwords to add to defaults
        """
        self.index_path = index_path
        self.model = None
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize stopwords
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')

        # Combine NLTK stopwords with custom Lincoln-specific stopwords
        self.stopwords = set(stopwords.words('english'))
        lincoln_specific_stopwords = {
            'abraham', 'lincoln' 
        }

        # Add any additional custom stopwords
        if custom_stopwords:
            lincoln_specific_stopwords.update(custom_stopwords)

        self.stopwords.update(lincoln_specific_stopwords)

        # Initialize Lincoln dictionary
        self._initialize_lincoln_dict(lincoln_dict)

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

    def load_index(self) -> None:
        """
        Load the ColBERT index.

        Raises:
            FileNotFoundError: If index not found at specified path
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"ColBERT index not found at {self.index_path}")
        self.model = RAGPretrainedModel.from_index(self.index_path)

    def preprocess_query(self, query: str) -> str:
        """
        Preprocess query by removing stopwords and normalizing text.

        Args:
            query: The search query string

        Returns:
            Preprocessed query string
        """
        # Tokenize the query
        tokens = word_tokenize(query.lower())

        # Remove stopwords and normalize
        filtered_tokens = [
            token for token in tokens
            if token not in self.stopwords and token.isalnum()
        ]

        # If the filtered query would be empty, return original query
        if not filtered_tokens:
            return query

        return ' '.join(filtered_tokens)

    def search(self, query: str, k: int = 5,
               skip_preprocessing: bool = False) -> pd.DataFrame:
        """
        Perform ColBERT search with preprocessed query.

        Args:
            query: The search query string
            k: Number of results to return
            skip_preprocessing: If True, skip query preprocessing

        Returns:
            DataFrame containing search results
        """
        if not self.model:
            try:
                self.load_index()
            except Exception as e:
                st.error(f"Failed to load ColBERT index: {str(e)}")
                return pd.DataFrame()

        try:
            # Preprocess query unless explicitly skipped
            processed_query = query if skip_preprocessing else self.preprocess_query(query)

            # Log original and processed queries for debugging
            st.debug(f"Original query: {query}")
            st.debug(f"Processed query: {processed_query}")

            # Perform search
            results = self.model.search(query=processed_query, k=k)
            return self._process_search_results(results)

        except Exception as e:
            st.error(f"ColBERT search error: {str(e)}")
            return pd.DataFrame()

    def _process_search_results(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process search results with enhanced error handling.

        Args:
            results: List of search results from ColBERT

        Returns:
            DataFrame containing processed search results
        """
        processed_results = []

        for result in results:
            try:
                doc_id = f"Text #: {result['document_id'].replace('Text #: ', '')}"
                lincoln_data = self.lincoln_dict.get(doc_id, {})

                # Normalize score to [0,1] range for better integration
                normalized_score = float(result['score']) / 100.0 if result['score'] > 0 else 0

                processed_results.append({
                    "text_id": result['document_id'],
                    "colbert_score": normalized_score,
                    "raw_score": float(result['score']),
                    "TopSegment": result['content'],
                    "source": lincoln_data.get('source', ''),
                    "summary": lincoln_data.get('summary', ''),
                    "search_type": "ColBERT",
                    "timestamp": pd.Timestamp.now()
                })
            except Exception as e:
                st.warning(f"Error processing result {result.get('document_id', 'unknown')}: {str(e)}")
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
