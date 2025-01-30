import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from typing import List, Set, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer
from ragatouille import RAGPretrainedModel
from modules.data_utils import load_lincoln_speech_corpus

class ColBERTSearcher:
    def __init__(self,
                 index_path: str = "data/LincolnCorpus_1",
                 lincoln_dict: Optional[dict] = None):
        """
        Initialize the enhanced ColBERT searcher with stopword filtering.

        Args:
            index_path: Path to the ColBERT index
            lincoln_dict: Optional dictionary of Lincoln corpus documents
        """
        self.index_path = index_path
        self.model = None
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._initialize_lincoln_dict(lincoln_dict)
        self._initialize_stopwords()
        self._cache = {}

    def _initialize_lincoln_dict(self, lincoln_dict: Optional[dict] = None) -> None:
        """
        Initialize the Lincoln corpus dictionary.

        Args:
            lincoln_dict: Optional pre-loaded dictionary of Lincoln corpus documents
        """
        if lincoln_dict is None:
            lincoln_data_df = load_lincoln_speech_corpus()
            lincoln_data = lincoln_data_df.to_dict("records")
            self.lincoln_dict = {item['text_id']: item for item in lincoln_data}
        else:
            self.lincoln_dict = lincoln_dict

    def _initialize_stopwords(self) -> None:
        """Initialize stopwords including standard NLTK stopwords and domain-specific terms."""
        try:
            nltk.download('stopwords')
            nltk.download('punkt_tab')
        except LookupError:
            nltk.download('stopwords')
            nltk.download('punkt_tab')

        # Get standard English stopwords
        self.stopwords = set(stopwords.words('english'))

        # Add domain-specific stopwords
        domain_stopwords = {"abraham", "lincoln"}
        self.stopwords.update(domain_stopwords)

    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the query by removing stopwords while preserving structure.

        Args:
            query: Original search query

        Returns:
            Preprocessed query with stopwords removed
        """
        # Tokenize the query
        tokens = word_tokenize(query.lower())

        # Remove stopwords while preserving structure
        filtered_tokens = [token for token in tokens if token.lower() not in self.stopwords]

        # Reconstruct the query
        return ' '.join(filtered_tokens)

    def load_index(self):
        """Load the ColBERT index."""
        if not self.model:
            self.model = RAGPretrainedModel.from_index(self.index_path)

    def search(self, query: str, k: int = 5) -> pd.DataFrame:
        """
        Perform ColBERT search with stopword filtering.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            DataFrame containing search results
        """
        # Check cache
        cache_key = f"{query}_{k}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Preprocess query
        processed_query = self._preprocess_query(query)

        try:
            if not self.model:
                self.load_index()

            # Perform search with processed query
            results = self.model.search(query=processed_query, k=k)

            # Process results
            processed_results = self._process_results(results)

            # Cache results
            self._cache[cache_key] = processed_results
            return processed_results

        except Exception as e:
            st.error(f"ColBERT search error: {str(e)}")
            return pd.DataFrame()

    def _process_results(self, results: List) -> pd.DataFrame:
        """Process and format search results."""
        processed_results = []
        for result in results:
            doc_id = f"Text #: {result['document_id'].replace('Text #: ', '')}"
            lincoln_data = self.lincoln_dict.get(doc_id, {})

            processed_results.append({
                "text_id": result['document_id'],
                "score": float(result['score']),
                "key_quote": result['content'],
                "source": lincoln_data.get('source', ''),
                "summary": lincoln_data.get('summary', ''),
                "search_method": "colbert"
            })

        df = pd.DataFrame(processed_results)
