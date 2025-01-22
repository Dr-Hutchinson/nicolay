# modules/colbert_search.py

from ragatouille import RAGPretrainedModel
from sentence_transformers import SentenceTransformer
import pandas as pd
import os

class ColBERTSearcher:
    def __init__(self, index_path="data/LincolnCorpus_1"):
        self.index_path = index_path
        self.model = None
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def load_index(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"ColBERT index not found at {self.index_path}")
        self.model = RAGPretrainedModel.from_index(self.index_path)

    def search(self, query, k=5):
        if not self.model:
            self.load_index()

        results = self.model.search(query=query, k=k)
        
        processed_results = []
        for result in results:
            # Add error handling and data validation
            try:
                processed_results.append({
                    "text_id": result.get("pid", result.get("document_id", "Unknown")),  # Handle both possible keys
                    "colbert_score": result.get("score", 0.0),
                    "TopSegment": result.get("content", result.get("passage", "")),  # Handle both possible keys
                    "search_type": "ColBERT"
                })
            except Exception as e:
                continue

        return pd.DataFrame(processed_results)

# End colbert_search.py
