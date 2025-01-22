# modules/colbert_search.py

from ragatouille import RAGPretrainedModel
from sentence_transformers import SentenceTransformer
import pandas as pd
import os

class ColBERTSearcher:
    def __init__(self, index_path="data/lincoln_colbert_index"):
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
            processed_results.append({
                "text_id": result["document_id"],
                "colbert_score": result["score"],
                "TopSegment": result["text"],  # Using same field name as semantic search
                "search_type": "ColBERT"
            })

        return pd.DataFrame(processed_results)

# End colbert_search.py
