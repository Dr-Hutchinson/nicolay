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

        try:
            results = self.model.search(query=query, k=k)
            processed_results = []
            for result in results:
                # Extract document ID from string format
                doc_id = result['document_id'].replace('Text #: ', '')
                processed_results.append({
                    "text_id": f"Text #: {doc_id}",
                    "colbert_score": float(result['score']),
                    "TopSegment": result['content'],
                    "source": "",  # Add if available
                    "summary": "",  # Add if available
                    "search_type": "ColBERT"
                })
            df = pd.DataFrame(processed_results)
            print(f"Processed ColBERT results: {df.shape}")
            return df
        except Exception as e:
            print(f"ColBERT search error: {str(e)}")
            return pd.DataFrame()

# End colbert_search.py
