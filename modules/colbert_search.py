# modules/colbert_search.py

from ragatouille import RAGPretrainedModel
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import streamlit as st

class ColBERTSearcher:
    def __init__(self, index_path="data/LincolnCorpus_1", lincoln_dict=None):
        self.index_path = index_path
        self.model = None
        self.lincoln_dict = lincoln_dict or {}
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def load_index(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"ColBERT index not found at {self.index_path}")
        self.model = RAGPretrainedModel.from_index(self.index_path)

    def search(self, query, k=5):
        try:
            results = self.model.search(query=query, k=k)
            processed_results = []
            for result in results:
                # Strip 'Text #: ' and any whitespace
                doc_id = result['document_id'].replace('Text #: ', '').strip()
                lincoln_data = self.lincoln_dict.get(doc_id, {})

                # Add debug print
                st.write(f"Doc ID: {doc_id}, Found in lincoln_dict: {doc_id in self.lincoln_dict}")

                processed_results.append({
                    "text_id": result['document_id'],
                    "colbert_score": float(result['score']),
                    "TopSegment": result['content'],
                    "source": lincoln_data.get('source', ''),
                    "summary": lincoln_data.get('summary', ''),
                    "search_type": "ColBERT"
                })
            return pd.DataFrame(processed_results)
        except Exception as e:
            st.write(f"ColBERT search error: {str(e)}")
            return pd.DataFrame()

# End colbert_search.py
