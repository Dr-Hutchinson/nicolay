import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import pandas as pd
import math

class RAGEvaluator:
    def __init__(self):
        """Initialize the RAG evaluator with ROUGE scorer."""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        self.k_values = [1, 3, 5]  # Standard cutoff points for precision/recall

    def evaluate_single_document(self, reference_text, generated_text, doc_id):
        """Evaluate BLEU and ROUGE scores for a single document."""
        bleu = self.calculate_bleu(reference_text, generated_text)
        rouge1, rougeL = self.calculate_rouge(reference_text, generated_text)

        return {
            'doc_id': doc_id,
            'text_length': len(reference_text),
            'bleu_score': bleu,
            'rouge1_score': rouge1,
            'rougeL_score': rougeL
        }

    def normalize_doc_id(self, doc_id):
        """Normalize document IDs by removing 'Text #: ' prefix and stripping whitespace."""
        doc_id = str(doc_id).strip()
        return doc_id.replace('Text #: ', '') if 'Text #: ' in doc_id else doc_id

    def calculate_retrieval_metrics(self, reranked_results, ideal_documents):
        """Calculate retrieval precision metrics."""
        retrieved_docs = reranked_results['Text ID'].tolist()

        # Debug print
        print("\nRetrieval Metrics Debugging:")
        print(f"Retrieved documents (before normalization): {retrieved_docs}")
        print(f"Ideal documents (before normalization): {ideal_documents}")

        # Normalize document IDs for comparison
        retrieved_docs = [self.normalize_doc_id(doc) for doc in retrieved_docs]
        ideal_documents = [self.normalize_doc_id(doc) for doc in ideal_documents]

        print(f"After normalization:")
        print(f"Retrieved documents: {retrieved_docs}")
        print(f"Ideal documents: {ideal_documents}")

        # Calculate MRR
        mrr = self.calculate_mrr(ideal_documents, retrieved_docs)

        # Calculate Precision@k and Recall@k for different k values
        precision_recall = {}
        for k in self.k_values:
            precision_recall[f'P@{k}'] = self.precision_at_k(ideal_documents, retrieved_docs, k)
            precision_recall[f'R@{k}'] = self.recall_at_k(ideal_documents, retrieved_docs, k)

        # Calculate nDCG
        ndcg = self.calculate_ndcg(ideal_documents, retrieved_docs)

        return {
            'mrr': mrr,
            'ndcg': ndcg,
            **precision_recall
        }

    def calculate_mrr(self, ideal_docs, retrieved_docs):
        """Calculate Mean Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in ideal_docs:
                return 1.0 / i
        return 0.0

    def calculate_ndcg(self, ideal_docs, retrieved_docs):
        """Calculate normalized DCG."""
        def dcg(docs, scores):
            return sum(rel / math.log2(i + 2)
                      for i, (doc, rel) in enumerate(zip(docs, scores)))

        relevance = [1 if doc in ideal_docs else 0
                    for doc in retrieved_docs]
        ideal_relevance = sorted(relevance, reverse=True)

        dcg_actual = dcg(retrieved_docs, relevance)
        dcg_ideal = dcg(retrieved_docs, ideal_relevance)

        return dcg_actual / dcg_ideal if dcg_ideal != 0 else 0

    def precision_at_k(self, ideal_docs, retrieved_docs, k):
        """Calculate Precision@k."""
        if not retrieved_docs or k <= 0:
            return 0.0

        top_k = retrieved_docs[:k]
        relevant_in_k = sum(1 for doc in top_k if doc in ideal_docs)
        return relevant_in_k / k

    def recall_at_k(self, ideal_docs, retrieved_docs, k):
        """Calculate Recall@k."""
        if not ideal_docs or not retrieved_docs or k <= 0:
            return 0.0

        top_k = retrieved_docs[:k]
        relevant_in_k = sum(1 for doc in top_k if doc in ideal_docs)
        return relevant_in_k / len(ideal_docs)

    def evaluate_rag_response(self, reranked_results, generated_response, ideal_documents=None):
        """Evaluate RAG response with both content and retrieval metrics."""
        print("\n=== Starting RAG Response Evaluation ===")
        generated_response = str(generated_response)

        # Split results by search method
        if 'Search Type' in reranked_results.columns:
            keyword_results = reranked_results[reranked_results['Search Type'] == 'Keyword'].copy()
            semantic_results = reranked_results[reranked_results['Search Type'] == 'Semantic'].copy()

            print(f"\nResults by search method:")
            print(f"Keyword search results: {len(keyword_results)}")
            print(f"Semantic search results: {len(semantic_results)}")

            # Calculate retrieval metrics for each method
            keyword_metrics = self.calculate_retrieval_metrics(keyword_results, ideal_documents) if not keyword_results.empty else {}
            semantic_metrics = self.calculate_retrieval_metrics(semantic_results, ideal_documents) if not semantic_results.empty else {}
        else:
            keyword_metrics = {}
            semantic_metrics = {}

        # Continue with regular evaluation
        doc_scores = []
        combined_reference_text = []

        # Process each document individually
        for idx, row in reranked_results.head(3).iterrows():
            if 'Key Quote' in row:
                doc_text = str(row['Key Quote'])
                doc_id = str(row['Text ID'])
                search_type = row.get('Search Type', 'Unknown')

                # Calculate individual document scores
                doc_score = self.evaluate_single_document(
                    doc_text,
                    generated_response,
                    doc_id
                )
                doc_score['search_type'] = search_type  # Add search type to scores
                doc_scores.append(doc_score)
                combined_reference_text.append(doc_text)

        # Calculate aggregate content scores
        combined_text = ' '.join(combined_reference_text)
        aggregate_bleu = self.calculate_bleu(combined_text, generated_response)
        aggregate_rouge1, aggregate_rougeL = self.calculate_rouge(combined_text, generated_response)

        # Calculate overall retrieval metrics
        retrieval_metrics = self.calculate_retrieval_metrics(reranked_results, ideal_documents) if ideal_documents is not None else {}

        return {
            'aggregate_scores': {
                'bleu_score': aggregate_bleu,
                'rouge1_score': aggregate_rouge1,
                'rougeL_score': aggregate_rougeL,
                'reference_text_length': len(combined_text),
                'generated_text_length': len(generated_response)
            },
            'individual_scores': doc_scores,
            'retrieval_metrics': retrieval_metrics,
            'search_method_comparison': {
                'keyword': keyword_metrics,
                'semantic': semantic_metrics
            }
        }

    def calculate_bleu(self, reference_text, generated_text):
        """Calculate BLEU score between reference and generated text."""
        if not reference_text or not generated_text:
            return 0.0
        return sentence_bleu([reference_text.split()], generated_text.split())

    def calculate_rouge(self, reference_text, generated_text):
        """Calculate ROUGE scores between reference and generated text."""
        if not reference_text or not generated_text:
            return (0.0, 0.0)
        scores = self.rouge_scorer.score(reference_text, generated_text)
        return (scores['rouge1'].fmeasure, scores['rougeL'].fmeasure)

def add_evaluator_to_benchmark(evaluator_results):
    """Create a formatted string of evaluation results for Streamlit display."""
    aggregate = evaluator_results['aggregate_scores']
    individual = evaluator_results['individual_scores']
    retrieval = evaluator_results.get('retrieval_metrics', {})
    search_comparison = evaluator_results.get('search_method_comparison', {})

    # Format individual document scores
    doc_scores_text = "\n### Individual Document Scores\n"
    for doc in individual:
        doc_scores_text += f"""
        Document {doc['doc_id']} ({doc.get('search_type', 'Unknown')}):
        - BLEU Score: {doc['bleu_score']:.4f}
        - ROUGE-1 Score: {doc['rouge1_score']:.4f}
        - ROUGE-L Score: {doc['rougeL_score']:.4f}
        - Text Length: {doc['text_length']} characters
        """

    # Format retrieval metrics if available
    retrieval_text = ""
    if retrieval:
        retrieval_text = f"""
        ### Overall Retrieval Precision Metrics
        - Mean Reciprocal Rank: {retrieval.get('mrr', 0):.4f}
        - NDCG: {retrieval.get('ndcg', 0):.4f}
        - Precision@1: {retrieval.get('P@1', 0):.4f}
        - Precision@3: {retrieval.get('P@3', 0):.4f}
        - Precision@5: {retrieval.get('P@5', 0):.4f}
        - Recall@1: {retrieval.get('R@1', 0):.4f}
        - Recall@3: {retrieval.get('R@3', 0):.4f}
        - Recall@5: {retrieval.get('R@5', 0):.4f}
        """

    # Format search method comparison
    search_comparison_text = ""
    if search_comparison:
        keyword = search_comparison.get('keyword', {})
        semantic = search_comparison.get('semantic', {})

        search_comparison_text = f"""
        ### Search Method Comparison

        Keyword Search Metrics:
        - MRR: {keyword.get('mrr', 0):.4f}
        - NDCG: {keyword.get('ndcg', 0):.4f}
        - P@3: {keyword.get('P@3', 0):.4f}
        - R@3: {keyword.get('R@3', 0):.4f}

        Semantic Search Metrics:
        - MRR: {semantic.get('mrr', 0):.4f}
        - NDCG: {semantic.get('ndcg', 0):.4f}
        - P@3: {semantic.get('P@3', 0):.4f}
        - R@3: {semantic.get('R@3', 0):.4f}
        """

    return f"""
    ### Aggregate RAG Response Evaluation Metrics
    - BLEU Score: {aggregate['bleu_score']:.4f}
    - ROUGE-1 Score: {aggregate['rouge1_score']:.4f}
    - ROUGE-L Score: {aggregate['rougeL_score']:.4f}

    Reference text length: {aggregate['reference_text_length']} characters
    Generated text length: {aggregate['generated_text_length']} characters

    {doc_scores_text}
    {retrieval_text}
    {search_comparison_text}
    """
