import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import pandas as pd

class RAGEvaluator:
    def __init__(self):
        """Initialize the RAG evaluator with ROUGE scorer."""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    def evaluate_single_document(self, reference_text, generated_text, doc_id):
        """
        Evaluate BLEU and ROUGE scores for a single document.

        Args:
            reference_text (str): Text from single retrieved document
            generated_text (str): Generated RAG response
            doc_id (str): Document identifier

        Returns:
            dict: Scores for this document
        """
        bleu = self.calculate_bleu(reference_text, generated_text)
        rouge1, rougeL = self.calculate_rouge(reference_text, generated_text)

        return {
            'doc_id': doc_id,
            'text_length': len(reference_text),
            'bleu_score': bleu,
            'rouge1_score': rouge1,
            'rougeL_score': rougeL
        }

    def evaluate_rag_response(self, reranked_results, generated_response):
        """
        Evaluate RAG response with both aggregate and per-document scores.
        """
        print("\n=== Starting RAG Response Evaluation ===")
        generated_response = str(generated_response)
        print(f"Generated response length: {len(generated_response)}")

        # Store individual document scores
        doc_scores = []
        combined_reference_text = []

        # Process each document individually
        for idx, row in reranked_results.head(3).iterrows():
            if 'Key Quote' in row:
                doc_text = str(row['Key Quote'])
                doc_id = str(row['Text ID'])

                # Calculate individual document scores
                doc_score = self.evaluate_single_document(
                    doc_text,
                    generated_response,
                    doc_id
                )
                doc_scores.append(doc_score)
                combined_reference_text.append(doc_text)

        # Calculate aggregate scores
        combined_text = ' '.join(combined_reference_text)
        aggregate_bleu = self.calculate_bleu(combined_text, generated_response)
        aggregate_rouge1, aggregate_rougeL = self.calculate_rouge(combined_text, generated_response)

        return {
            'aggregate_scores': {
                'bleu_score': aggregate_bleu,
                'rouge1_score': aggregate_rouge1,
                'rougeL_score': aggregate_rougeL,
                'reference_text_length': len(combined_text),
                'generated_text_length': len(generated_response)
            },
            'individual_scores': doc_scores
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

    # Format individual document scores
    doc_scores_text = "\n### Individual Document Scores\n"
    for doc in individual:
        doc_scores_text += f"""
        Document {doc['doc_id']}:
        - BLEU Score: {doc['bleu_score']:.4f}
        - ROUGE-1 Score: {doc['rouge1_score']:.4f}
        - ROUGE-L Score: {doc['rougeL_score']:.4f}
        - Text Length: {doc['text_length']} characters
        """

    return f"""
    ### Aggregate RAG Response Evaluation Metrics
    - BLEU Score: {aggregate['bleu_score']:.4f}
    - ROUGE-1 Score: {aggregate['rouge1_score']:.4f}
    - ROUGE-L Score: {aggregate['rougeL_score']:.4f}

    Reference text length: {aggregate['reference_text_length']} characters
    Generated text length: {aggregate['generated_text_length']} characters

    {doc_scores_text}
    """
