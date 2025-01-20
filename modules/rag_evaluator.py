import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import pandas as pd

class RAGEvaluator:
    def __init__(self):
        """Initialize the RAG evaluator with ROUGE scorer."""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    def prepare_reference_text(self, reranked_results):
        """
        Prepare reference text from reranked results.

        Args:
            reranked_results (pd.DataFrame): DataFrame containing reranked search results
                                           Should have columns: 'Text', 'Score'

        Returns:
            str: Concatenated reference text from top results
        """
        # Get top results and their texts
        top_results = reranked_results.head(3)  # Using top 3 results
        reference_texts = []

        for _, row in top_results.iterrows():
            if 'Text' in row:  # Make sure we have the text column
                reference_texts.append(row['Text'])

        return ' '.join(reference_texts)

    def calculate_bleu(self, reference_text, generated_text):
        """
        Calculate BLEU score between reference and generated text.

        Args:
            reference_text (str): The reference text (from source documents)
            generated_text (str): The generated RAG response

        Returns:
            float: BLEU score
        """
        return sentence_bleu([reference_text.split()], generated_text.split())

    def calculate_rouge(self, reference_text, generated_text):
        """
        Calculate ROUGE scores between reference and generated text.

        Args:
            reference_text (str): The reference text (from source documents)
            generated_text (str): The generated RAG response

        Returns:
            tuple: (ROUGE-1 F1 score, ROUGE-L F1 score)
        """
        scores = self.rouge_scorer.score(reference_text, generated_text)
        return (scores['rouge1'].fmeasure, scores['rougeL'].fmeasure)

    def evaluate_rag_response(self, reranked_results, generated_response):
        """
        Evaluate RAG response using both BLEU and ROUGE metrics.

        Args:
            reranked_results (pd.DataFrame): DataFrame of reranked search results
            generated_response (str): The final generated response

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        reference_text = self.prepare_reference_text(reranked_results)

        bleu_score = self.calculate_bleu(reference_text, generated_response)
        rouge1_score, rougeL_score = self.calculate_rouge(reference_text, generated_response)

        return {
            'bleu_score': bleu_score,
            'rouge1_score': rouge1_score,
            'rougeL_score': rougeL_score,
            'reference_text': reference_text  # Including for potential inspection
        }

def add_evaluator_to_benchmark(evaluator_results):
    """
    Create a formatted string of evaluation results for Streamlit display.

    Args:
        evaluator_results (dict): Results from RAGEvaluator

    Returns:
        str: Formatted evaluation results
    """
    return f"""
    ### RAG Response Evaluation Metrics
    - BLEU Score: {evaluator_results['bleu_score']:.4f}
    - ROUGE-1 Score: {evaluator_results['rouge1_score']:.4f}
    - ROUGE-L Score: {evaluator_results['rougeL_score']:.4f}
    """
