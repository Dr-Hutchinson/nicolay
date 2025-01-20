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
        Prepare reference text from reranked results with debug info.
        """
        # Debug: st.write columns available in reranked_results
        st.write("Available columns in reranked_results:", reranked_results.columns.tolist())

        # Get top results and their texts
        top_results = reranked_results.head(3)  # Using top 3 results
        reference_texts = []

        # Debug: st.write each row being processed
        for idx, row in top_results.iterrows():
            st.write(f"\nProcessing result {idx + 1}:")
            st.write("Row content:", row.to_dict())

            # Look for common text column names
            text_column = None
            for possible_name in ['Text', 'text', 'content', 'full_text', 'passage']:
                if possible_name in row:
                    text_column = possible_name
                    break

            if text_column:
                st.write(f"Found text in column: {text_column}")
                reference_texts.append(str(row[text_column]))
            else:
                st.write("No text column found in this row")

        reference_text = ' '.join(reference_texts)
        st.write("\nCombined reference text length:", len(reference_text))
        st.write("First 200 chars of reference text:", reference_text[:200] if reference_text else "No reference text")
        return reference_text

    def calculate_bleu(self, reference_text, generated_text):
        """Calculate BLEU score between reference and generated text."""
        if not reference_text or not generated_text:
            st.write("Warning: Empty reference or generated text in BLEU calculation")
            return 0.0

        reference_tokens = reference_text.split()
        generated_tokens = generated_text.split()

        st.write("\nBLEU Calculation Debug:")
        st.write(f"Reference token count: {len(reference_tokens)}")
        st.write(f"Generated token count: {len(generated_tokens)}")

        return sentence_bleu([reference_tokens], generated_tokens)

    def calculate_rouge(self, reference_text, generated_text):
        """Calculate ROUGE scores between reference and generated text."""
        if not reference_text or not generated_text:
            st.write("Warning: Empty reference or generated text in ROUGE calculation")
            return (0.0, 0.0)

        st.write("\nROUGE Calculation Debug:")
        st.write(f"Reference text length: {len(reference_text)}")
        st.write(f"Generated text length: {len(generated_text)}")

        scores = self.rouge_scorer.score(reference_text, generated_text)
        return (scores['rouge1'].fmeasure, scores['rougeL'].fmeasure)

    def evaluate_rag_response(self, reranked_results, generated_response):
        """Evaluate RAG response using both BLEU and ROUGE metrics."""
        st.write("\n=== Starting RAG Response Evaluation ===")
        st.write(f"Generated response length: {len(str(generated_response))}")
        st.write("First 200 chars of generated response:", str(generated_response)[:200])

        reference_text = self.prepare_reference_text(reranked_results)

        bleu_score = self.calculate_bleu(reference_text, str(generated_response))
        rouge1_score, rougeL_score = self.calculate_rouge(reference_text, str(generated_response))

        return {
            'bleu_score': bleu_score,
            'rouge1_score': rouge1_score,
            'rougeL_score': rougeL_score,
            'reference_text': reference_text,
            'reference_text_length': len(reference_text),
            'generated_text_length': len(str(generated_response))
        }

def add_evaluator_to_benchmark(evaluator_results):
    """Create a formatted string of evaluation results for Streamlit display."""
    return f"""
    ### RAG Response Evaluation Metrics
    - BLEU Score: {evaluator_results['bleu_score']:.4f}
    - ROUGE-1 Score: {evaluator_results['rouge1_score']:.4f}
    - ROUGE-L Score: {evaluator_results['rougeL_score']:.4f}

    Debug Information:
    - Reference text length: {evaluator_results['reference_text_length']} characters
    - Generated text length: {evaluator_results['generated_text_length']} characters
    """
