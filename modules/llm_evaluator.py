import json
from openai import OpenAI
import streamlit as st

class LLMResponseEvaluator:
    def __init__(self, api_key=None):
        """Initialize evaluator with OpenAI client."""
        self.client = OpenAI(api_key=api_key or st.secrets["openai_api_key"])

    def evaluate_response(self, query, response, source_texts, ideal_docs):
        """Evaluate RAG response using OpenAI."""
        try:
            # Create evaluation prompt
            eval_prompt = self.create_evaluation_prompt(query, response, source_texts, ideal_docs)

            # Make API call to OpenAI
            completion = self.client.chat.completions.create(
                model="gpt-4",  # Using GPT-4 for comprehensive evaluation
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of RAG (Retrieval Augmented Generation) systems, specializing in historical document analysis and academic writing assessment."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent evaluation
                max_tokens=2000
            )

            # Parse response
            try:
                eval_results = json.loads(completion.choices[0].message.content)
                return eval_results
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print("Raw response:", completion.choices[0].message.content)
                return None

        except Exception as e:
            print(f"Error during evaluation: {e}")
            return None

    def create_evaluation_prompt(self, query, response, source_texts, ideal_docs):
        """Create a comprehensive evaluation prompt for the LLM."""
        # Format source texts for prompt
        formatted_sources = "\n\n".join([f"Source {i+1}:\n{text}"
                                       for i, text in enumerate(source_texts)])

        # Define JSON template separately
        json_template = '''{
            "evaluation_scores": {
                "query_response_quality": {"score": X, "examples": [], "suggestions": []},
                "quote_usage": {"score": X, "examples": [], "suggestions": []},
                "citation_accuracy": {"score": X, "examples": [], "suggestions": []},
                "source_integration": {"score": X, "examples": [], "suggestions": []},
                "historical_context": {"score": X, "examples": [], "suggestions": []},
                "response_structure": {"score": X, "examples": [], "suggestions": []}
            },
            "analysis": {
                "hallucinations": [],
                "missed_opportunities": [],
                "suggested_contexts": []
            },
            "overall_assessment": {
                "total_score": X,
                "strengths": [],
                "weaknesses": [],
                "improvement_priorities": []
            }
        }'''

        return f"""
        Evaluate this RAG (Retrieval Augmented Generation) response based on the following information:

        USER QUERY: {query}

        RESPONSE TO EVALUATE:
        {response}

        SOURCE TEXTS USED:
        {formatted_sources}

        EVALUATION CRITERIA:

        1. Query Response Quality (Score 1-5):
        - Does the response directly answer the query?
        - Is the information historically accurate and supported by sources?
        - Is the response comprehensive and well-organized?

        2. Quote Usage & Accuracy (Score 1-5):
        - Are quotes accurately reproduced from source texts?
        - Are the selected quotes relevant and impactful?
        - Do quotes effectively support the response's claims?

        3. Citation Accuracy (Score 1-5):
        - Are all quotes properly cited?
        - Do citations match the correct source documents?
        - Are citations formatted consistently?

        4. Source Integration (Score 1-5):
        - How well are multiple sources synthesized?
        - Is there a logical flow between different sources?
        - Is context preserved when combining sources?

        5. Historical Context (Score 1-5):
        - Is appropriate historical context provided?
        - Are temporal relationships clear?
        - Are historical interpretations accurate?

        6. Response Structure (Score 1-5):
        - Is the response well-organized?
        - Is there a clear progression of ideas?
        - Are transitions between topics smooth?

        Provide your evaluation in the following JSON format exactly:
        {
            "evaluation_scores": {
                "query_response_quality": {"score": X, "examples": [], "suggestions": []},
                "quote_usage": {"score": X, "examples": [], "suggestions": []},
                "citation_accuracy": {"score": X, "examples": [], "suggestions": []},
                "source_integration": {"score": X, "examples": [], "suggestions": []},
                "historical_context": {"score": X, "examples": [], "suggestions": []},
                "response_structure": {"score": X, "examples": [], "suggestions": []}
            },
            "analysis": {
                "hallucinations": [],
                "missed_opportunities": [],
                "suggested_contexts": []
            },
            "overall_assessment": {
                "total_score": X,
                "strengths": [],
                "weaknesses": [],
                "improvement_priorities": []
            }
        }

        Ensure your response is valid JSON and includes specific examples for each criterion.
        """

    def format_evaluation_results(self, eval_results):
        """Format LLM evaluation results for display."""
        if not eval_results:
            return "Error: Unable to generate evaluation results."

        return f"""
        ### RAG Response Evaluation Summary

        #### Overall Assessment
        Total Score: {eval_results['overall_assessment']['total_score']}/30

        **Strengths:**
        {self._format_list(eval_results['overall_assessment']['strengths'])}

        **Areas for Improvement:**
        {self._format_list(eval_results['overall_assessment']['weaknesses'])}

        #### Detailed Scores

        **Query Response Quality:** {eval_results['evaluation_scores']['query_response_quality']['score']}/5
        Examples: {self._format_list(eval_results['evaluation_scores']['query_response_quality']['examples'])}
        Suggestions: {self._format_list(eval_results['evaluation_scores']['query_response_quality']['suggestions'])}

        **Quote Usage:** {eval_results['evaluation_scores']['quote_usage']['score']}/5
        Examples: {self._format_list(eval_results['evaluation_scores']['quote_usage']['examples'])}
        Suggestions: {self._format_list(eval_results['evaluation_scores']['quote_usage']['suggestions'])}

        **Citation Accuracy:** {eval_results['evaluation_scores']['citation_accuracy']['score']}/5
        Examples: {self._format_list(eval_results['evaluation_scores']['citation_accuracy']['examples'])}
        Suggestions: {self._format_list(eval_results['evaluation_scores']['citation_accuracy']['suggestions'])}

        **Source Integration:** {eval_results['evaluation_scores']['source_integration']['score']}/5
        Examples: {self._format_list(eval_results['evaluation_scores']['source_integration']['examples'])}
        Suggestions: {self._format_list(eval_results['evaluation_scores']['source_integration']['suggestions'])}

        **Historical Context:** {eval_results['evaluation_scores']['historical_context']['score']}/5
        Examples: {self._format_list(eval_results['evaluation_scores']['historical_context']['examples'])}
        Suggestions: {self._format_list(eval_results['evaluation_scores']['historical_context']['suggestions'])}

        **Response Structure:** {eval_results['evaluation_scores']['response_structure']['score']}/5
        Examples: {self._format_list(eval_results['evaluation_scores']['response_structure']['examples'])}
        Suggestions: {self._format_list(eval_results['evaluation_scores']['response_structure']['suggestions'])}

        #### Additional Analysis

        **Potential Hallucinations:**
        {self._format_list(eval_results['analysis']['hallucinations'])}

        **Missed Opportunities:**
        {self._format_list(eval_results['analysis']['missed_opportunities'])}

        **Suggested Additional Context:**
        {self._format_list(eval_results['analysis']['suggested_contexts'])}
        """

    def _format_list(self, items):
        """Helper method to format lists for display."""
        if not items:
            return "None identified"
        return "\n".join(f"- {item}" for item in items)
