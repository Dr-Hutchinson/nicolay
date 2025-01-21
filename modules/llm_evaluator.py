import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from openai import OpenAI
import streamlit as st

class LLMEvaluator:
    def __init__(self, api_key=None):
        """Initialize evaluator with OpenAI client."""
        self.client = OpenAI(api_key=api_key or st.secrets["openai_api_key"])
        self.category_prompts = {
            "factual_retrieval": self._get_factual_criteria(),
            "analysis": self._get_analysis_criteria(),
            "comparative_analysis": self._get_comparative_criteria(),
            "synthesis": self._get_synthesis_criteria()
        }

    def evaluate_response(self, query, response, source_texts, ideal_docs, category='factual_retrieval'):
        """Evaluate RAG response using OpenAI with category-specific criteria."""
        try:
            # Get category-specific criteria
            category_criteria = self.category_prompts.get(category, self._get_default_criteria())

            # Create evaluation prompt
            eval_prompt = self.create_evaluation_prompt(
                query=query,
                response=response,
                source_texts=source_texts,
                ideal_docs=ideal_docs,
                category_criteria=category_criteria
            )

            # Make API call to OpenAI
            completion = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of RAG systems, specializing in historical document analysis and academic writing assessment."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            try:
                eval_results = json.loads(completion.choices[0].message.content)
                # Add category to results for reference
                eval_results['category'] = category
                return eval_results
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print("Raw response:", completion.choices[0].message.content)
                return None

        except Exception as e:
            print(f"Error during evaluation: {e}")
            return None

    def create_evaluation_prompt(self, query, response, source_texts, ideal_docs, category_criteria):
        """Create a comprehensive evaluation prompt for the LLM."""
        # Format source texts for prompt
        formatted_sources = "\n\n".join([f"Source {i+1}:\n{text}"
                                       for i, text in enumerate(source_texts)])

        base_criteria = """
        Standard Evaluation Criteria:
        1. Overall Response Quality (Score 1-5):
        - Does the response directly answer the query?
        - Is the information accurate and supported by sources?
        - Is the response well-organized?

        2. Source Usage (Score 1-5):
        - Are quotes accurately reproduced?
        - Are citations properly formatted?
        - Are sources integrated effectively?

        3. Writing Quality (Score 1-5):
        - Is the writing clear and coherent?
        - Are transitions smooth?
        - Is the style appropriate for academic writing?
        """

        json_format = """
        {
            "standard_evaluation": {
                "response_quality": {"score": X, "comments": []},
                "source_usage": {"score": X, "comments": []},
                "writing_quality": {"score": X, "comments": []}
            },
            "category_specific": {
                "criterion1": {"score": X, "comments": []},
                "criterion2": {"score": X, "comments": []},
                "criterion3": {"score": X, "comments": []}
            },
            "overall_assessment": {
                "total_score": X,
                "strengths": [],
                "weaknesses": [],
                "suggestions": []
            }
        }"""

        prompt = f"""
        Evaluate this RAG response based on the following information:

        QUERY: {query}

        RESPONSE TO EVALUATE:
        {response}

        SOURCE TEXTS USED:
        {formatted_sources}

        EVALUATION CRITERIA:
        {base_criteria}

        CATEGORY-SPECIFIC CRITERIA:
        {category_criteria}

        Provide your evaluation in the following JSON format:
        {json_format}

        Ensure your response is valid JSON and includes specific examples and suggestions."""

        return prompt

    def _get_default_criteria(self):
        return """
        1. Basic Historical Analysis (Score 1-5):
        - Is the historical information accurate?
        - Are sources used appropriately?
        - Is context provided where needed?
        """

    def _get_factual_criteria(self):
        return """
        1. Factual Accuracy (Score 1-5):
        - Are all stated facts directly supported by sources?
        - Are dates, numbers, and specific details correct?
        - Is information presented without interpretive bias?

        2. Source Citation (Score 1-5):
        - Are facts properly attributed?
        - Are citations clear and consistent?
        - Is evidence properly quoted?

        3. Completeness (Score 1-5):
        - Are all relevant facts included?
        - Is context provided where necessary?
        - Are there any significant omissions?
        """

    def _get_analysis_criteria(self):
        return """
        1. Analytical Depth (Score 1-5):
        - Is there meaningful interpretation?
        - Are arguments well-supported?
        - Is analysis historically sound?

        2. Evidence Usage (Score 1-5):
        - Are sources effectively analyzed?
        - Is context preserved?
        - Are interpretations justified?

        3. Argument Structure (Score 1-5):
        - Is there a clear thesis?
        - Are points logically developed?
        - Are conclusions well-supported?
        """

    def _get_comparative_criteria(self):
        return """
        1. Comparison Framework (Score 1-5):
        - Is there a clear basis for comparison?
        - Are differences and similarities identified?
        - Is chronological context maintained?

        2. Evidence Balance (Score 1-5):
        - Are comparisons supported by evidence?
        - Is evidence drawn from multiple sources?
        - Are temporal changes documented?

        3. Analytical Synthesis (Score 1-5):
        - Are comparisons meaningful?
        - Are patterns identified?
        - Are conclusions well-reasoned?
        """

    def _get_synthesis_criteria(self):
        return """
        1. Thematic Analysis (Score 1-5):
        - Are themes clearly identified?
        - Is thematic analysis supported?
        - Are themes historically contextualized?

        2. Source Integration (Score 1-5):
        - Are multiple sources synthesized?
        - Is source usage balanced?
        - Are connections well-established?

        3. Interpretative Framework (Score 1-5):
        - Is there a coherent framework?
        - Are conclusions supported?
        - Is complexity acknowledged?
        """

    def format_evaluation_results(self, eval_results):
        """Format evaluation results for display."""
        if not eval_results:
            return "Error: Unable to generate evaluation results."

        standard_eval = eval_results.get('standard_evaluation', {})
        category_eval = eval_results.get('category_specific', {})
        overall = eval_results.get('overall_assessment', {})

        return f"""
        ### RAG Response Evaluation Summary

        #### Standard Evaluation Metrics

        **Response Quality:** {standard_eval.get('response_quality', {}).get('score', 0)}/5
        {self._format_comments(standard_eval.get('response_quality', {}).get('comments', []))}

        **Source Usage:** {standard_eval.get('source_usage', {}).get('score', 0)}/5
        {self._format_comments(standard_eval.get('source_usage', {}).get('comments', []))}

        **Writing Quality:** {standard_eval.get('writing_quality', {}).get('score', 0)}/5
        {self._format_comments(standard_eval.get('writing_quality', {}).get('comments', []))}

        #### Category-Specific Evaluation ({eval_results.get('category', 'default')})

        **Criterion 1:** {category_eval.get('criterion1', {}).get('score', 0)}/5
        {self._format_comments(category_eval.get('criterion1', {}).get('comments', []))}

        **Criterion 2:** {category_eval.get('criterion2', {}).get('score', 0)}/5
        {self._format_comments(category_eval.get('criterion2', {}).get('comments', []))}

        **Criterion 3:** {category_eval.get('criterion3', {}).get('score', 0)}/5
        {self._format_comments(category_eval.get('criterion3', {}).get('comments', []))}

        #### Overall Assessment

        **Total Score:** {overall.get('total_score', 0)}/30

        **Strengths:**
        {self._format_list(overall.get('strengths', []))}

        **Weaknesses:**
        {self._format_list(overall.get('weaknesses', []))}

        **Suggestions for Improvement:**
        {self._format_list(overall.get('suggestions', []))}
        """

    def _format_comments(self, comments):
        """Format comments for display."""
        if not comments:
            return "No comments provided"
        return "\n".join(f"- {comment}" for comment in comments)

    def _format_list(self, items):
        """Helper method to format lists for display."""
        if not items:
            return "None identified"
        return "\n".join(f"- {item}" for item in items)
