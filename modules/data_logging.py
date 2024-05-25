# data_logging.py

import json
import pandas as pd
from datetime import datetime as dt

class DataLogger:
    def __init__(self, gc, sheet_name):
        self.gc = gc
        self.sheet = self.gc.open(sheet_name).sheet1

    def record_api_outputs(self, data_dict):
        now = dt.now()
        data_dict['Timestamp'] = now  # Add timestamp to the data

        # Convert the data dictionary to a DataFrame
        df = pd.DataFrame([data_dict])

        # Find the next empty row in the sheet to avoid overwriting existing data
        end_row = len(self.sheet.get_all_records()) + 2

        # Append the new data row to the sheet
        self.sheet.set_dataframe(df, (end_row, 1), copy_head=False, extend=True)

def log_keyword_search_results(keyword_results_logger, search_results, user_query, initial_answer, model_weighted_keywords, model_year_keywords, model_text_keywords):
    if isinstance(search_results, pd.DataFrame):
        now = dt.now()  # Current timestamp

        for idx, result in search_results.iterrows():
            record = {
                'Timestamp': now,
                'UserQuery': user_query,
                'initial_Answer': initial_answer,
                'Weighted_Keywoords': model_weighted_keywords,
                'Year_Keywords': model_year_keywords,
                'text_keywords': model_text_keywords,
                'TextID': result['text_id'],
                'KeyQuote': result['quote'],
                'WeightedScore': result['weighted_score'],
                'KeywordCounts': json.dumps(result['keyword_counts'])  # Convert dict to JSON string
            }
            keyword_results_logger.record_api_outputs(record)
    else:
        raise ValueError("search_results should be a DataFrame")

def log_semantic_search_results(semantic_results_logger, semantic_matches, initial_answer):
    if isinstance(semantic_matches, pd.DataFrame):
        now = dt.now()  # Current timestamp

        for idx, row in semantic_matches.iterrows():
            record = {
                'Timestamp': now,
                'UserQuery': row['UserQuery'],
                'HyDE_Query': initial_answer,
                'TextID': row['text_id'],  # Changed to 'text_id' after renaming
                'SimilarityScore': row['similarities'],
                'TopSegment': row['TopSegment']
            }
            semantic_results_logger.record_api_outputs(record)
    else:
        raise ValueError("semantic_matches should be a DataFrame")

def log_reranking_results(reranking_results_logger, reranked_df, user_query):
    if isinstance(reranked_df, pd.DataFrame):
        now = dt.now()  # Current timestamp

        for idx, row in reranked_df.iterrows():
            record = {
                'Timestamp': now,
                'UserQuery': user_query,
                'Rank': row['Rank'],
                'SearchType': row['Search Type'],
                'TextID': row['Text ID'],
                'KeyQuote': row['Key Quote'],
                'Relevance_Score': row['Relevance Score']
            }
            reranking_results_logger.record_api_outputs(record)
    else:
        raise ValueError("reranked_df should be a DataFrame")

def log_nicolay_model_output(nicolay_data_logger, model_output, user_query, initial_answer, highlight_success_dict):
    """
    Logs the final model output from Nicolay to Google Sheets.
    """
    try:
        # If model_output is a string, we should handle it correctly
        if isinstance(model_output, str):
            model_output = json.loads(model_output)

        # Extract key information from model output
        final_answer_text = model_output.get("FinalAnswer", {}).get("Text", "No response available")
        references = ", ".join(model_output.get("FinalAnswer", {}).get("References", []))

        # User query analysis
        query_intent = model_output.get("User Query Analysis", {}).get("Query Intent", "")
        historical_context = model_output.get("User Query Analysis", {}).get("Historical Context", "")

        # Initial answer review
        answer_evaluation = model_output.get("Initial Answer Review", {}).get("Answer Evaluation", "")
        quote_integration = model_output.get("Initial Answer Review", {}).get("Quote Integration Points", "")

        # Response effectiveness and suggestions
        response_effectiveness = model_output.get("Model Feedback", {}).get("Response Effectiveness", "")
        suggestions_for_improvement = model_output.get("Model Feedback", {}).get("Suggestions for Improvement", "")

        # Match analysis - concatenating details of each match into single strings
        match_analysis = model_output.get("Match Analysis", {})
        match_fields = ['Text ID', 'Source', 'Summary', 'Key Quote', 'Historical Context', 'Relevance Assessment']
        match_data = {}

        for match_key, match_details in match_analysis.items():
            match_info = [f"{field}: {match_details.get(field, '')}" for field in match_fields]
            match_data[match_key] = "; ".join(match_info)  # Concatenate with a separator

            if 'speech' in match_details:
                highlight_success_dict[match_key] = match_details['speech']
            else:
                highlight_success_dict[match_key] = False

        # Meta analysis
        meta_strategy = model_output.get("Meta Analysis", {}).get("Strategy for Response Composition", {})
        meta_synthesis = model_output.get("Meta Analysis", {}).get("Synthesis", "")

        # Construct a record for logging
        record = {
            'Timestamp': dt.now(),
            'UserQuery': user_query,
            'initial_Answer': initial_answer,
            'FinalAnswer': final_answer_text,
            'References': references,
            'QueryIntent': query_intent,
            'HistoricalContext': historical_context,
            'AnswerEvaluation': answer_evaluation,
            'QuoteIntegration': quote_integration,
            **match_data,  # Unpack match data into the record
            'MetaStrategy': str(meta_strategy),  # Convert dictionary to string if needed
            'MetaSynthesis': meta_synthesis,
            'ResponseEffectiveness': response_effectiveness,
            'Suggestions': suggestions_for_improvement
        }

        # Add highlight success information for each match
        for match_key, success in highlight_success_dict.items():
            record[f'{match_key}_HighlightSuccess'] = success

        # Log the record
        nicolay_data_logger.record_api_outputs(record)

    except Exception as e:
        st.error(f"Error in logging Nicolay model output: {e}")
