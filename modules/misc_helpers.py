import pandas as pd
import re

def get_source_and_summary(text_id, lincoln_dict):
    """
    Get the source and summary for a given text_id from the Lincoln dictionary.

    Parameters:
    text_id (int): The ID of the text to search for.
    lincoln_dict (dict): Dictionary containing Lincoln's speeches data.

    Returns:
    tuple: A tuple containing the source and summary.
    """
    # Convert numerical text_id to string format used in JSON
    text_id_str = f"Text #: {text_id}"
    return lincoln_dict.get(text_id_str, {}).get('source'), lincoln_dict.get(text_id_str, {}).get('summary')

def extract_full_text(record):
    """
    Extract the full text from a record string based on a marker.

    Parameters:
    record (str): The record string to extract the text from.

    Returns:
    str: The extracted full text.
    """
    marker = "Full Text:\n"
    if isinstance(record, str):
        marker_index = record.find(marker)
        if marker_index != -1:
            return record[marker_index + len(marker):].strip()
        else:
            return ""
    else:
        return ""

def remove_duplicates(search_results, semantic_matches):
    """
    Remove duplicate entries from combined search results and semantic matches.

    Parameters:
    search_results (DataFrame): DataFrame containing search results.
    semantic_matches (DataFrame): DataFrame containing semantic matches.

    Returns:
    DataFrame: A DataFrame with duplicates removed.
    """
    combined_results = pd.concat([search_results, semantic_matches])
    deduplicated_results = combined_results.drop_duplicates(subset='text_id')
    return deduplicated_results

def get_full_text_by_id(text_id, data):
    """
    Get the full text for a given text_id from the data.

    Parameters:
    text_id (int): The ID of the text to search for.
    data (list): List containing Lincoln's speeches data.

    Returns:
    str: The full text of the specified text_id.
    """
    return next((item['full_text'] for item in data if item['text_id'] == text_id), None)

def highlight_key_quote(text, key_quote):
    """
    Highlight a key quote within a text.

    Parameters:
    text (str): The text to search within.
    key_quote (str): The quote to highlight.

    Returns:
    str: The text with the key quote highlighted.
    """
    parts = key_quote.split("...")
    if len(parts) >= 2:
        pattern = re.escape(parts[0]) + r"\s*.*?\s*" + re.escape(parts[-1]) + r"[.;,]?"
    else:
        pattern = re.escape(key_quote) + r"\s*[.;,]?"
    regex = re.compile(pattern, re.IGNORECASE)
    matches = regex.findall(text)
    for match in matches:
        text = text.replace(match, f"<mark>{match}</mark>")
    return text
