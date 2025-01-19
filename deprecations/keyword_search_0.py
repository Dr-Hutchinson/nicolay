import json
import re

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def find_instances_expanded_search(dynamic_weights, original_weights, data, year_keywords=None, text_keywords=None, top_n=5):
    instances = []
    if text_keywords:
        if isinstance(text_keywords, list):
            text_keywords_list = [keyword.strip().lower() for keyword in text_keywords]
        else:
            text_keywords_list = [keyword.strip().lower() for keyword in text_keywords.split(',')]
    else:
        text_keywords_list = []
    for entry in data:
        if 'full_text' in entry and 'source' in entry:
            entry_text_lower = entry['full_text'].lower()
            source_lower = entry['source'].lower()
            summary_lower = entry.get('summary', '').lower()
            keywords_lower = ' '.join(entry.get('keywords', [])).lower()
            match_source_year = not year_keywords or any(str(year) in source_lower for year in year_keywords)
            match_source_text = not text_keywords or any(re.search(r'\b' + re.escape(keyword.lower()) + r'\b', source_lower) for keyword in text_keywords_list)
            if match_source_year and match_source_text:
                total_dynamic_weighted_score = 0
                keyword_counts = {}
                keyword_positions = {}
                combined_text = entry_text_lower + ' ' + summary_lower + ' ' + keywords_lower
                for keyword in original_weights.keys():
                    keyword_lower = keyword.lower()
                    for match in re.finditer(r'\b' + re.escape(keyword_lower) + r'\b', combined_text):
                        count = len(re.findall(r'\b' + re.escape(keyword_lower) + r'\b', combined_text))
                        dynamic_weight = dynamic_weights.get(keyword, 0)
                        if count > 0:
                            keyword_counts[keyword] = count
                            total_dynamic_weighted_score += count * dynamic_weight
                            keyword_index = match.start()
                            original_weight = original_weights[keyword]
                            keyword_positions[keyword_index] = (keyword, original_weight)
                if keyword_positions:
                    highest_original_weighted_position = max(keyword_positions.items(), key=lambda x: x[1][1])[0]
                    context_length = 300
                    start_quote = max(0, highest_original_weighted_position - context_length)
                    end_quote = min(len(entry_text_lower), highest_original_weighted_position + context_length)
                    snippet = entry['full_text'][start_quote:end_quote]
                    instances.append({
                        "text_id": entry['text_id'],
                        "source": entry['source'],
                        "summary": entry.get('summary', ''),
                        "quote": snippet.replace('\n', ' '),
                        "weighted_score": total_dynamic_weighted_score,
                        "keyword_counts": keyword_counts
                    })
    instances.sort(key=lambda x: x['weighted_score'], reverse=True)
    return instances[:top_n]

def search_with_dynamic_weights_expanded(user_keywords, json_data, year_keywords=None, text_keywords=None, top_n_results=5):
    total_words = sum(term['rawFreq'] for term in json_data['corpusTerms']['terms'])
    relative_frequencies = {term['term'].lower(): term['rawFreq'] / total_words for term in json_data['corpusTerms']['terms']}
    inverse_weights = {keyword: 1 / relative_frequencies.get(keyword.lower(), 1) for keyword in user_keywords}
    max_weight = max(inverse_weights.values())
    normalized_weights = {keyword: (weight / max_weight) * 10 for keyword, weight in inverse_weights.items()}
    return find_instances_expanded_search(
        dynamic_weights=normalized_weights,
        original_weights=user_keywords,  # Using user-provided keywords as original weights for snippet centering
        data=json_data,
        year_keywords=year_keywords,
        text_keywords=text_keywords,
        top_n=top_n_results
    )

def keyword_search(user_query, json_data_path, year_keywords=None, text_keywords=None, top_n=5):
    keyword_data = load_json(json_data_path)
    return search_with_dynamic_weights_expanded(
        user_keywords=user_query,
        json_data=keyword_data,
        year_keywords=year_keywords,
        text_keywords=text_keywords,
        top_n_results=top_n
    )
