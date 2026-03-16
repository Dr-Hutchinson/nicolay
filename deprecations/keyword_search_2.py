# keyword_search.py

import json
import re


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def _norm_source(s: str) -> str:
    """Normalize source strings for robust matching (case/punctuation/Source: prefix)."""
    if not s:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"^source:\s*", "", s)
    s = re.sub(r"[^0-9a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _coerce_keywords_list(text_keywords):
    if not text_keywords:
        return []
    if isinstance(text_keywords, list):
        return [str(k).strip() for k in text_keywords if str(k).strip()]
    # accept comma-separated string
    return [k.strip() for k in str(text_keywords).split(",") if k.strip()]


def find_instances_expanded_search(
    dynamic_weights,
    original_weights,
    data,
    year_keywords=None,
    text_keywords=None,
    top_n=5,
):
    """Keyword search over full_text (plus summary/keywords for scoring), with *soft* year/text constraints.

    Key fixes vs prior version:
    - Text keyword gating is punctuation-insensitive and does NOT use \b boundaries (critical for strings like
      'First Inaugural Address. March 4, 1861.').
    - Year/text constraints are used as *priority buckets* / score boosts, not hard gates that can zero results.
    - Keyword scoring is computed once per keyword (prevents score inflation).
    - Snippet centering uses indices from full_text (prevents bad snippet positions from combined_text matches).
    """

    instances = []

    # Normalize constraint lists
    years_list = list(year_keywords) if year_keywords else []
    text_keywords_list = _coerce_keywords_list(text_keywords)
    text_keywords_norm = [_norm_source(k) for k in text_keywords_list if _norm_source(k)]

    # Normalize weights input shape
    if isinstance(original_weights, dict):
        keyword_list = list(original_weights.keys())
        orig_w = original_weights
    else:
        keyword_list = list(original_weights) if original_weights else []
        orig_w = {k: 1 for k in keyword_list}

    for entry in data:
        if "full_text" not in entry or "source" not in entry:
            continue

        full_text = entry.get("full_text") or ""
        entry_text_lower = str(full_text).lower()

        source_raw = entry.get("source", "")
        source_norm = _norm_source(source_raw)

        summary_lower = str(entry.get("summary", "") or "").lower()
        keywords_lower = " ".join(entry.get("keywords", []) or []).lower()
        combined_text = f"{entry_text_lower} {summary_lower} {keywords_lower}"

        # Constraint matches (for bucketing/boosting)
        match_source_year = (not years_list) or any(str(y) in source_norm for y in years_list)
        match_source_text = (not text_keywords_norm) or any(k in source_norm for k in text_keywords_norm)

        total_score = 0.0
        keyword_counts = {}
        # For snippet centering: (best_original_weight, earliest_position)
        best_center = None

        for keyword in keyword_list:
            kw = str(keyword).lower().strip()
            if not kw:
                continue

            # Keyword occurrence count (score) across combined text
            pat = r"\b" + re.escape(kw) + r"\b"
            count = len(re.findall(pat, combined_text))
            if count <= 0:
                continue

            dynamic_weight = float(dynamic_weights.get(keyword, 0) or 0)
            total_score += count * dynamic_weight
            keyword_counts[keyword] = count

            # Snippet centering uses *full_text* position only
            m = re.search(pat, entry_text_lower)
            if m:
                pos = m.start()
                ow = float(orig_w.get(keyword, 1) or 1)
                if best_center is None:
                    best_center = (ow, pos)
                else:
                    # higher original weight wins; tie-breaker = earlier in text
                    if ow > best_center[0] or (ow == best_center[0] and pos < best_center[1]):
                        best_center = (ow, pos)

        if total_score <= 0:
            continue

        # Boosts (soft constraints)
        if match_source_year:
            total_score += 5.0
        if match_source_text:
            total_score += 8.0

        # Snippet extraction (always from full_text)
        context_length = 300
        center_pos = best_center[1] if best_center else 0
        start_quote = max(0, center_pos - context_length)
        end_quote = min(len(full_text), center_pos + context_length)
        snippet = str(full_text)[start_quote:end_quote].replace("\n", " ")

        instances.append(
            {
                "text_id": entry.get("text_id"),
                "source": source_raw,
                "summary": entry.get("summary", ""),
                "quote": snippet,
                "weighted_score": total_score,
                "keyword_counts": keyword_counts,
                # debug fields (helpful when diagnosing why keyword search is empty)
                "_match_source_year": bool(match_source_year),
                "_match_source_text": bool(match_source_text),
                "_source_norm": source_norm,
            }
        )

    # Priority buckets: both constraints → year-only → everything else
    both = [x for x in instances if x["_match_source_year"] and x["_match_source_text"]]
    year_only = [x for x in instances if x["_match_source_year"] and not x["_match_source_text"]]
    rest = [x for x in instances if not x["_match_source_year"]]

    for bucket in (both, year_only, rest):
        bucket.sort(key=lambda x: x["weighted_score"], reverse=True)

    ordered = both + year_only + rest
    # strip debug keys before returning
    cleaned = []
    for x in ordered[:top_n]:
        x = dict(x)
        x.pop("_match_source_year", None)
        x.pop("_match_source_text", None)
        x.pop("_source_norm", None)
        cleaned.append(x)

    return cleaned


def search_with_dynamic_weights_expanded(user_keywords, corpus_terms, data, year_keywords=None, text_keywords=None, top_n_results=5):
    total_words = sum(term['rawFreq'] for term in corpus_terms['terms'])
    relative_frequencies = {term['term'].lower(): term['rawFreq'] / total_words for term in corpus_terms['terms']}

    # user_keywords can be dict (weights) or list
    if isinstance(user_keywords, dict):
        user_kw_list = list(user_keywords.keys())
        orig_weights = user_keywords
    else:
        user_kw_list = list(user_keywords) if user_keywords else []
        orig_weights = {k: 1 for k in user_kw_list}

    inverse_weights = {keyword: 1 / relative_frequencies.get(str(keyword).lower(), 1) for keyword in user_kw_list}
    max_weight = max(inverse_weights.values()) if inverse_weights else 1
    normalized_weights = {keyword: (weight / max_weight) * 10 for keyword, weight in inverse_weights.items()}

    return find_instances_expanded_search(
        dynamic_weights=normalized_weights,
        original_weights=orig_weights,
        data=data,
        year_keywords=year_keywords,
        text_keywords=text_keywords,
        top_n=top_n_results
    )


def keyword_search(user_query, json_data_path, data_path, year_keywords=None, text_keywords=None, top_n=5):
    keyword_data = load_json(json_data_path)
    speech_data = load_json(data_path)  # Assuming data is stored in JSON
    return search_with_dynamic_weights_expanded(
        user_keywords=user_query,
        corpus_terms=keyword_data,  # Assuming 'corpus_terms' are in keyword_data
        data=speech_data,
        year_keywords=year_keywords,
        text_keywords=text_keywords,
        top_n_results=top_n
    )
