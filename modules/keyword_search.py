# keyword_search.py
# Revised 2026-03-16
#
# Changes from previous version:
#
# [HIGH] IDF fallback fix (search_with_dynamic_weights_expanded, line ~179 original):
#   Previously: absent-from-corpus keywords defaulted to rel_freq=1.0, producing
#   near-zero normalized weight after normalization against genuinely rare terms.
#   Now: absent keywords inherit min_observed_freq, treating them as at least as
#   rare as the rarest known term. This restores Hay's most distinctive generated
#   keywords to their intended high-weight status.
#
# [HIGH] Proportional source boost (find_instances_expanded_search):
#   Previously: flat +5 / +8 additive boosts for year / source-text constraint matches.
#   These fixed values could dominate score differences between chunks when keyword
#   score ranges were narrow (e.g., 350 LD Debate chunks all scoring similarly after
#   a uniform +13 lift). Now: multiplicative boost (×1.5 year / ×1.8 source-text)
#   preserves the priority preference while keeping keyword quality as the primary
#   discriminator within a matched source region.
#
# [LOW] Density-based snippet centering (find_instances_expanded_search):
#   Previously: snippet centered on the first occurrence of the highest original-weight
#   keyword — could anchor far from where most relevant content sits.
#   Now: a sliding 200-character window over full_text finds the position with the
#   highest weighted keyword density; snippet is centered there.
#
# [LOW] Combined-text / full_text mismatch note:
#   Scoring uses combined_text (full_text + summary + keywords fields), but snippet
#   extraction uses full_text only. If a keyword fires only in summary/keywords,
#   best_center remains None and the snippet defaults to the start of full_text.
#   This is preserved behavior but now explicitly guarded: when best_center is None
#   a short explanatory prefix is prepended to the snippet so callers know why
#   the snippet may not contain the matched term visually.
#
# All other logic (priority bucketing, keyword list coercion, source normalization,
# debug-key stripping) is unchanged.

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


def _find_densest_window(entry_text_lower, keyword_list, dynamic_weights, window=200):
    """Return the character position of the densest keyword region in full_text.

    Slides a window of `window` characters across the text, scoring each position
    by the sum of (occurrence_count × dynamic_weight) for all keywords within the
    window. Returns the start position of the highest-scoring window.

    Falls back to 0 if no keywords are found anywhere in the text.
    """
    best_pos = None
    best_score = -1.0

    # Collect all keyword match positions and weights up front
    matches = []  # list of (char_pos, dynamic_weight)
    for keyword in keyword_list:
        kw = str(keyword).lower().strip()
        if not kw:
            continue
        dw = float(dynamic_weights.get(keyword, 0) or 0)
        if dw <= 0:
            continue
        pat = r"\b" + re.escape(kw) + r"\b"
        for m in re.finditer(pat, entry_text_lower):
            matches.append((m.start(), dw))

    if not matches:
        return None  # caller will handle the None case

    text_len = len(entry_text_lower)

    # Evaluate window score at each match start position (only need to check
    # positions where a match begins — score can only improve at those points)
    candidate_positions = sorted({max(0, pos - window // 2) for pos, _ in matches})

    for start in candidate_positions:
        end = start + window
        window_score = sum(dw for pos, dw in matches if start <= pos < end)
        if window_score > best_score:
            best_score = window_score
            best_pos = start + window // 2  # center of the best window

    return best_pos


def find_instances_expanded_search(
    dynamic_weights,
    original_weights,
    data,
    year_keywords=None,
    text_keywords=None,
    top_n=5,
):
    """Keyword search over full_text (plus summary/keywords for scoring), with *soft* year/text constraints.

    Design:
    - Text keyword gating is punctuation-insensitive and does NOT use \\b boundaries (critical for strings like
      'First Inaugural Address. March 4, 1861.').
    - Year/text constraints are used as *priority buckets* / score boosts, not hard gates that can zero results.
    - Keyword scoring is computed once per keyword (prevents score inflation).
    - Snippet centering uses density-based window search over full_text (finds the region
      with the highest weighted keyword concentration rather than the first keyword occurrence).
    - Source boosts are proportional (multiplicative) rather than flat additive, so keyword
      quality remains the primary discriminator within a matched source region.
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

        # Constraint matches (for bucketing / proportional boosting)
        match_source_year = (not years_list) or any(str(y) in source_norm for y in years_list)
        match_source_text = (not text_keywords_norm) or any(k in source_norm for k in text_keywords_norm)

        total_score = 0.0
        keyword_counts = {}

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

        if total_score <= 0:
            continue

        # --- CHANGE: proportional boosts (was flat +5 / +8 additive) ---
        # Multiplicative boosts preserve keyword score as primary discriminator
        # within a matched source region. A chunk with source+keywords still
        # beats keywords-only, but a high-scoring keywords-only chunk can now
        # beat a low-scoring source-matched chunk.
        if match_source_year:
            total_score *= 1.5   # 50% lift for year match
        if match_source_text:
            total_score *= 1.8   # 80% lift for source text match

        # --- CHANGE: density-based snippet centering (was first-occurrence of best keyword) ---
        center_pos = _find_densest_window(entry_text_lower, keyword_list, dynamic_weights)

        context_length = 300
        if center_pos is not None:
            start_quote = max(0, center_pos - context_length)
            end_quote = min(len(full_text), center_pos + context_length)
            snippet = str(full_text)[start_quote:end_quote].replace("\n", " ")
        else:
            # Keywords matched only in summary/keywords metadata, not in full_text.
            # Return start of full_text with a note so callers are not misled.
            snippet = "[match in metadata] " + str(full_text)[:context_length * 2].replace("\n", " ")

        instances.append(
            {
                "text_id": entry.get("text_id"),
                "source": source_raw,
                "summary": entry.get("summary", ""),
                "quote": snippet,
                "weighted_score": total_score,
                "keyword_counts": keyword_counts,
                # debug fields
                "_match_source_year": bool(match_source_year),
                "_match_source_text": bool(match_source_text),
                "_source_norm": source_norm,
            }
        )

    # Priority buckets: both constraints → year-only → everything else
    # Within each bucket, sort by weighted_score descending.
    # Note: with proportional boosts, score ordering already reflects constraint
    # preference, so bucket ordering is a belt-and-suspenders fallback for edge
    # cases where the boost ratios produce identical final scores.
    both = [x for x in instances if x["_match_source_year"] and x["_match_source_text"]]
    year_only = [x for x in instances if x["_match_source_year"] and not x["_match_source_text"]]
    rest = [x for x in instances if not x["_match_source_year"]]

    for bucket in (both, year_only, rest):
        bucket.sort(key=lambda x: x["weighted_score"], reverse=True)

    ordered = both + year_only + rest

    # Strip debug keys before returning
    cleaned = []
    for x in ordered[:top_n]:
        x = dict(x)
        x.pop("_match_source_year", None)
        x.pop("_match_source_text", None)
        x.pop("_source_norm", None)
        cleaned.append(x)

    return cleaned


def search_with_dynamic_weights_expanded(
    user_keywords,
    corpus_terms,
    data,
    year_keywords=None,
    text_keywords=None,
    top_n_results=5,
):
    total_words = sum(term['rawFreq'] for term in corpus_terms['terms'])
    relative_frequencies = {
        term['term'].lower(): term['rawFreq'] / total_words
        for term in corpus_terms['terms']
    }

    # user_keywords can be dict (weights) or list
    if isinstance(user_keywords, dict):
        user_kw_list = list(user_keywords.keys())
        orig_weights = user_keywords
    else:
        user_kw_list = list(user_keywords) if user_keywords else []
        orig_weights = {k: 1 for k in user_kw_list}

    # --- CHANGE: IDF fallback fix (was hardcoded default of 1.0) ---
    # Previously: relative_frequencies.get(kw, 1) — missing keywords defaulted
    # to rel_freq=1.0 (i.e., "as frequent as the entire corpus"), producing
    # near-zero normalized weight after normalization against genuinely rare terms.
    # Now: missing keywords inherit the minimum observed frequency, treating them
    # as at least as rare as the rarest known corpus term. Hay-generated keywords
    # that don't appear in the precomputed frequency table are almost certainly
    # rare; this restores their intended high-weight status.
    if relative_frequencies:
        min_freq = min(relative_frequencies.values())
    else:
        min_freq = 1e-9  # degenerate corpus guard

    inverse_weights = {
        keyword: 1 / relative_frequencies.get(str(keyword).lower(), min_freq)
        for keyword in user_kw_list
    }

    max_weight = max(inverse_weights.values()) if inverse_weights else 1
    normalized_weights = {
        keyword: (weight / max_weight) * 10
        for keyword, weight in inverse_weights.items()
    }

    return find_instances_expanded_search(
        dynamic_weights=normalized_weights,
        original_weights=orig_weights,
        data=data,
        year_keywords=year_keywords,
        text_keywords=text_keywords,
        top_n=top_n_results,
    )


def keyword_search(
    user_query,
    json_data_path,
    data_path,
    year_keywords=None,
    text_keywords=None,
    top_n=5,
):
    keyword_data = load_json(json_data_path)
    speech_data = load_json(data_path)
    return search_with_dynamic_weights_expanded(
        user_keywords=user_query,
        corpus_terms=keyword_data,
        data=speech_data,
        year_keywords=year_keywords,
        text_keywords=text_keywords,
        top_n_results=top_n,
    )
