import os
import os
from pathlib import Path
import json
import msgpack
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor


# NOTE: This module is kept for backward compatibility, but it now mirrors the
# repaired-corpus logic used in data_utils.py to avoid mismatched file names and
# stale-cached outputs.

def _decode_bytes(obj):
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8")
        except Exception:
            return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {_decode_bytes(k): _decode_bytes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_bytes(x) for x in obj]
    return obj


def _find_first_existing(paths: list[str]) -> str:
    for p in paths:
        if Path(p).exists():
            return p
    return paths[0]


@st.cache_data(persist="disk")
def _df_from_msgpack_bytes(blob: bytes) -> pd.DataFrame:
    data = msgpack.unpackb(blob, raw=False)
    data = _decode_bytes(data)
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Expected corpus msgpack to decode to a list of records; got {type(data)}")
    return pd.DataFrame(data)


@st.cache_data(persist="disk")
def load_lincoln_speech_corpus() -> pd.DataFrame:
    base = "lincoln_speech_corpus_reindex_keep"
    path = _find_first_existing([
        f"data/{base}.msgpack", f"Data/{base}.msgpack", f"{base}.msgpack",
        f"data/{base}.json",    f"Data/{base}.json",    f"{base}.json",
    ])
    p = Path(path)
    blob = p.read_bytes()
    if p.suffix.lower() == ".json":
        data = json.loads(blob.decode("utf-8"))
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            data = data["data"]
        return pd.DataFrame(data)
    return _df_from_msgpack_bytes(blob)


@st.cache_data(persist="disk")
def load_voyant_word_counts() -> pd.DataFrame:
    path = _find_first_existing([
        "data/voyant_word_counts.msgpack", "Data/voyant_word_counts.msgpack", "voyant_word_counts.msgpack"
    ])
    data = msgpack.unpackb(Path(path).read_bytes(), raw=False)
    data = _decode_bytes(data)
    df = pd.DataFrame([data])
    if "corpusTerms" in df.columns:
        df["corpusTerms"] = df["corpusTerms"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df


def _normalize_text_id(text_id_val):
    """Normalize a text_id value to the canonical 'Text #: N' form when possible."""
    if text_id_val is None:
        return None
    try:
        import pandas as _pd
        if _pd.isna(text_id_val):
            return None
    except Exception:
        pass
    s = str(text_id_val).strip()
    if not s:
        return None
    import re
    m = re.search(r"(\d+)", s)
    if not m:
        return s
    return f"Text #: {int(m.group(1))}"


def _extract_full_text_from_combined(combined_val: str) -> str:
    """Extract the chunk's Full Text block from the 'combined' field (best-effort)."""
    if not isinstance(combined_val, str):
        return ""
    import re
    m = re.search(r"Full Text:\s*\n(.*?)(?:\n\s*Summary:|\Z)", combined_val, flags=re.S)
    return m.group(1).strip() if m else ""


@st.cache_data(persist="disk")
def _load_lincoln_index_embedded_cached(path: str, mtime: float) -> pd.DataFrame:
    """Load the semantic embedding index parquet and normalize required fields.

    The `mtime` argument participates in the cache key so file updates invalidate
    Streamlit's disk cache automatically.
    """
    import os as _os
    df = pd.read_parquet(path)

    # --- text_id normalization ---
    # Prefer an existing text_id column; do NOT overwrite it from `combined` (which may be stale).
    if "text_id" in df.columns:
        df["text_id"] = df["text_id"].apply(_normalize_text_id)
    elif "combined" in df.columns:
        df["text_id"] = df["combined"].str.extract(r"(Text #: \d+)")
    else:
        df = df.reset_index().rename(columns={"index": "text_id"})
        df["text_id"] = df["text_id"].apply(_normalize_text_id)

    # --- full_text extraction (load-time) ---
    # The pipeline's semantic segmentation expects df['full_text'].
    if "full_text" not in df.columns:
        if "combined" in df.columns:
            df["full_text"] = df["combined"].apply(_extract_full_text_from_combined)
        else:
            df["full_text"] = ""

    # Ensure full_text is a string and not NaN
    df["full_text"] = df["full_text"].fillna("").astype(str)

    return df


def load_lincoln_index_embedded() -> pd.DataFrame:
    path = _find_first_existing([
        "data/lincoln_index_embedded_reindex.parquet", "Data/lincoln_index_embedded_reindex.parquet", "lincoln_index_embedded_reindex.parquet",
        "data/lincoln_index_embedded.parquet",         "Data/lincoln_index_embedded.parquet",         "lincoln_index_embedded.parquet",
    ])
    mtime = os.path.getmtime(path)
    return _load_lincoln_index_embedded_cached(path, mtime)


@st.cache_data(persist="disk")
def load_all_data():
    with ThreadPoolExecutor() as executor:
        lincoln_future = executor.submit(load_lincoln_speech_corpus)
        voyant_future = executor.submit(load_voyant_word_counts)
        index_future = executor.submit(load_lincoln_index_embedded)

        lincoln_df = lincoln_future.result()
        voyant_df = voyant_future.result()
        index_df = index_future.result()

    return lincoln_df, voyant_df, index_df
