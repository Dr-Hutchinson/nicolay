import os
import re
from pathlib import Path
import json
import msgpack
import pandas as pd
import streamlit as st


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
    m = re.search(r"(\d+)", s)
    if not m:
        return s
    return f"Text #: {int(m.group(1))}"


def _extract_full_text_from_combined(combined_val: str) -> str:
    """Extract the chunk's Full Text block from the 'combined' field (best-effort)."""
    if not isinstance(combined_val, str):
        return ""
    # Expected shape:
    # Text #: N
    # Source: ...
    # Full Text:
    # <chunk text>
    # Summary: ...
    m = re.search(r"Full Text:\s*\n(.*?)(?:\n\s*Summary:|\Z)", combined_val, flags=re.S)
    return m.group(1).strip() if m else ""



def _find_first_existing(paths: list[str]) -> str:
    for p in paths:
        if Path(p).exists():
            return p
    return paths[0]


@st.cache_data(persist="disk")
def _df_from_msgpack_bytes(blob: bytes) -> pd.DataFrame:
    """Cache-busted by file content; decodes msgpack and returns a DataFrame."""
    data = msgpack.unpackb(blob, raw=False)
    data = _decode_bytes(data)

    # Some writers wrap the list in a {"data": [...]} object
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        data = data["data"]

    # If the msgpack was written as a stream (multiple top-level objs) and then re-packed,
    # you might end up with an unexpected scalar/bytes. Fail loudly with a helpful message.
    if isinstance(data, (bytes, bytearray)):
        raise ValueError(
            "Corpus msgpack decoded to raw bytes. This usually means the file contains a packed "
            "binary blob (e.g., packed JSON bytes) instead of a list of records."
        )

    if not isinstance(data, list):
        raise ValueError(f"Expected corpus msgpack to decode to a list of records; got {type(data)}")

    return pd.DataFrame(data)


@st.cache_data(persist="disk")
def _df_from_json_bytes(blob: bytes) -> pd.DataFrame:
    data = json.loads(blob.decode("utf-8"))
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Expected corpus JSON to be a list of records; got {type(data)}")
    return pd.DataFrame(data)


@st.cache_data(persist="disk")
def load_lincoln_speech_corpus() -> pd.DataFrame:
    """Loads the reindexed Lincoln speech corpus (886 chunks) as a DataFrame.

    Uses file-content-based caching so updating the underlying msgpack invalidates the cache.
    """
    base = "lincoln_speech_corpus_reindex_keep"
    path = _find_first_existing([
        f"data/{base}.msgpack", f"Data/{base}.msgpack", f"{base}.msgpack",
        f"data/{base}.json",    f"Data/{base}.json",    f"{base}.json",
    ])
    p = Path(path)
    blob = p.read_bytes()

    if p.suffix.lower() == ".json":
        return _df_from_json_bytes(blob)
    return _df_from_msgpack_bytes(blob)


@st.cache_data(persist="disk")
def load_voyant_word_counts() -> pd.DataFrame:
    """Loads Voyant word counts msgpack as a single-row DataFrame."""
    path = _find_first_existing([
        "data/voyant_word_counts.msgpack", "Data/voyant_word_counts.msgpack", "voyant_word_counts.msgpack"
    ])
    blob = Path(path).read_bytes()
    data = msgpack.unpackb(blob, raw=False)
    data = _decode_bytes(data)
    df = pd.DataFrame([data])
    if "corpusTerms" in df.columns:
        df["corpusTerms"] = df["corpusTerms"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df


@st.cache_data(persist="disk")
def _load_lincoln_index_embedded_cached(path: str, mtime: float) -> pd.DataFrame:
    """Load the semantic embedding index parquet and normalize required fields.

    Note: the `mtime` argument intentionally participates in the cache key so updates
    to the parquet file invalidate Streamlit's cache automatically.
    """
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
        "data/lincoln_index_embedded.parquet", "Data/lincoln_index_embedded.parquet", "lincoln_index_embedded.parquet"
    ])
    mtime = os.path.getmtime(path)
    return _load_lincoln_index_embedded_cached(path, mtime)
