import os
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
    """Loads the *repaired* Lincoln speech corpus (772 chunks) as a DataFrame.

    Uses file-content-based caching so updating the underlying msgpack invalidates the cache.
    """
    base = "lincoln_speech_corpus_repaired_1"
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
def load_lincoln_index_embedded() -> pd.DataFrame:
    path = _find_first_existing([
        "data/lincoln_index_embedded.parquet", "Data/lincoln_index_embedded.parquet", "lincoln_index_embedded.parquet"
    ])
    df = pd.read_parquet(path)
    # Normalize text_id to match the corpus convention ("Text #: N")
    if "combined" in df.columns:
        df["text_id"] = df["combined"].str.extract(r"(Text #: \d+)")
    elif "text_id" not in df.columns:
        # Last resort: ensure there is a text_id column, even if it is just the index
        df = df.reset_index().rename(columns={"index": "text_id"})
    return df
