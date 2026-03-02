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
    base = "lincoln_speech_corpus_repaired_1"
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


@st.cache_data(persist="disk")
def load_lincoln_index_embedded() -> pd.DataFrame:
    path = _find_first_existing([
        "data/lincoln_index_embedded.parquet", "Data/lincoln_index_embedded.parquet", "lincoln_index_embedded.parquet"
    ])
    df = pd.read_parquet(path)
    # Ensure text_id matches the corpus convention if possible
    if "combined" in df.columns:
        df["text_id"] = df["combined"].str.extract(r"(Text #: \d+)")
    else:
        df = df.reset_index().rename(columns={"index": "text_id"})
    return df


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
