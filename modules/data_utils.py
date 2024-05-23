import msgpack
import json
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Load data functions with caching
@st.cache_data(persist="disk")
def load_lincoln_speech_corpus():
    with open('data/lincoln_speech_corpus.msgpack', 'rb') as file:
        return msgpack.unpackb(file.read())

@st.cache_data(persist="disk")
def load_voyant_word_counts():
    with open('data/voyant_word_counts.msgpack', 'rb') as file:
        return msgpack.unpackb(file.read())

@st.cache_data(persist="disk")
def load_lincoln_index_embedded():
    df = pd.read_parquet('data/lincoln_index_embedded.parquet')
    # Extract text_id from combined
    df['text_id'] = df['combined'].str.extract(r'Text #: (\d+)')
    df['text_id'] = df['text_id'].astype(str)
    return df

@st.cache_data(persist="disk")
def load_all_data():
    with ThreadPoolExecutor() as executor:
        lincoln_future = executor.submit(load_lincoln_speech_corpus)
        voyant_future = executor.submit(load_voyant_word_counts)
        index_future = executor.submit(load_lincoln_index_embedded)

        lincoln_data = lincoln_future.result()
        voyant_data = voyant_future.result()
        index_data = index_future.result()

    return lincoln_data, voyant_data, index_data
