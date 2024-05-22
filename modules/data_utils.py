import msgpack
import json
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Convert JSON to MessagePack (one-time conversion)
def convert_json_to_msgpack(json_file, msgpack_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    with open(msgpack_file, 'wb') as file:
        file.write(msgpack.packb(data))

# Convert CSV to Parquet (one-time conversion)
def convert_csv_to_parquet(csv_file, parquet_file):
    df = pd.read_csv(csv_file)
    df.to_parquet(parquet_file)

# One-time conversions (uncomment if needed)
# convert_json_to_msgpack('data/lincoln_speech_corpus.json', 'data/lincoln_speech_corpus.msgpack')
# convert_json_to_msgpack('data/voyant_word_counts.json', 'data/voyant_word_counts.msgpack')
# convert_csv_to_parquet('data/lincoln_index_embedded.csv', 'data/lincoln_index_embedded.parquet')

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
    return pd.read_parquet('data/lincoln_index_embedded.parquet')

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
