# data_utils.py
# Convert JSON to MessagePack (one-time conversion)
import msgpack
import json
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

with open('data/lincoln_speech_corpus.json', 'r') as file:
    lincoln_data = json.load(file)

with open('data/lincoln_speech_corpus.msgpack', 'wb') as file:
    file.write(msgpack.packb(lincoln_data))

# Convert the loading function to read MessagePack
@st.cache_data(persist="disk")
def load_lincoln_speech_corpus():
    with open('data/lincoln_speech_corpus.msgpack', 'rb') as file:
        return msgpack.unpackb(file.read())

# Similarly, for voyant_word_counts.json
with open('data/voyant_word_counts.json', 'r') as file:
    voyant_data = json.load(file)

with open('data/voyant_word_counts.msgpack', 'wb') as file:
    file.write(msgpack.packb(voyant_data))

@st.cache_data(persist="disk")
def load_voyant_word_counts():
    with open('data/voyant_word_counts.msgpack', 'rb') as file:
        return msgpack.unpackb(file.read())

# Read the CSV file
df = pd.read_csv('lincoln_index_embedded.csv')

# Convert the DataFrame to Parquet format
df.to_parquet('lincoln_index_embedded.parquet')

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

# Load data in __init__ method
self.lincoln_data, self.voyant_data, self.lincoln_index_df = load_all_data()
