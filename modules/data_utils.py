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
        data = msgpack.unpackb(file.read())
    # Verify data structure
    if not all('full_text' in item and 'source' in item and 'text_id' in item for item in data):
        raise ValueError("Data loaded from msgpack does not have the expected structure.")
    return data


@st.cache_data(persist="disk")
def load_voyant_word_counts():
    with open('data/voyant_word_counts.msgpack', 'rb') as file:
        data = msgpack.unpackb(file.read())
    # Verify data structure
    if 'corpusTerms' not in data or 'terms' not in data['corpusTerms']:
        raise ValueError("Data loaded from msgpack does not have the expected structure.")
    return data

@st.cache_data(persist="disk")
def load_lincoln_index_embedded():
    df = pd.read_parquet('data/lincoln_index_embedded.parquet')
    # Verify data structure
    if 'combined' not in df.columns or 'Unnamed: 0' not in df.columns:
        raise ValueError("Data loaded from parquet does not have the expected structure.")
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'text_id'}, inplace=True)
    return df


lincoln_data = load_lincoln_speech_corpus()
voyant_data = load_voyant_word_counts()
lincoln_index_df = load_lincoln_index_embedded()

# Inspect a few samples
st.write("Sample from lincoln_data:", lincoln_data[:3])
st.write("Sample from voyant_data:", voyant_data['corpusTerms']['terms'][:3])
st.write("Sample from lincoln_index_df:", lincoln_index_df.head(3))


#@st.cache_data(persist="disk")
#def load_all_data():
#    with ThreadPoolExecutor() as executor:
#        lincoln_future = executor.submit(load_lincoln_speech_corpus)
#        voyant_future = executor.submit(load_voyant_word_counts)
#        index_future = executor.submit(load_lincoln_index_embedded)

#        lincoln_data = lincoln_future.result()
#        voyant_data = voyant_future.result()
#        index_data = index_future.result()

#    return lincoln_data, voyant_data, index_data
