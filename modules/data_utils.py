# data_utils.py
import pandas as pd
import msgpack
import streamlit as st

@st.cache_data(persist="disk")
def load_lincoln_speech_corpus():
    with open('/mnt/data/lincoln_speech_corpus.msgpack', 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        data = [unpacked for unpacked in unpacker]
        # Flatten the data: from columns to list of dictionaries
        flat_data = [data[0][str(i)] for i in range(len(data[0]))]
        return flat_data

@st.cache_data(persist="disk")
def load_voyant_word_counts():
    with open('/mnt/data/voyant_word_counts.msgpack', 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        data = [unpacked for unpacked in unpacker]
        return data[0]

@st.cache_data(persist="disk")
def load_lincoln_index_embedded():
    return pd.read_parquet('/mnt/data/lincoln_index_embedded.parquet')
