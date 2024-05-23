import pandas as pd
import msgpack
import streamlit as st

@st.cache_data(persist="disk")
def load_lincoln_speech_corpus():
    with open('/mnt/data/lincoln_speech_corpus.msgpack', 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        data = [unpacked for unpacked in unpacker]
        if not data or not isinstance(data[0], dict):
            raise ValueError("The data structure is not as expected. Ensure the msgpack file is formatted correctly.")
        # Debugging: Print the type and keys of the first item in the data
        st.write(f"Data type: {type(data)}")
        st.write(f"First item type: {type(data[0])}")
        if isinstance(data[0], dict):
            st.write(f"Keys of first item: {list(data[0].keys())}")
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
