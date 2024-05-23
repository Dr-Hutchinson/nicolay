import pandas as pd
import msgpack
import streamlit as st

@st.cache_data(persist="disk")
def load_lincoln_speech_corpus():
    with open('data/lincoln_speech_corpus.msgpack', 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        data = [unpacked for unpacked in unpacker]

        # Ensure data is not empty
        if not data:
            raise ValueError("The data is empty. Ensure the msgpack file is formatted correctly.")

        # The first item is a list of dictionaries
        first_item = data[0]

        # Log the type of first_item
        st.write(f"Type of first item: {type(first_item)}")

        # Verify if it's a list of dictionaries
        if isinstance(first_item, list) and len(first_item) > 0 and isinstance(first_item[0], dict):
            st.write(f"Keys of first dictionary item: {list(first_item[0].keys())}")
        else:
            raise ValueError("The data structure is not as expected. Ensure the msgpack file is formatted correctly.")

        # Extract the dictionaries from the list
        flat_data = first_item
        return flat_data

@st.cache_data(persist="disk")
def load_voyant_word_counts():
    with open('data/voyant_word_counts.msgpack', 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        data = [unpacked for unpacked in unpacker]

        # Ensure data is not empty
        if not data:
            raise ValueError("The data is empty. Ensure the msgpack file is formatted correctly.")

        # The first item should be a dictionary
        first_item = data[0]

        # Log the type and keys of first_item
        st.write(f"Type of first item: {type(first_item)}")
        st.write(f"Keys of first item: {list(first_item.keys())}")

        return first_item

@st.cache_data(persist="disk")
def load_lincoln_index_embedded():
    return pd.read_parquet('data/lincoln_index_embedded.parquet')
