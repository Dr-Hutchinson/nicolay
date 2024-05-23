import pandas as pd
import msgpack
import streamlit as st
import json

@st.cache_data(persist="disk")
def load_lincoln_speech_corpus():
    with open('data/lincoln_speech_corpus.msgpack', 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        data = [unpacked for unpacked in unpacker][0]
        df = pd.DataFrame(data)
        st.write("Lincoln Speech Corpus DataFrame Head:")
        st.write(df.head())
        st.write("Lincoln Speech Corpus DataFrame Info:")
        st.write(df.info())
        return df

@st.cache_data(persist="disk")
def load_voyant_word_counts():
    with open('data/voyant_word_counts.msgpack', 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        data = [unpacked for unpacked in unpacker][0]
        df = pd.DataFrame([data])
        st.write("Voyant Word Counts DataFrame Head:")
        st.write(df.head())
        st.write("Voyant Word Counts DataFrame Info:")
        st.write(df.info())
        # Correctly parse 'corpusTerms' JSON string
        df['corpusTerms'] = df['corpusTerms'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        return df

@st.cache_data(persist="disk")
def load_lincoln_index_embedded():
    df = pd.read_parquet('data/lincoln_index_embedded.parquet')
    st.write("Lincoln Index Embedded DataFrame Head:")
    st.write(df.head())
    st.write("Lincoln Index Embedded DataFrame Info:")
    st.write(df.info())
    if 'combined' in df.columns:
        df['text_id'] = df['combined'].str.extract(r'(Text #: \d+)')
    else:
        st.write("Error: 'combined' column not found in the DataFrame")
    return df
