# data_utils.py
import streamlit as st
import json
import pandas as pd

@st.cache_data(persist="disk")
def load_lincoln_speech_corpus():
    with open('data/lincoln_speech_corpus.json', 'r') as file:
        return json.load(file)

@st.cache_data(persist="disk")
def load_voyant_word_counts():
    with open('data/voyant_word_counts.json', 'r') as file:
        return json.load(file)

@st.cache_data(persist="disk")
def load_lincoln_index_embedded():
    return pd.read_csv('lincoln_index_embedded.csv')
