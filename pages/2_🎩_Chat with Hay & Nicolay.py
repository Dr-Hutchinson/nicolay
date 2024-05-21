import streamlit as st
from modules.rag_process import RAGProcess
from modules.data_logging import DataLogger, log_keyword_search_results, log_semantic_search_results, log_reranking_results, log_nicolay_model_output
import json
import os
from openai import OpenAI
import cohere
import pygsheets
from google.oauth2 import service_account
import llama_index
import pkgutil

# chatbot development - 0.0 - basic UI for RAG search and data logging

st.set_page_config(
    page_title="Nicolay: Exploring the Speeches of Abraham Lincoln with AI (version 0.2)",
    layout='wide',
    page_icon='🎩'
)

def list_module_attributes(module, module_name):
    st.write(f"Attributes and methods in {module_name}:")
    for attr in dir(module):
        st.write(attr)

def explore_modules(package):
    for importer, modname, ispkg in pkgutil.walk_packages(package.__path__):
        full_path = f"{package.__name__}.{modname}"
        st.write(f"Exploring module: {full_path}")
        try:
            module = __import__(full_path, fromlist=[""])
            list_module_attributes(module, full_path)
        except ImportError as e:
            st.write(f"Failed to import {full_path}: {e}")

explore_modules(llama_index)
