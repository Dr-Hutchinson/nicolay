import streamlit as st
from modules.rag_process import RAGProcess
from modules.data_logging import DataLogger, log_keyword_search_results, log_semantic_search_results, log_reranking_results, log_nicolay_model_output
import json
import os
from openai import OpenAI
import cohere
import pygsheets
from google.oauth2 import service_account

import pkgutil
import llama_index

st.set_page_config(
    page_title="Nicolay: Exploring the Speeches of Abraham Lincoln with AI (version 0.2)",
    layout='wide',
    page_icon='🎩'
)


def find_module(module_name):
    for importer, modname, ispkg in pkgutil.walk_packages(llama_index.__path__):
        if module_name in modname:
            st.write(f"Found module: {modname}")

find_module("SimpleDirectoryReader")
