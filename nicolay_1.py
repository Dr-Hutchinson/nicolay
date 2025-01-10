import streamlit as st
import json
import pygsheets
from google.oauth2 import service_account
import re
from openai import OpenAI
import cohere
import os
import pandas as pd
import numpy as np
from datetime import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(
    page_title="Nicolay: Exploring the Speeches of Abraham Lincoln with AI (version 0.2)",
    layout='wide',
    page_icon='🎩'
)

# global environment settings

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
#openai.api_key = os.getenv("OPENAI_API_KEY")

#client = OpenAI(api_key=openai_api_key)
client = OpenAI()


cohere_api_key = st.secrets["cohere_api_key"]
co = cohere.Client(cohere_api_key)

scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"], scopes = scope)

gc = pygsheets.authorize(custom_credentials=credentials)

api_sheet = gc.open('api_outputs')
api_outputs = api_sheet.sheet1


image_url = 'http://danielhutchinson.org/wp-content/uploads/2023/05/title_card.png'
st.markdown(f'<img src="{image_url}" width="500">', unsafe_allow_html=True)

#col1, col2, col3, col4, col5 = st.columns(5)

#with col1:
#    st.write(' ')

#with col2:
#    #st.image('./title_card.png', width=600)
#    image_url = 'http://danielhutchinson.org/wp-content/uploads/2023/05/title_card.png'
#    st.markdown(f'<img src="{image_url}" width="500">', unsafe_allow_html=True)


#with col3:
#    st.write(' ')

#with col4:
#    st.write(' ')

#with col5:
#    st.write(' ')

    #st.title("Can AIs Accurately Interpret History? A Digital History Experiment")

st.header("Nicolay: Exploring the Speeches of Abraham Lincoln with AI")

st.subheader("Project Description")

st.write("In an era where advancements in artificial intelligence, machine learning, and 'deepfakes' have raised concerns about their potential to distort our understanding of the past, this project spearheads a different approach. We explore whether these same technologies, particularly Retrieval Augmented Generation (RAG), can enhance our interaction with historic texts. Our focus: the collected speeches of Abraham Lincoln as a lens for understanding the Civil War era.\n\nThis project, using RAG techniques, opens a new window into these pivotal moments. *Nicolay*, our AI-powered app named after Lincoln's personal secretary, navigates Lincoln's collected speeches words to respond to your questions about this crucial period in the history of the United States.")

st.header("Project Elements")

st.write("**Exploring RAG with Hay and Nicolay:** This section explores how Retrieval Augmented Generation enables large language models to explore extensive text collections. Users can gain familarity with the mechanics of RAG via an interface for conducting their own searches over the Lincoln speech corpus using two finetuned LLMs, Hay and Nicolay.")

#st.write("**Ask Nicolay:** (Coming Winter 2024) This interactive feature invites users to engage in a dynamic dialogue with Lincoln’s speeches. Powered by a sophisticated RAG-based chatbot, this section offers an unprecedented opportunity to ask open-ended questions and receive responses showcasing Lincoln's historic language.")

st.write("**Evaluating Nicolay:** (Coming Winter 2025) This section presents preliminary data on the effectiveness of RAG techniques for analyzing the Lincoln corpus. It will offer insights into the potential improvements and current limitations of these technologies, providing a transparent view into the evolving capabilities of AI for historical research and interpretation.")

st.write("**Project Goals, Methods, and Acknowledgements**: Explores the aims of this project, some of the methods used, and thanks those who contributed to this project.")

st.subheader("**Developer:**")

st.write("Nicolay was developed by [Daniel Hutchinson](https://danielhutchinson.org/) for [Honest Abe's Information Emporium](https://honestabes.info/), a digital history project developed as part of the inaugural Digital Literacy Accelerator program offered by the U.S. Department of Education (2021-2022) and supported by a team of faculty at the University of Texas-San Antonio. Many thanks to Dr. Abe Gibson for the support on this project.")
