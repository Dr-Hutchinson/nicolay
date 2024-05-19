import streamlit as st

def load_prompt(file_name):
    """
    Load a prompt from a file.

    Parameters:
    file_name (str): The path to the file containing the prompt.

    Returns:
    str: The content of the file.
    """
    try:
        with open(file_name, 'r') as file:
            return file.read()
    except FileNotFoundError:
        st.error(f"File not found: {file_name}")
        return ""
    except Exception as e:
        st.error(f"An error occurred while loading the file {file_name}: {e}")
        return ""

def load_prompts():
    """
    Load various prompts into the Streamlit session state if they are not already loaded.
    """
    prompts = {
        'keyword_model_system_prompt': 'prompts/keyword_model_system_prompt.txt',
        'response_model_system_prompt': 'prompts/response_model_system_prompt.txt',
        'app_intro': 'prompts/app_intro.txt',
        'keyword_search_explainer': 'prompts/keyword_search_explainer.txt',
        'semantic_search_explainer': 'prompts/semantic_search_explainer.txt',
        'relevance_ranking_explainer': 'prompts/relevance_ranking_explainer.txt',
        'nicolay_model_explainer': 'prompts/nicolay_model_explainer.txt'
    }

    for key, file_path in prompts.items():
        if key not in st.session_state:
            st.session_state[key] = load_prompt(file_path)

# Ensure prompts are loaded
load_prompts()

# Now you can use the prompts from session state
keyword_prompt = st.session_state['keyword_model_system_prompt']
response_prompt = st.session_state['response_model_system_prompt']
app_intro = st.session_state['app_intro']
keyword_search_explainer = st.session_state['keyword_search_explainer']
semantic_search_explainer = st.session_state['semantic_search_explainer']
relevance_ranking_explainer = st.session_state['relevance_ranking_explainer']
nicolay_model_explainer = st.session_state['nicolay_model_explainer']
