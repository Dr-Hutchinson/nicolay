# modules/astra_connection.py

import streamlit as st
import os
from typing import Optional, Dict, Any, Tuple
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_astra_credentials() -> Tuple[Optional[str], Optional[str]]:
    """
    Retrieve Astra DB credentials from Streamlit secrets or environment variables.

    Returns:
        Tuple of (astra_db_id, astra_db_token) or (None, None) if not found
    """
    # Try to get from Streamlit secrets first
    astra_db_id = st.secrets.get("ASTRA_DB_ID") if "ASTRA_DB_ID" in st.secrets else None
    astra_db_token = st.secrets.get("ASTRA_DB_APPLICATION_TOKEN") if "ASTRA_DB_APPLICATION_TOKEN" in st.secrets else None

    # Fall back to environment variables if not in secrets
    if not astra_db_id:
        astra_db_id = os.getenv("ASTRA_DB_ID")

    if not astra_db_token:
        astra_db_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

    return astra_db_id, astra_db_token

def validate_astra_credentials() -> Tuple[bool, str]:
    """
    Validate that Astra DB credentials are available.

    Returns:
        Tuple of (is_valid, message)
    """
    astra_db_id, astra_db_token = get_astra_credentials()

    if not astra_db_id or not astra_db_token:
        missing = []
        if not astra_db_id:
            missing.append("ASTRA_DB_ID")
        if not astra_db_token:
            missing.append("ASTRA_DB_APPLICATION_TOKEN")

        return False, f"Missing Astra DB credentials: {', '.join(missing)}"

    return True, "Astra DB credentials found"

def attempt_connection(test_query: str = "Lincoln presidency") -> Tuple[bool, str, Any]:
    """
    Attempt to connect to Astra DB and perform a test query.

    Args:
        test_query: Query to use for testing connection

    Returns:
        Tuple of (success, message, results)
    """
    from modules.colbert_search import ColBERTSearcher

    try:
        # Create a minimal searcher instance
        searcher = ColBERTSearcher()

        # Perform a test search
        results = searcher.search(test_query, k=2)

        if results is not None:
            if not results.empty:
                return True, f"Successfully connected to Astra DB and retrieved {len(results)} results", results
            else:
                return True, "Successfully connected to Astra DB but no results found for test query", results
        else:
            return False, "Connection established but search returned None", None

    except Exception as e:
        error_trace = traceback.format_exc()
        return False, f"Connection failed: {str(e)}\n{error_trace}", None

def display_connection_status():
    """
    Display Astra DB connection status in the Streamlit UI.
    """
    is_valid, message = validate_astra_credentials()

    if is_valid:
        st.sidebar.success("✅ Astra DB credentials: Valid")
    else:
        st.sidebar.error(f"❌ Astra DB credentials: {message}")
        st.sidebar.info("""
        To connect to Astra DB:
        1. Create an account at datastax.com
        2. Create a database and generate an application token
        3. Add credentials to your .streamlit/secrets.toml file or environment variables
        """)

    return is_valid
