"""
Session State Utility for Streamlit.
"""

import streamlit as st

def init_session_state() -> None:
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "distilbert"
    if "theme" not in st.session_state:
        st.session_state.theme = "recruiter"
