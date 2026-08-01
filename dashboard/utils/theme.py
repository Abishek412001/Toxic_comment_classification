"""
Theme Utility for Streamlit.
"""

import os
import streamlit as st

def load_css(css_path: str = "dashboard/assets/css/style.css") -> None:
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
