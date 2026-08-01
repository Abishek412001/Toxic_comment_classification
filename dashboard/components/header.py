"""
Header Component for Streamlit UI.
"""

import streamlit as st

def render_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.divider()
