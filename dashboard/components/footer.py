"""
Footer Component for Streamlit UI.
"""

import streamlit as st

def render_footer() -> None:
    st.divider()
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>Toxic Comment Classification & Intelligence System | Production Architecture</p>", unsafe_allow_html=True)
