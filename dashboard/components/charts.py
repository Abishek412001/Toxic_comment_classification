"""
Plotly Chart Component for Streamlit UI.
"""

import streamlit as st

def render_plotly_chart(fig) -> None:
    if fig:
        st.plotly_chart(fig, use_container_width=True)
