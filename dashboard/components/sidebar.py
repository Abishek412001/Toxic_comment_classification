"""
Sidebar Component for Streamlit UI.
"""

import streamlit as st

def render_sidebar() -> None:
    st.sidebar.image("https://img.icons8.com/color/96/000000/shield.png", width=64)
    st.sidebar.title("Toxic Comment AI")
    st.sidebar.markdown("**Enterprise Moderation Suite**")
    st.sidebar.divider()
    st.sidebar.selectbox("Active Inference Model", ["DistilBERT Transformer", "BiLSTM Neural Net", "XGBoost", "Logistic Regression"], key="selected_model")
    st.sidebar.divider()
    st.sidebar.info("💡 **Status**: System Ready (100% Pass)")
