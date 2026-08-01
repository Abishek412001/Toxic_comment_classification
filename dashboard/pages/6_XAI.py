"""
Step 127: Explainable AI Page.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from dashboard.components.header import render_header
from dashboard.components.footer import render_footer
from src.xai.xai_pipeline import XAIPipeline
from src.xai.lime_local import LIMELocalExplainer

render_header("🔍 Explainable AI (XAI)", "SHAP & LIME Feature Attribution & Text Highlighting")

method = st.radio("Select Interpretability Method:", ["SHAP", "LIME"], horizontal=True)
text = st.text_input("Enter text to explain:", "You are an idiot and a fool!")

if st.button("Generate Explanation"):
    pipeline = XAIPipeline()
    res = pipeline.explain_text(text, None)
    st.write("Top Positive Word Contributors (Increases Toxicity):", res["positive_contributors"])
    st.write("Top Negative Word Contributors (Decreases Toxicity):", res["negative_contributors"])

st.divider()
st.image("outputs/figures/xai_master_dashboard.png", use_container_width=True)

render_footer()
