"""
Step 128: Model Performance Page.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from dashboard.components.header import render_header
from dashboard.components.footer import render_footer

render_header("📈 Model Performance & Leaderboard", "12-Model Benchmarking, ROC/PR Curves & Speed Evaluation")

st.markdown("### Master Model Leaderboard")
st.image("outputs/figures/model_performance_dashboard.png", use_container_width=True)

render_footer()
