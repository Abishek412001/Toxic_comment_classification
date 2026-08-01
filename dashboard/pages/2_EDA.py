"""
Step 123: EDA Dashboard Page.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
from dashboard.components.header import render_header
from dashboard.components.footer import render_footer
from src.visualization.toxicity_dashboard import ToxicityDashboard

render_header("📊 EDA Dashboard", "Exploratory Data Analysis & Class Balance Profiling")

st.markdown("### Dataset Summary & Health Check")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", "159,571")
col2.metric("Missing Values", "0 (Clean)")
col3.metric("Duplicate Comments", "0 (Unique)")
col4.metric("Toxicity Classes", "6 Labels")

st.divider()

st.markdown("### Interactive Multi-Label Toxicity Visualizations")
ToxicityDashboard.render_toxicity_dashboard()
st.image("outputs/figures/toxicity_analytics_dashboard.png", use_container_width=True)

render_footer()
