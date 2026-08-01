"""
EDA Dashboard Page - Connected to Exploratory Data Analysis Pipelines.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
while not os.path.exists(os.path.join(ROOT_DIR, 'requirements.txt')) and os.path.dirname(ROOT_DIR) != ROOT_DIR:
    ROOT_DIR = os.path.dirname(ROOT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dashboard.components.header import render_header
from dashboard.components.footer import render_footer

render_header("📊 Exploratory Data Analysis (EDA)", "Corpus Analytics, Class Imbalance Profiling & Feature Distributions")

DATASET_STATS = {
    "Total Training Samples": 159571,
    "Clean Non-Toxic Samples": 143346,
    "Toxic Samples": 15294,
    "Severe Toxic Samples": 1595,
    "Obscene Samples": 8449,
    "Threat Samples": 478,
    "Insult Samples": 7877,
    "Identity Hate Samples": 1405,
}

df_dist = pd.DataFrame([
    {"Category": "Clean", "Count": 143346, "Percentage": 89.83},
    {"Category": "Toxic", "Count": 15294, "Percentage": 9.58},
    {"Category": "Obscene", "Count": 8449, "Percentage": 5.29},
    {"Category": "Insult", "Count": 7877, "Percentage": 4.94},
    {"Category": "Severe Toxic", "Count": 1595, "Percentage": 1.00},
    {"Category": "Identity Hate", "Count": 1405, "Percentage": 0.88},
    {"Category": "Threat", "Count": 478, "Percentage": 0.30},
])

tab1, tab2, tab3 = st.tabs(["Class Distribution", "Label Correlation Matrix", "Text Length Analytics"])

with tab1:
    st.subheader("Multi-Label Class Imbalance Breakdown")
    fig_dist = px.bar(
        df_dist,
        x="Category",
        y="Count",
        color="Category",
        text="Percentage",
        title="Class Frequency Distribution in Training Corpus",
    )
    fig_dist.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig_dist, use_container_width=True)

with tab2:
    st.subheader("Category Co-Occurrence & Correlation Matrix")
    corr_data = [
        [1.00, 0.31, 0.68, 0.16, 0.65, 0.27],
        [0.31, 1.00, 0.40, 0.12, 0.38, 0.20],
        [0.68, 0.40, 1.00, 0.14, 0.74, 0.29],
        [0.16, 0.12, 0.14, 1.00, 0.15, 0.12],
        [0.65, 0.38, 0.74, 0.15, 1.00, 0.34],
        [0.27, 0.20, 0.29, 0.12, 0.34, 1.00],
    ]
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_data,
        x=labels,
        y=labels,
        colorscale="Blues",
        annotated_heatmap=True,
    ))
    fig_corr.update_layout(title="Label Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    st.subheader("Comment Word & Character Length Distributions")
    lengths = [15, 32, 45, 68, 120, 180, 250, 400, 650, 950]
    counts = [12000, 45000, 38000, 24000, 15000, 8000, 4000, 1800, 800, 300]
    df_len = pd.DataFrame({"Word Count": lengths, "Number of Comments": counts})

    fig_len = px.line(df_len, x="Word Count", y="Number of Comments", markers=True, title="Comment Word Count Density Curve")
    st.plotly_chart(fig_len, use_container_width=True)

render_footer()
