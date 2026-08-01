"""
Step 129: Report Download Center Page.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from dashboard.components.header import render_header
from dashboard.components.footer import render_footer

render_header("📥 Report Download Center", "Download Executive Summaries & HTML/PDF/CSV Artifacts")

st.markdown("### Available Project Reports")

reports = [
    ("Executive Multi-Label Summary (Markdown)", "outputs/reports/multilabel_final_report.md", "text/markdown"),
    ("Sentiment Analysis Summary (Markdown)", "outputs/reports/sentiment_analysis_summary.md", "text/markdown"),
    ("Emotion Mining Summary (Markdown)", "outputs/reports/emotion_mining_summary.md", "text/markdown"),
    ("Explainable AI Summary (Markdown)", "outputs/reports/xai_summary.md", "text/markdown"),
    ("Visualization Summary (Markdown)", "outputs/reports/visualization_summary.md", "text/markdown"),
    ("Toxicity Analytics Interactive HTML", "outputs/reports/toxicity_analytics.html", "text/html"),
    ("Sentiment Analytics Interactive HTML", "outputs/reports/sentiment_analytics.html", "text/html"),
    ("Emotion Analytics Interactive HTML", "outputs/reports/emotion_analytics.html", "text/html"),
    ("Executive KPIs Interactive HTML", "outputs/reports/executive_kpis.html", "text/html"),
]

for label, path, mime in reports:
    if os.path.exists(path):
        with open(path, "rb") as f:
            st.download_button(f"📥 Download {label}", f.read(), file_name=os.path.basename(path), mime=mime)

render_footer()
