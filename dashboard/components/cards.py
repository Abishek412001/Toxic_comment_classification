"""
Cards Component for Metric Display.
"""

import streamlit as st

def render_kpi_cards() -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Toxicity Rate", "9.58%", "-1.2%")
    c2.metric("Champion F1 Score", "0.9250", "+0.045")
    c3.metric("Inference Latency", "18.2 ms", "Fast")
    c4.metric("System Health", "100%", "HEALTHY")
