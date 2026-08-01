"""
Home Page - System Architecture & Live System Status.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
while not os.path.exists(os.path.join(ROOT_DIR, 'requirements.txt')) and os.path.dirname(ROOT_DIR) != ROOT_DIR:
    ROOT_DIR = os.path.dirname(ROOT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from dashboard.components.header import render_header
from dashboard.components.footer import render_footer
from dashboard.components.cards import render_kpi_cards
from opentrust_core.health import health_checker

render_header("🏠 Home - System Overview & Architecture", "Enterprise Multi-Label Content Moderation Platform")

selected_model_label = st.session_state.get("selected_model_label", "DistilBERT Transformer")
selected_model_id = st.session_state.get("selected_model", "distilbert")

st.markdown("""
## Executive System Summary
Welcome to **OpenTrust AI — Enterprise AI Trust & Safety Platform**.
This platform combines 12 machine learning, deep learning, and transformer models to deliver real-time content moderation, sentiment mining, 8-class emotion extraction, Explainable AI (SHAP/LIME), LLM prompt guardrails, and automated drift monitoring.
""")

st.info(f"⚡ **Active Selected Model**: **{selected_model_label}** (`{selected_model_id}`) | Change model in the left sidebar menu!")

render_kpi_cards()

st.divider()

st.subheader("🖥️ Platform Health & Microservices Status")
health_data = health_checker.get_full_health()
st.json({
    "status": health_data["status"],
    "version": health_data["version"],
    "checks": health_data["checks"],
})

st.divider()

st.subheader("System Architecture & Pipeline Workflow")
st.markdown("""
```
Input Text → Preprocessing & PII Masking → Feature Extraction → Selected Model Engine → Decision Audit Log
                                                                 ├── 6 Multi-Label Toxicity Categories
                                                                 ├── 3-Class Sentiment Polarity (VADER/TextBlob)
                                                                 ├── 8-Class NRC Emotion Distributions
                                                                 └── SHAP / LIME Feature Attributions
```
""")

render_footer()
