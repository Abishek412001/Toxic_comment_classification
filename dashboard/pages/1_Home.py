"""
Step 122: Home Page.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from dashboard.components.header import render_header
from dashboard.components.footer import render_footer
from dashboard.components.cards import render_kpi_cards

render_header("🏠 Home - System Overview & Architecture", "Enterprise Multi-Label Content Moderation Platform")

st.markdown("""
## Hero Executive Summary
Welcome to the **Toxic Comment Classification, Sentiment Analysis, and Emotion Mining System**.
This enterprise solution combines 12 machine learning, deep learning, and transformer models to process online user feedback in real time.
""")

render_kpi_cards()

st.divider()

st.subheader("System Architecture & Pipeline Workflow")
st.markdown("""
```
Input Text → Preprocessing & Tokenization → Feature Extraction → 12-Model Ensemble → Moderation Decisions
                                                                 ├── Multi-Label Toxicity (6 Classes)
                                                                 ├── 3-Class Sentiment Analysis
                                                                 └── 7-Class Emotion Mining
```
""")

st.subheader("Technology Stack")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### Core ML & DL\n- Python 3.10\n- Scikit-Learn\n- PyTorch & TensorFlow\n- XGBoost & LightGBM")
with col2:
    st.markdown("### NLP & Transformers\n- HuggingFace Transformers\n- DistilBERT & RoBERTa\n- VADER & TextBlob\n- NLTK & Spacy")
with col3:
    st.markdown("### Explainability & UI\n- SHAP & LIME XAI\n- Plotly & Seaborn\n- Streamlit Framework\n- REST API & Docker")

render_footer()
