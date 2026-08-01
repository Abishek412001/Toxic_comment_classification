"""
Root Streamlit Entry Point for Streamlit Cloud Deployment.
OpenTrust AI - Toxic Comment Classification & Intelligence System.
"""

import os
import sys

# Ensure project root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from dashboard.utils.theme import load_css
from dashboard.utils.session import init_session_state
from dashboard.components.sidebar import render_sidebar
from dashboard.components.footer import render_footer

st.set_page_config(
    page_title="Toxic Comment AI Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session & Theme
init_session_state()
load_css()

# Render Global Sidebar
render_sidebar()

# Landing Page Banner & Documentation
st.title("🛡️ Toxic Comment Classification & Intelligence System")
st.markdown("### Production-Grade Enterprise Content Moderation, Sentiment & Emotion Mining Suite")
st.info("👈 **Use the left sidebar navigation menu** to explore the 8 interactive dashboard pages!")

st.markdown("""
### Multi-Page Navigation Overview:
1. **🏠 Home**: Project Architecture, System Benchmarks & Technology Stack
2. **📊 EDA Dashboard**: Exploratory Data Analysis & Class Balance Profiling
3. **⚠️ Toxicity Prediction**: Multi-Label Toxicity Classification Service
4. **😊 Sentiment Analysis**: VADER, TextBlob & Transformer Sentiment Mining
5. **🎭 Emotion Mining**: 7-Class Emotion Detection & Top-3 Probabilities
6. **🔍 Explainable AI (XAI)**: SHAP & LIME Feature Attribution & Text Highlighting
7. **📈 Model Performance**: 12-Model Leaderboards & ROC/PR Evaluation
8. **📥 Report Download Center**: Downloadable Reports (Markdown, PDF, CSV, HTML)
""")

render_footer()
