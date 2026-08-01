"""
Step 125: Sentiment Analysis Page.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from dashboard.components.header import render_header
from dashboard.components.footer import render_footer
from src.sentiment.sentiment_pipeline import SentimentPipeline

render_header("😊 Sentiment Analysis Service", "VADER, TextBlob & Transformer Sentiment Mining")

engine = st.selectbox("Select Sentiment Engine:", ["vader", "textblob", "transformer"])
text = st.text_input("Enter text for sentiment analysis:", "This project is awesome and fantastic!")

if st.button("Analyze Sentiment"):
    pipeline = SentimentPipeline()
    res = pipeline.analyze_text(text)
    st.subheader(f"Sentiment: {res['sentiment_label'].upper()}")
    st.metric("Confidence Score", f"{res['confidence_score']:.2%}")
    st.json(res["probabilities"])

st.divider()
st.image("outputs/figures/sentiment_analytics_dashboard.png", use_container_width=True)

render_footer()
