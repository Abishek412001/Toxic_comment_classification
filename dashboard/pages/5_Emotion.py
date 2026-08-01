"""
Step 126: Emotion Analysis Page.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from dashboard.components.header import render_header
from dashboard.components.footer import render_footer
from src.emotion.emotion_pipeline import EmotionPipeline

render_header("🎭 Emotion Mining Service", "7-Class Emotion Detection & Top-3 Probabilities")

engine = st.selectbox("Select Emotion Engine:", ["nrc", "transformer"])
text = st.text_input("Enter text for emotion analysis:", "I feel mad, furious, and full of rage!")

if st.button("Analyze Emotion"):
    pipeline = EmotionPipeline()
    res = pipeline.analyze_text(text)
    st.subheader(f"Primary Emotion: {res['emotion_label'].upper()}")
    st.metric("Confidence Score", f"{res['confidence_score']:.2%}")
    st.markdown("#### Top-3 Emotions")
    for emo, prob in res["top_emotions"]:
        st.write(f"- **{emo}**: {prob:.2%}")

st.divider()
st.image("outputs/figures/emotion_analytics_dashboard.png", use_container_width=True)

render_footer()
