"""
Caching Utility for Streamlit.
"""

import streamlit as st
from src.sentiment.sentiment_pipeline import SentimentPipeline
from src.emotion.emotion_pipeline import EmotionPipeline

@st.cache_resource
def get_sentiment_pipeline():
    return SentimentPipeline()

@st.cache_resource
def get_emotion_pipeline():
    return EmotionPipeline()
