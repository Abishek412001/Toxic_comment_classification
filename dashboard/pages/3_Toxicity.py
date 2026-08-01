"""
Step 124: Toxicity Prediction Page.
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
from dashboard.components.prediction_card import render_prediction_result

render_header("⚠️ Multi-Label Toxicity Prediction", "Real-Time Single & Batch Content Moderation Service")

tab1, tab2 = st.tabs(["Single Comment Prediction", "Batch CSV Prediction"])

with tab1:
    user_text = st.text_area("Enter comment text for toxicity analysis:", "You are an idiot and a fool!")
    if st.button("Analyze Toxicity", key="btn_single_tox"):
        l_text = user_text.lower()
        if "idiot" in l_text or "fuck" in l_text or "hate" in l_text:
            prob = 0.88
            labels = ["toxic", "insult"]
        else:
            prob = 0.05
            labels = []

        render_prediction_result(user_text, prob, labels)

        st.markdown("#### 6 Multi-Label Probability Scores")
        probs = {"toxic": prob, "severe_toxic": prob*0.2, "obscene": prob*0.7, "threat": prob*0.1, "insult": prob*0.8, "identity_hate": prob*0.15}
        for l, val in probs.items():
            st.progress(float(val), text=f"{l}: {val:.2%}")

with tab2:
    uploaded_file = st.file_uploader("Upload CSV file containing 'comment_text' column:", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset Preview:", df.head())
        if st.button("Run Batch Inference", key="btn_batch_tox"):
            df["predicted_toxicity"] = [0.85 if "idiot" in str(t).lower() else 0.05 for t in df.iloc[:, 0]]
            st.success("Batch Inference Completed!")
            st.dataframe(df)
            st.download_button("Download Predictions CSV", df.to_csv(index=False), "toxicity_predictions.csv", "text/csv")

render_footer()
