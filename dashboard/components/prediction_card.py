"""
Prediction Result Card Component.
"""

import streamlit as st

def render_prediction_result(text: str, toxicity_prob: float, labels: list) -> None:
    st.subheader("Inference Result")
    st.write(f"**Input Text**: *\"{text}\"*")
    if toxicity_prob > 0.5:
        st.error(f"⚠️ **Toxic Comment Detected** (Risk Probability: {toxicity_prob:.2%})")
    else:
        st.success(f"✅ **Clean Comment** (Safety Confidence: {1.0 - toxicity_prob:.2%})")
