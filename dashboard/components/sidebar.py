"""
Sidebar Component for Streamlit UI - Connected to Session State & Model Registry.
"""

from typing import Dict, Any
import streamlit as st

MODEL_OPTIONS = {
    "DistilBERT Transformer": "distilbert",
    "BiLSTM Neural Net": "bilstm",
    "XGBoost Classifier": "xgboost",
    "Multi-Label Logistic Regression": "logistic_regression",
}


def render_sidebar() -> Dict[str, Any]:
    """Renders global sidebar controls and syncs selections with st.session_state."""
    st.sidebar.image("https://img.icons8.com/color/96/000000/shield.png", width=64)
    st.sidebar.title("Toxic Comment AI")
    st.sidebar.markdown("**Enterprise Moderation Suite**")
    st.sidebar.divider()

    # Active Inference Model Selector
    selected_label = st.sidebar.selectbox(
        "Active Inference Model",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
        key="selected_model_label",
    )

    selected_model_id = MODEL_OPTIONS[selected_label]
    st.session_state["selected_model"] = selected_model_id
    st.session_state["selected_model_label"] = selected_label

    # Threshold Adjustment Slider
    confidence_threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        key="confidence_threshold",
    )

    # Enable PII & Guardrails Toggle
    enable_guardrails = st.sidebar.checkbox(
        "Enable LLM Guardrails & PII Redaction",
        value=True,
        key="enable_guardrails",
    )

    st.sidebar.divider()
    st.sidebar.info(f"💡 **Active Model**: {selected_label}\n\n**Threshold**: {confidence_threshold:.2f}")

    return {
        "model_id": selected_model_id,
        "model_label": selected_label,
        "threshold": confidence_threshold,
        "enable_guardrails": enable_guardrails,
    }
