"""
Explainable AI (XAI) Page - Connected to SHAP & LIME Feature Attribution Engines.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
while not os.path.exists(os.path.join(ROOT_DIR, 'requirements.txt')) and os.path.dirname(ROOT_DIR) != ROOT_DIR:
    ROOT_DIR = os.path.dirname(ROOT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import plotly.express as px
from dashboard.components.header import render_header
from dashboard.components.footer import render_footer
from services.xai_service.engine import xai_engine
from services.xai_service.schemas import XAIExplanationRequest, XAIMethodEnum

render_header("🔍 Explainable AI (XAI) & Model Interpretability", "SHAP & LIME Word-Level Feature Attribution & Audit Reports")

selected_model_id = st.session_state.get("selected_model", "distilbert")
selected_model_label = st.session_state.get("selected_model_label", "DistilBERT Transformer")

st.info(f"🤖 **Active Selected Model Explainer Context**: {selected_model_label} (`{selected_model_id}`)")

xai_method_choice = st.radio(
    "Select Feature Attribution Method:",
    options=["SHAP (Shapley Additive exPlanations)", "LIME (Local Interpretable Model-agnostic Explanations)"],
    index=0,
    horizontal=True,
)

method_enum = XAIMethodEnum.SHAP if "SHAP" in xai_method_choice else XAIMethodEnum.LIME

sample_text = st.text_area(
    "Enter comment text to compute word-level feature attributions:",
    value="You are an absolute fraud, a liar, and a terrible human being!",
    height=100,
)

if st.button("Generate XAI Feature Explanation", key="btn_run_xai"):
    with st.spinner(f"Computing {method_enum.value.upper()} attributions for {selected_model_label}..."):
        res = xai_engine.explain_prediction(
            XAIExplanationRequest(
                text=sample_text,
                model_id=selected_model_id,
                method=method_enum,
                target_label="toxic",
            )
        )

        st.success(f"✅ {res.method.value.upper()} Explanation Generated (Latency: {res.latency_ms:.2f} ms)")

        # Feature Attribution Bar Chart
        features = [f.feature for f in res.attributions]
        scores = [f.importance_score for f in res.attributions]

        df_feat = pd.DataFrame({"Token / Word": features, "Importance Score": scores})
        df_feat = df_feat.sort_values(by="Importance Score", ascending=True)

        fig_bar = px.bar(
            df_feat,
            x="Importance Score",
            y="Token / Word",
            orientation="h",
            color="Importance Score",
            color_continuous_scale="Reds",
            title=f"Word-Level Feature Contribution to TOXIC Classification ({method_enum.value.upper()})",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Highlighted HTML Render
        st.markdown("#### Word Attribution Heatmap")
        if res.html_export:
            st.components.v1.html(res.html_export, height=120, scrolling=True)
        else:
            st.markdown(res.explanation_summary)

        # Download HTML Explanation
        if res.html_export:
            st.download_button(
                label="📥 Download Interactive HTML XAI Report",
                data=res.html_export,
                file_name=f"opentrust_xai_{method_enum.value.lower()}_report.html",
                mime="text/html",
            )

render_footer()
