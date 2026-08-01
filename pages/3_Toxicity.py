"""
Toxicity Prediction Page - Connected to Moderation Engine, Guardrails & Decision Service.
"""

import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
while not os.path.exists(os.path.join(ROOT_DIR, 'requirements.txt')) and os.path.dirname(ROOT_DIR) != ROOT_DIR:
    ROOT_DIR = os.path.dirname(ROOT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
from dashboard.components.header import render_header
from dashboard.components.footer import render_footer
from dashboard.components.prediction_card import render_prediction_result
from services.moderation_service.engine import moderation_engine
from services.guardrail_service.engine import guardrail_engine
from services.decision_service.engine import decision_engine
from services.guardrail_service.schemas import PromptGuardrailRequest

render_header("⚠️ Multi-Label Toxicity Prediction", "Real-Time Content Moderation Engine & Batch Classification")

# Read active model and threshold from session state / sidebar
selected_model_id = st.session_state.get("selected_model", "distilbert")
selected_model_label = st.session_state.get("selected_model_label", "DistilBERT Transformer")
threshold = st.session_state.get("confidence_threshold", 0.50)
enable_guardrails = st.session_state.get("enable_guardrails", True)

st.info(f"🤖 **Active Inference Model**: {selected_model_label} (`{selected_model_id}`) | **Confidence Threshold**: `{threshold:.2f}`")

tab1, tab2 = st.tabs(["Single Comment Analysis", "Batch CSV Inference"])

with tab1:
    user_text = st.text_area(
        "Enter text snippet for multi-label toxicity evaluation:",
        value="This product is absolute garbage and the creator is a total fraud!",
        height=100,
    )

    if st.button("Run Toxicity Moderation", key="btn_single_tox"):
        start_time = time.perf_counter()

        # Step 1: Guardrail Inspection (PII & Injection Check)
        sanitized_text = user_text
        if enable_guardrails:
            guard_res = guardrail_engine.inspect_prompt(
                PromptGuardrailRequest(prompt=user_text, detect_pii=True, detect_injection=True)
            )
            sanitized_text = guard_res.sanitized_prompt

            if guard_res.contains_pii:
                st.warning(f"🔒 **PII Redaction Applied**: `{sanitized_text}`")
            if guard_res.contains_injection:
                st.error(f"🛡️ **Prompt Injection Attack Detected**: `{guard_res.injection_type}`")

        # Step 2: Moderation Model Prediction with active selected_model_id
        scores_obj = moderation_engine.classify_text(sanitized_text, model_id=selected_model_id)
        category_scores = scores_obj.model_dump()
        max_prob = max(category_scores.values()) if category_scores else 0.0

        flagged_labels = [k for k, v in category_scores.items() if v >= threshold]

        # Step 3: Decision Engine Audit Logging
        dec_res = decision_engine.evaluate_prediction(
            text=sanitized_text,
            predictions=category_scores,
            model_id=selected_model_id,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Save to session history
        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []
        st.session_state.prediction_history.append({
            "text": user_text,
            "sanitized_text": sanitized_text,
            "model": selected_model_label,
            "flagged": flagged_labels,
            "max_prob": max_prob,
            "risk_level": dec_res.risk_level.value,
            "latency_ms": latency_ms,
        })

        # Render Results UI
        render_prediction_result(user_text, max_prob, flagged_labels)

        st.markdown("#### 6 Multi-Label Category Probabilities")
        for category, prob in category_scores.items():
            st.progress(float(prob), text=f"**{category.upper()}**: {prob:.2%} {'(FLAGGED)' if prob >= threshold else ''}")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Risk Level", dec_res.risk_level.value)
        col_b.metric("HITL Review Required", "YES" if dec_res.requires_human_review else "NO")
        col_c.metric("Inference Latency", f"{latency_ms:.2f} ms")

with tab2:
    uploaded_file = st.file_uploader("Upload CSV dataset containing text column:", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📁 **Uploaded Dataset Preview**:", df.head())

        text_col = df.columns[0]
        if st.button("Run Batch Inference on CSV", key="btn_batch_tox"):
            with st.spinner(f"Processing batch moderation pipeline using {selected_model_label}..."):
                results = []
                for val in df[text_col]:
                    s_obj = moderation_engine.classify_text(str(val), model_id=selected_model_id)
                    s_dict = s_obj.model_dump()
                    max_p = max(s_dict.values()) if s_dict else 0.0
                    flags = [k for k, v in s_dict.items() if v >= threshold]
                    results.append({
                        "max_toxicity_score": round(max_p, 4),
                        "flagged_categories": ", ".join(flags) if flags else "CLEAN",
                        "status": "FLAGGED" if flags else "PASS",
                    })

                res_df = pd.concat([df, pd.DataFrame(results)], axis=1)
                st.success("✅ Batch Inference Completed Successfully!")
                st.dataframe(res_df)

                csv_data = res_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Annotated Predictions CSV",
                    data=csv_data,
                    file_name="opentrust_batch_toxicity_predictions.csv",
                    mime="text/csv",
                )

render_footer()
