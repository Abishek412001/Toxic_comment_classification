"""
Emotion Mining Page - Connected to 8-Class NRC Emotion Analyzer Engine.
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
import plotly.graph_objects as go
from dashboard.components.header import render_header
from dashboard.components.footer import render_footer
from services.emotion_service.engine import emotion_engine
from services.emotion_service.schemas import EmotionDetectionRequest

render_header("🎭 Emotion Intelligence & Mining", "8-Class Emotion Distribution Analyzer (NRC Lexicon & Transformer)")

selected_model_label = st.session_state.get("selected_model_label", "DistilBERT Transformer")
st.info(f"🤖 **Active Selected Model Context**: {selected_model_label}")

tab1, tab2 = st.tabs(["Single Text Emotion Detection", "Batch CSV Emotion Extraction"])

with tab1:
    sample_text = st.text_area(
        "Enter text for emotion analysis:",
        value="I am terrified of losing my job and feel completely helpless about the future.",
        height=100,
    )

    if st.button("Detect Emotions", key="btn_single_emo"):
        res = emotion_engine.detect_emotions(EmotionDetectionRequest(text=sample_text))

        st.subheader(f"Primary Emotion: {res.primary_emotion.upper()} (Dominance: {res.primary_confidence:.2%})")

        # Top-3 Emotions Display
        st.markdown("#### Top-3 Ranked Emotions")
        cols = st.columns(3)
        for idx, emo in enumerate(res.top_emotions[:3]):
            cols[idx].metric(f"Rank #{idx+1} Emotion", emo.emotion.upper(), f"{emo.probability:.2%}")

        # Interactive Radar Chart for 8 Emotions
        emo_names = list(res.emotion_scores.keys())
        emo_probs = list(res.emotion_scores.values())

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=emo_probs,
            theta=[e.upper() for e in emo_names],
            fill='toself',
            name='Emotion Profile',
            line_color='#6366F1',
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.0])),
            showlegend=False,
            height=350,
            title="8-Class Emotion Radar Distribution",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Bar Chart of All Emotion Scores
        df_emo = pd.DataFrame({"Emotion": [e.upper() for e in emo_names], "Probability": emo_probs})
        fig_bar = px.bar(df_emo, x="Emotion", y="Probability", color="Probability", color_continuous_scale="Blues", title="Emotion Probability Breakdown")
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    uploaded_file = st.file_uploader("Upload CSV file containing text column:", type=["csv"], key="csv_emo")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded File Preview:", df.head())
        text_col = df.columns[0]

        if st.button("Run Batch Emotion Extraction", key="btn_batch_emo"):
            with st.spinner("Extracting emotion distributions across dataset..."):
                primary_list, top3_list = [], []
                for val in df[text_col]:
                    r = emotion_engine.detect_emotions(EmotionDetectionRequest(text=str(val)))
                    primary_list.append(r.primary_emotion)
                    top3_list.append(", ".join([e.emotion for e in r.top_emotions[:3]]))

                res_df = df.copy()
                res_df["primary_emotion"] = primary_list
                res_df["top3_emotions"] = top3_list

                st.success("Batch Emotion Extraction Complete!")
                st.dataframe(res_df)

                fig_pie = px.pie(res_df, names="primary_emotion", title="Dataset Dominant Emotion Distribution", color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_pie, use_container_width=True)

                st.download_button(
                    label="📥 Download Emotion Extraction CSV",
                    data=res_df.to_csv(index=False),
                    file_name="opentrust_emotion_extraction_results.csv",
                    mime="text/csv",
                )

render_footer()
