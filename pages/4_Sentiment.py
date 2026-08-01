"""
Sentiment Analysis Page - Connected to VADER, TextBlob & Ensemble Sentiment Engines.
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
import plotly.graph_objects as go
import plotly.express as px
from dashboard.components.header import render_header
from dashboard.components.footer import render_footer
from services.sentiment_service.engine import sentiment_engine
from services.sentiment_service.schemas import SentimentEngineEnum, SentimentAnalysisRequest

render_header("😊 Sentiment Analysis & Subjectivity Mining", "Multi-Engine Polarity & Subjectivity Intelligence Engine")

selected_model_label = st.session_state.get("selected_model_label", "DistilBERT Transformer")
st.info(f"🤖 **Active Selected Model Context**: {selected_model_label}")

tab1, tab2 = st.tabs(["Single Text Sentiment", "Batch CSV Sentiment"])

with tab1:
    engine_choice = st.selectbox(
        "Select Sentiment Engine Algorithm:",
        options=["ENSEMBLE (VADER + TextBlob)", "VADER Lexicon", "TextBlob NLP"],
        index=0,
    )

    engine_enum_map = {
        "ENSEMBLE (VADER + TextBlob)": SentimentEngineEnum.ENSEMBLE,
        "VADER Lexicon": SentimentEngineEnum.VADER,
        "TextBlob NLP": SentimentEngineEnum.TEXTBLOB,
    }
    chosen_enum = engine_enum_map[engine_choice]

    sample_text = st.text_area(
        "Enter text for sentiment analysis:",
        value="The customer service team was extremely helpful, fast, and polite!",
        height=100,
    )

    if st.button("Analyze Sentiment", key="btn_single_sent"):
        res = sentiment_engine.analyze_sentiment(
            SentimentAnalysisRequest(text=sample_text, engine=chosen_enum)
        )

        st.subheader("Sentiment Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Polarity Label", res.label.value.upper())
        col2.metric("Compound Score", f"{res.compound_score:+.4f}")
        col3.metric("Subjectivity", f"{res.subjectivity:.2%}")
        col4.metric("Confidence Score", f"{res.confidence:.2%}")

        # Plotly Sentiment Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=res.compound_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Sentiment Polarity Score ({res.label.value.upper()})"},
            delta={'reference': 0.0},
            gauge={
                'axis': {'range': [-1.0, 1.0]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1.0, -0.05], 'color': "#EF4444"},
                    {'range': [-0.05, 0.05], 'color': "#F59E0B"},
                    {'range': [0.05, 1.0], 'color': "#10B981"}
                ],
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    uploaded_file = st.file_uploader("Upload CSV file containing text column:", type=["csv"], key="csv_sent")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded File Preview:", df.head())
        text_col = df.columns[0]

        if st.button("Run Batch Sentiment Mining", key="btn_batch_sent"):
            with st.spinner("Mining sentiment across dataset..."):
                labels, compounds, subjs = [], [], []
                for text_val in df[text_col]:
                    r = sentiment_engine.analyze_sentiment(
                        SentimentAnalysisRequest(text=str(text_val), engine=chosen_enum)
                    )
                    labels.append(r.label.value)
                    compounds.append(r.compound_score)
                    subjs.append(r.subjectivity)

                res_df = df.copy()
                res_df["sentiment_label"] = labels
                res_df["compound_score"] = compounds
                res_df["subjectivity"] = subjs

                st.success("Batch Sentiment Mining Complete!")
                st.dataframe(res_df)

                fig_pie = px.pie(res_df, names="sentiment_label", title="Batch Sentiment Distribution", color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig_pie, use_container_width=True)

                st.download_button(
                    label="📥 Download Sentiment Analysis CSV",
                    data=res_df.to_csv(index=False),
                    file_name="opentrust_sentiment_results.csv",
                    mime="text/csv",
                )

render_footer()
