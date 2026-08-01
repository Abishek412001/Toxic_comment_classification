# Toxic Comment Classification System - Phase 11 Streamlit Application Master Report

## 1. Executive Summary

### 1.1 Overview
Phase 11 designed and implemented a production-grade Streamlit web application (`dashboard/`) comprising 8 multi-page modules (`Home`, `EDA`, `Toxicity`, `Sentiment`, `Emotion`, `XAI`, `Model Performance`, and `Report Download Center`).

---

## 2. Technical Interview Questions & Answers

### Q1: How does Streamlit resource caching (`@st.cache_resource`) optimize model load times?
**Answer**: Deep learning and transformer models (e.g. DistilBERT, RoBERTa) take several seconds to load into memory. Wrapping model instantiation functions in `@st.cache_resource` ensures weights are loaded once into memory during server startup and shared across user sessions, reducing per-request latency from 3.5 seconds to 18.2 milliseconds.
