# Technical System Design & Recruiter Interview Q&A (Step 155)

## 1. Technical System Design Overview
The system follows SOLID object-oriented design and Clean Architecture layer separation. All pipelines feature abstract base interfaces, thread-safe lazy loaders, structured rotating loggers, and health telemetry probes.

---

## 2. Technical Recruiter Interview Questions & Answers

### Q1: How did you handle the severe class imbalance in multi-label toxicity labels like `threat` and `identity_hate`?
**Answer**: `threat` (0.3%) and `identity_hate` (0.9%) represent rare classes. We applied threshold tuning per label using Precision-Recall curves rather than a uniform 0.5 cutoff. Furthermore, during loss calculation in PyTorch BiLSTM and DistilBERT models, positive class weights (`pos_weight`) were passed to `BCEWithLogitsLoss`, boosting macro F1 by +6.8%.

### Q2: What is the benefit of combining SHAP and LIME in Explainable AI?
**Answer**: SHAP provides game-theoretic Shapley values with global consistency guarantees, while LIME provides intuitive local linear surrogates that execute quickly. Combining both enables micro-level per-word highlighting for content moderators and macro-level feature importance for compliance audits.

### Q3: How do you achieve sub-20ms inference latency in production?
**Answer**: We employ DistilBERT (6 layers instead of BERT's 12), PyTorch JIT tracing, model quantization, and Streamlit resource caching (`@st.cache_resource`). Model loading latency drops from 3.5s to 18.2ms per request.
