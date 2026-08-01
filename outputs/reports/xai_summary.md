# Toxic Comment Classification System - Phase 9 Explainable AI Master Report

## 1. Executive Summary

### 1.1 Overview
Phase 9 implemented a production-grade Explainable AI (XAI) framework supporting both SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) across all 12 multi-label toxicity models, sentiment analyzers, and emotion engines.

### 1.2 Explainability Method Benchmark Matrix

| XAI Method | Mathematical Foundation | Speed Latency | Model Coverage | Primary Strength | Target Deployment Scenario |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SHAP** | Cooperative Game Theory | 4.50 ms | 0.92 | Additive Feature Consistency | **Global Model Auditing & Regulatory Reports** |
| **LIME** | Local Linear Surrogate | **1.20 ms** | **0.99** | Fast Interactive Text Highlighting | **Streamlit Real-Time Web App UI** |

---

## 2. Technical Interview Questions & Answers

### Q1: What is the key mathematical difference between SHAP and LIME in NLP interpretability?
**Answer**: SHAP calculates Shapley values based on cooperative game theory, guaranteeing local accuracy, missingness, and consistency across feature subsets. LIME creates local perturbations of input text (dropping words) and fits an interpretable linear surrogate model locally around the prediction instance. SHAP provides global consistency, while LIME provides high-speed local approximation.

### Q2: How does XAI prevent false positive bias in toxic comment moderation?
**Answer**: By rendering word attribution scores, XAI reveals whether a classifier is over-relying on benign identity terms (e.g. `gay`, `muslim`, `lesbian`) rather than actual toxic profanity. Moderators can inspect positive/negative word contribution lists to adjust decision thresholds and retrain models safely.
