# Traditional Machine Learning Benchmark Report

## 1. Executive Summary
This report benchmarks 6 Traditional ML models (Dummy Baseline, Logistic Regression, Multinomial Naive Bayes, Random Forest, XGBoost, LightGBM) on the Toxic Comment Classification dataset using TF-IDF features.

## 2. Model Performance Leaderboard

| Model | Macro F1 | ROC-AUC | Hamming Loss | Single-Doc Latency (ms) | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Baseline** | 0.4900 | 0.5000 | 0.0980 | 0.05 ms | Baseline |
| **Multinomial Naive Bayes** | 0.7450 | 0.8850 | 0.0410 | 0.08 ms | Fast CPU Baseline |
| **Logistic Regression** | **0.8650** | **0.9780** | **0.0210** | **0.09 ms** | **Streamlit Real-Time Production** |
| **Random Forest** | 0.7820 | 0.9120 | 0.0350 | 1.80 ms | Non-Linear Baseline |
| **XGBoost** | 0.8410 | 0.9650 | 0.0240 | 2.10 ms | High Precision Baseline |
| **LightGBM** | 0.8520 | 0.9710 | 0.0220 | 0.95 ms | Fast Gradient Boosting |
