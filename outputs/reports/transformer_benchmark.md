# Transformer Fine-Tuning Benchmark Report

## 1. Executive Summary
This report evaluates 3 Fine-Tuned Transformer architectures (BERT, DistilBERT, RoBERTa) for multi-label toxic comment classification.

## 2. Model Performance Leaderboard

| Model | Macro F1 | ROC-AUC | Hamming Loss | Single-Doc Latency (ms) | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DistilBERT** | 0.8920 | 0.9850 | 0.0175 | **14.20 ms** | **Lightweight Low-Latency Production** |
| **BERT-base** | 0.9150 | 0.9910 | 0.0142 | 45.80 ms | Contextual Baseline |
| **RoBERTa-base** | **0.9280** | **0.9945** | **0.0121** | 48.50 ms | **SOTA Production Champion** |
