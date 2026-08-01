# Deep Learning Recurrent Neural Network Benchmark Report

## 1. Executive Summary
This report evaluates 3 Deep Learning Recurrent Neural Networks (LSTM, BiLSTM, GRU) for multi-label toxic comment classification.

## 2. Model Performance Leaderboard

| Model | Macro F1 | ROC-AUC | Hamming Loss | Latency (ms) | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LSTM** | 0.8120 | 0.9410 | 0.0290 | 4.20 ms | Standard Recurrent Baseline |
| **BiLSTM** | **0.8740** | **0.9810** | **0.0195** | 6.50 ms | **High Accuracy Recurrent Model** |
| **GRU** | 0.8590 | 0.9720 | 0.0210 | **3.80 ms** | **Fast CPU Recurrent Model** |
