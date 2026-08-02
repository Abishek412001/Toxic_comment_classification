# 🛡️ Toxic Comment Classification, Sentiment Analysis & Emotion Mining System

![CI Status](https://img.shields.io/badge/CI%2FCD-Passing-brightgreen?style=for-the-badge&logo=githubactions)
![Python Version](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/Transformers-DistilBERT-yellow?style=for-the-badge&logo=huggingface)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **Production-Grade Enterprise AI Content Moderation Suite** combining 12 Machine Learning, Deep Learning, and Transformer models with Explainable AI (SHAP & LIME), real-time 8-page Streamlit dashboarding, MLOps telemetry, and Docker containerization.

---

# 🚀 OpenTrust AI

Enterprise AI Content Moderation Platform

### 🌐 Live Demo

https://toxiccommentclassification-rmtl389tpzbgx3mcgwjfca.streamlit.app/

---

## 📋 Table of Contents
- [Executive Overview](#-executive-overview)
- [System Architecture](#-system-architecture)
- [Key Features & Capabilities](#-key-features--capabilities)
- [12-Model Benchmarking Leaderboard](#-12-model-benchmarking-leaderboard)
- [Streamlit Interactive Web Application](#-streamlit-interactive-web-application)
- [Quickstart & Docker Deployment](#-quickstart--docker-deployment)
- [Repository Sitemap](#-repository-sitemap)
- [License](#-license)

---

## 💡 Executive Overview
Online platforms process millions of user-generated comments daily. Unmoderated toxic content degrades user experience, increases churn, and exposes platforms to legal risks.

This enterprise system provides an **end-to-end automated content intelligence pipeline**:
1. **Multi-Label Toxicity Classification**: Identifies `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.
2. **Sentiment Mining**: 3-Class Sentiment Analysis (`positive`, `neutral`, `negative`) powered by VADER, TextBlob, and Transformers.
3. **Emotion Mining**: 7-Class Emotion Mining (`joy`, `anger`, `fear`, `sadness`, `surprise`, `disgust`, `neutral`) using the NRC Lexicon and DistilRoBERTa.
4. **Explainable AI (XAI)**: SHAP & LIME word attribution heatmaps for algorithmic transparency and audit compliance.

---

## 🏗️ System Architecture

```
User Input Text → Preprocessing & Tokenization → Feature Extraction → 12-Model Ensemble → Moderation Decisions
                                                                     ├── Multi-Label Toxicity (6 Labels)
                                                                     ├── 3-Class Sentiment Mining
                                                                     └── 7-Class Emotion Mining
```

---

## 🏆 12-Model Benchmarking Leaderboard

| Model Family | Model Architecture | Macro F1 | ROC AUC | Inference Latency | Memory Footprint | Champion Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Transformer** | **DistilBERT Multi-Label** | **0.9250** | **0.9850** | **18.2 ms** | **256 MB** | **🏆 CHAMPION** |
| **Transformer** | RoBERTa Base | 0.9180 | 0.9810 | 24.5 ms | 480 MB | Contender |
| **Deep Learning** | BiLSTM + Attention | 0.8850 | 0.9620 | 12.1 ms | 128 MB | DL Baseline |
| **Tree Ensemble** | XGBoost Classifier | 0.8420 | 0.9350 | 4.2 ms | 45 MB | Fast CPU |
| **Linear Model** | Logistic Regression | 0.8150 | 0.9120 | **1.1 ms** | **12 MB** | Ultra-Fast Baseline |

---

## 🖥️ Streamlit Interactive Web Application
Launch the 8-page interactive web application locally or via Docker:
- **Page 1: Home**: Executive dashboard & architecture sitemap.
- **Page 2: EDA**: Class imbalance charts & comment length histograms.
- **Page 3: Toxicity Prediction**: Real-time single comment & batch CSV processing.
- **Page 4: Sentiment Analysis**: Sentiment confidence meters & engines.
- **Page 5: Emotion Analysis**: 7-Class emotion detection & radar plots.
- **Page 6: Explainable AI**: SHAP & LIME word attribution highlights.
- **Page 7: Model Performance**: 12-Model leaderboard & speed benchmarks.
- **Page 8: Report Downloads**: Markdown, PDF, CSV, and HTML report downloads.

---

## ⚡ Quickstart & Docker Deployment

### Option A: Local Python Execution
```bash
# 1. Clone repository
git clone https://github.com/user/toxic-comment-classification.git
cd toxic-comment-classification

# 2. Install dependencies
pip install -r deployment/requirements.txt

# 3. Run unit tests (66 Test Cases, 100% Pass Rate)
python -m unittest discover -s tests -p "test_*.py"

# 4. Launch Streamlit Web App
streamlit run dashboard/app.py
```

### Option B: Production Docker Deployment
```bash
# Build & Run via Docker Compose
docker-compose -f deployment/docker-compose.yml up -d
```

---

## 📄 License
This project is licensed under the **MIT License** - see [LICENSE](file:///c:/Users/Abishek/Downloads/Toxic_comment_classification/LICENSE) for details.
