# End-to-End Data Flow Specification (Step 154)

## 1. End-to-End Data Flow Pipeline

```mermaid
flowchart LR
    A[Raw Kaggle Dataset] --> B[Text Cleaner & Preprocessor]
    B --> C[Tokenized Text & TF-IDF Vectors]
    C --> D[Multi-Label Toxicity Classifier]
    C --> E[Sentiment Mining Engine]
    C --> F[Emotion Mining Engine]
    D --> G[SHAP & LIME XAI Explainer]
    D --> H[Dashboard & Plotly Visualizer]
    E --> H
    F --> H
    G --> H
    H --> I[Markdown & PDF Executive Reports]
```

---

## 2. Detailed Data Transformation Steps

1. **Ingestion**: Raw text input read from single form input or batch CSV file.
2. **Preprocessing**: Lowercasing, URL removal, contraction expansion, lemmatization via `src/preprocessing/text_cleaner.py`.
3. **Feature Extraction**: Vocabulary lookup and TF-IDF transformation via `src/features/tfidf_extractor.py`.
4. **Inference**: Parallel evaluation across multi-label toxicity, 3-class sentiment, and 7-class emotion engines.
5. **Explanation**: Computation of SHAP shapley values and LIME local linear surrogate models.
6. **Rendering**: Dynamic chart generation via Plotly and Streamlit.
