# Python API Reference Documentation (Step 147)

## 1. Preprocessing (`src.preprocessing`)
- `TextCleaner`: Standardizes text, strips URLs, handles contractions.

## 2. Feature Engineering (`src.features`)
- `TFIDFFeatureExtractor`: Fits N-gram TF-IDF vocabulary.

## 3. Models (`src.models`)
- `MultiLabelLogisticRegression`: Logistic regression baseline.

## 4. Sentiment & Emotion (`src.sentiment`, `src.emotion`)
- `SentimentPipeline`: VADER & Transformer sentiment classifier.
- `EmotionPipeline`: NRC & Transformer 7-class emotion classifier.

## 5. Explainable AI (`src.xai`)
- `XAIPipeline`: SHAP & LIME interpretability engine.
