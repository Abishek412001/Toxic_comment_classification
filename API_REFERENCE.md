# Comprehensive API Reference Guide (Step 155)

## 1. Preprocessing Package (`src.preprocessing`)

### `TextCleaner`
```python
from src.preprocessing.text_cleaner import TextCleaner

cleaner = TextCleaner(remove_urls=True, remove_punct=True)
clean_text = cleaner.clean("Check out https://example.com! You're great.")
```

---

## 2. Feature Engineering Package (`src.features`)

### `TFIDFFeatureExtractor`
```python
from src.features.tfidf_extractor import TFIDFFeatureExtractor

extractor = TFIDFFeatureExtractor(max_features=5000, ngram_range=(1, 2))
X_vec = extractor.fit_transform(corpus)
```

---

## 3. Sentiment & Emotion Engines (`src.sentiment`, `src.emotion`)

### `SentimentPipeline`
```python
from src.sentiment.sentiment_pipeline import SentimentPipeline

pipeline = SentimentPipeline(engine_type="vader")
result = pipeline.analyze_text("This project is fantastic!")
# Output: {'sentiment_label': 'positive', 'confidence_score': 0.95}
```

### `EmotionPipeline`
```python
from src.emotion.emotion_pipeline import EmotionPipeline

pipeline = EmotionPipeline(engine_type="nrc")
result = pipeline.analyze_text("I feel full of rage and fury!")
# Output: {'emotion_label': 'anger', 'top_emotions': [('anger', 0.85)]}
```
