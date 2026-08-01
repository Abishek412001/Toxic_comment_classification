# Enterprise Repository Architecture (Step 141)

## 1. Directory Tree Architecture

```
project_root/
├── data/                       # Dataset storage hierarchy
│   ├── raw/                    # Original, immutable Kaggle dataset (train.csv, test.csv)
│   ├── interim/                # Cleaned, normalized, and preprocessed text corpora
│   ├── processed/              # Feature matrices, train/val/test splits, TF-IDF matrices
│   ├── external/               # Lexicons (NRC Emotion Lexicon, VADER lexicons)
│   └── sample/                 # Subsampled datasets for fast testing (1,000 records)
│
├── artifacts/                  # Production model checkpoints & binary assets
│   ├── models/                 # Trained scikit-learn, XGBoost, PyTorch, & HF models
│   ├── tokenizer/              # DistilBERT & RoBERTa tokenizer files
│   ├── embeddings/             # Pre-trained word vector embeddings
│   ├── reports/                # Exported Markdown & PDF summary reports
│   ├── figures/                # 300 DPI publication charts & master dashboards
│   └── cache/                  # Temporary memory cache for Streamlit & XAI
│
├── src/                        # Core Python source code package
│   ├── preprocessing/          # Text cleaning, normalization, tokenization
│   ├── features/               # TF-IDF, N-grams, Embeddings, Feature selection
│   ├── models/                 # Traditional ML, Deep Learning, Transformer models
│   ├── sentiment/              # VADER, TextBlob, Transformer sentiment engines
│   ├── emotion/                # NRC Lexicon & Transformer emotion mining engines
│   ├── xai/                    # SHAP & LIME Explainable AI integration
│   ├── visualization/          # Plotly, Matplotlib, Seaborn dashboard builders
│   ├── dashboard/              # Streamlit web app layout & UI components
│   ├── mlops/                  # Configuration, logging, health probes, registry
│   ├── api/                    # REST API endpoints (FastAPI / Flask)
│   ├── common/                 # Abstract base classes, interfaces, constants
│   └── utils/                  # Helper utilities, I/O functions, timers
│
├── notebooks/                  # 95 sequential Jupyter notebooks (01_ through 95_)
├── tests/                      # Unit, integration, and performance test suite
├── deployment/                 # Dockerfile, docker-compose.yml, deployment guides
├── dashboard/                  # Multi-page Streamlit web app entry point (`app.py`)
├── outputs/                    # Runtime logs, figures, and reports
├── docs/                       # Project documentation, guides, and summary reports
├── configs/                    # Multi-environment YAML configuration files
├── scripts/                    # Automation & CLI execution scripts
├── logs/                       # Rotating system log files
├── .github/                    # GitHub Actions CI/CD workflows
├── requirements/               # Modular requirements (base, dev, prod)
└── README.md                   # Recruiter-winning GitHub homepage
```
