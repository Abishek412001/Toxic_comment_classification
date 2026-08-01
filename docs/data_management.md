# Data & Artifact Management Strategy (Step 143)

## 1. Data Split Architecture
- **Raw (`data/raw/`)**: Original Kaggle datasets (`train.csv`, `test.csv`). Immutable.
- **Interim (`data/interim/`)**: Cleaned and tokenized corpora.
- **Processed (`data/processed/`)**: Feature matrices (`X_train.npz`, `y_train.npy`).

---

## 2. Model Artifact Integrity & Checksums
- Artifacts saved under `artifacts/models/` with SHA256 integrity checksums saved in `models/registry.json`.
