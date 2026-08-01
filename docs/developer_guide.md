# Developer Setup & Onboarding Guide (Step 147)

## 1. Quickstart Environment Setup
```bash
# Clone repository
git clone https://github.com/user/toxic-comment-classification.git
cd toxic-comment-classification

# Install dependencies
pip install -r deployment/requirements.txt -r deployment/requirements-dev.txt

# Run unit tests
python -m unittest discover -s tests -p "test_*.py"

# Launch Streamlit app
streamlit run dashboard/app.py
```
