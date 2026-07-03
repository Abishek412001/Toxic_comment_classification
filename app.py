
import streamlit as st, torch, json, re, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SAVE_DIR = "microsoft/deberta-v3-base"
MAX_LEN    = 128

@st.cache_resource
def load_assets():
    tok = AutoTokenizer.from_pretrained(SAVE_DIR)
    # FORCE the model onto the CPU for predictable cloud execution
    model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to("cpu")
    model.eval()
    with open(f"{SAVE_DIR}/thresholds.json") as f:
        thresh = json.load(f)
    return tok, model, thresh

tokenizer, model, thresholds = load_assets()

_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML = re.compile(r"<[^>]+>")

def clean(t):
    t = t.lower()
    t = _URL.sub(" ", t)
    t = _HTML.sub(" ", t)
    return re.sub(r" {2,}", " ", t).strip()

@torch.no_grad()
def predict(text):
    cleaned = clean(text)
    # Ensure inputs map to cpu explicitly
    enc = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=MAX_LEN).to("cpu")
    logits = model(**enc).logits[0]
    probs = torch.sigmoid(logits).numpy()
    bin_score = float(np.max(probs))
    return bin_score, {name: float(p) for name, p in zip(LABEL_COLS, probs)}

st.set_page_config(page_title="Toxic Classifier", page_icon="shield")
st.title("Toxic Comment Classifier")
st.write("Enter a comment and click Analyse.")

user_input = st.text_area("Comment", height=120)
if st.button("Analyse") and user_input.strip():
    bin_score, label_probs = predict(user_input)
    verdict = "TOXIC" if bin_score > thresholds.get("binary", 0.5) else "NON-TOXIC"
    st.subheader(f"Verdict: {verdict}  (score={bin_score:.3f})")
    for name, p in label_probs.items():
        flagged = p > thresholds.get(name, 0.5)
        st.write(f"{'⚠️' if flagged else '  '} **{name}**: {p:.3f}")
