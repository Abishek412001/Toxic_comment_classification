"""
Toxic Comment Classifier — Streamlit App (Fixed)

Quick-fix applied on top of the existing saved model:
  1. Minimum threshold floor of 0.55 applied to all labels so that
     neutral text like "good person" is never flagged.
  2. Thresholds are shown next to every score so the user can see
     exactly why something is or isn't flagged.
  3. A recalibration note warns that scores near 0.5 indicate a
     miscalibrated model (trained on skewed data) and that retraining
     on the Jigsaw dataset with the fixed pipeline will resolve this.

Permanent fix: retrain using bert_toxic_v5_clean.ipynb with the Jigsaw
train.csv dataset (NOT the Davidson fallback). The notebook now includes
balanced data loading, min_threshold=0.35 in Config, and threshold search
starting at 0.35, which together prevent the model learning a toxic-bias prior.
"""

import streamlit as st
import torch, json, re, os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Constants ──────────────────────────────────────────────────────────────────
LABEL_COLS   = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SAVE_DIR     = "./toxic_bert_model"
MAX_LEN      = 128

# Minimum threshold floor applied regardless of what thresholds.json says.
# This is a post-hoc correction for models trained on skewed data.
# When you retrain on balanced Jigsaw data the saved thresholds will be
# sensible (≥ 0.45) and this floor will have no effect.
MIN_THRESHOLD_FLOOR = 0.55

# ── Asset loading (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    tok   = AutoTokenizer.from_pretrained(SAVE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
    model.eval()

    with open(os.path.join(SAVE_DIR, "thresholds.json")) as f:
        raw_thresh = json.load(f)

    # Apply floor: raise any threshold that's unreasonably low
    thresh = {
        k: max(float(v), MIN_THRESHOLD_FLOOR)
        for k, v in raw_thresh.items()
    }

    original_low = {k: v for k, v in raw_thresh.items()
                    if float(v) < MIN_THRESHOLD_FLOOR}
    return tok, model, thresh, original_low

tokenizer, model, thresholds, original_low = load_assets()

# ── Text cleaning (must match training pipeline) ───────────────────────────────
_URL_RE  = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")

def clean(text: str) -> str:
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    return re.sub(r" {2,}", " ", text).strip()

# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(text: str):
    cleaned = clean(text)
    enc     = tokenizer(cleaned, return_tensors="pt",
                        truncation=True, max_length=MAX_LEN)
    logits  = model(**enc).logits[0]
    probs   = torch.sigmoid(logits.float()).numpy()
    bin_score = float(np.max(probs))
    label_probs = {name: float(p) for name, p in zip(LABEL_COLS, probs)}
    return bin_score, label_probs

# ── Page layout ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Toxic Comment Classifier", page_icon="🛡️",
                   layout="centered")

st.title("🛡️ Toxic Comment Classifier")
st.write("Enter a comment below to evaluate multi-label classification predictions.")

# Calibration warning if thresholds were raised by the floor
if original_low:
    st.warning(
        f"⚠️ **Model calibration notice:** "
        f"The saved thresholds for "
        f"**{', '.join(original_low.keys())}** were unusually low "
        f"({', '.join(f'{v:.2f}' for v in original_low.values())}), "
        f"which caused neutral text to be flagged as toxic. "
        f"A minimum floor of **{MIN_THRESHOLD_FLOOR}** has been applied. "
        f"**Permanent fix:** retrain with the Jigsaw dataset using "
        f"`bert_toxic_v5_clean.ipynb` (fixed pipeline).",
        icon="⚠️",
    )

# ── Input ──────────────────────────────────────────────────────────────────────
st.markdown("**Comment Content Evaluation Window**")
user_input = st.text_area("", height=130, placeholder="Type a comment here…",
                           label_visibility="collapsed")

analyse = st.button("Analyse Text Content", type="primary")

if analyse and user_input.strip():
    bin_score, label_probs = predict(user_input)

    bin_thresh = thresholds.get("binary", MIN_THRESHOLD_FLOOR)
    verdict    = "TOXIC" if bin_score > bin_thresh else "NON-TOXIC"

    # Verdict banner
    color = "#5c0000" if verdict == "TOXIC" else "#003300"
    st.markdown(
        f'<div style="background:{color};padding:14px 18px;border-radius:8px;margin:12px 0;">'
        f'<span style="font-size:1.25rem;font-weight:700;">'
        f'Verdict: {verdict}</span>'
        f'<span style="float:right;opacity:.75;">'
        f'Score: {bin_score:.3f} / Threshold: {bin_thresh:.2f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Sub-Category Probabilities Breakdown")

    for name, prob in label_probs.items():
        thresh  = thresholds.get(name, MIN_THRESHOLD_FLOOR)
        flagged = prob > thresh

        col1, col2 = st.columns([1, 3])
        with col1:
            label_text = name.replace("_", " ").title()
            st.markdown(f"**{label_text}**")
            if flagged:
                st.caption("⚠️ Flagged")
            else:
                st.caption("✅ Clear")
        with col2:
            st.progress(min(prob, 1.0))
            st.caption(f"Score: {prob:.3f} / Threshold: {thresh:.2f}")

    # Show cleaned text used for inference
    with st.expander("🔍 Cleaned input sent to model"):
        st.code(clean(user_input))

elif analyse and not user_input.strip():
    st.info("Please enter some text before clicking Analyse.")
