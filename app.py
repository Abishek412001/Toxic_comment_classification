"""
Toxic Comment Classifier — Streamlit App
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"   # suppress vision-module scan noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # suppress tokenizer fork warning

import streamlit as st, torch, json, re, urllib.request
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. MUST BE THE ABSOLUTE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Toxic Classifier", page_icon="🛡️")

# --- 2. CONFIGURATIONS & CONSTANTS ---
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_LEN    = 128
LOCAL_MODEL_DIR = "./toxic_bert_model"
BASE_MODEL = "microsoft/deberta-v3-base"
MIN_THRESHOLD_FLOOR = 0.55

# --- 3. ASSET CACHING LAYER ---
@st.cache_resource
def load_assets():
    """Downloads fine-tuned weights dynamically if missing, then loads all assets."""
    if not os.path.exists(LOCAL_MODEL_DIR):
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        
    # --- DOWNLOAD THE WEIGHTS ---
    WEIGHTS_URL = "https://huggingface.co/abishekw412001/toxic-deberta-v5/resolve/main/best_checkpoint.pt?download=true"
    local_weights_path = os.path.join(LOCAL_MODEL_DIR, "best_checkpoint.pt")
    
    if not os.path.exists(local_weights_path):
        with st.spinner("Downloading fine-tuned weights matrix... Please wait."):
            temp_path = local_weights_path + ".tmp"
            urllib.request.urlretrieve(WEIGHTS_URL, temp_path)
            os.rename(temp_path, local_weights_path)

    # --- LOAD TOKENIZER ---
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
    # --- BUILD MODEL ARCHITECTURE CLEANLY ---
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=6)
    
    try:
        raw_checkpoint = torch.load(local_weights_path, map_location="cpu")
        if isinstance(raw_checkpoint, dict) and "state_dict" in raw_checkpoint:
            state_dict = raw_checkpoint["state_dict"]
        else:
            state_dict = raw_checkpoint
            
        clean_state_dict = {}
        for k, v in state_dict.items():
            clean_key = k.replace("model.", "").replace("module.", "")
            clean_state_dict[clean_key] = v
            
        model.load_state_dict(clean_state_dict, strict=False)
    except Exception as e:
        st.warning(f"Custom weights load failed, falling back to base model topology. Error: {e}")
        
    model.to("cpu")
    model.eval()
    
    # --- LOAD THRESHOLDS ---
    threshold_path = "thresholds.json"
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            raw = json.load(f)
    else:
        raw = {name: 0.5 for name in LABEL_COLS}
        raw["binary"] = 0.5
        
    thresh = {k: max(float(v), MIN_THRESHOLD_FLOOR) for k, v in raw.items()}
    original_low = {k: float(v) for k, v in raw.items() if float(v) < MIN_THRESHOLD_FLOOR}
        
    return tokenizer, model, thresh, original_low

tokenizer, model, thresholds, original_low = load_assets()

# --- Text Cleaning Engine ---
_URL  = re.compile(r"https?://\S+|www\.\S+")
_HTML = re.compile(r"<[^>]+>")

def clean(text):
    return re.sub(r" {2,}", " ", _HTML.sub(" ", _URL.sub(" ", text.lower()))).strip()

@torch.no_grad()
def predict(text):
    cleaned = clean(text)
    enc = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=MAX_LEN).to("cpu")
    logits = model(**enc).logits[0]
    probs = torch.sigmoid(logits.float()).numpy()
    bin_score = float(np.max(probs))
    return bin_score, {name: float(p) for name, p in zip(LABEL_COLS, probs)}

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🛡️ Toxic Comment Classifier")
st.write("Enter a comment below to evaluate multi-label classification predictions.")

if original_low:
    st.warning(
        f"⚠️ Saved thresholds for **{', '.join(original_low)}** were below "
        f"{MIN_THRESHOLD_FLOOR} ({', '.join(f'{v:.2f}' for v in original_low.values())}). "
        f"Floor applied. Retrain on Jigsaw data for a permanent fix.",
        icon="⚠️",
    )

st.markdown("**Comment Content Evaluation Window**")
user_input = st.text_area("", height=130, placeholder="Type a comment here…",
                           label_visibility="collapsed")

if st.button("Analyse Text Content", type="primary") and user_input.strip():
    bin_score, label_probs = predict(user_input)
    bin_thresh = thresholds.get("binary", MIN_THRESHOLD_FLOOR)
    verdict    = "TOXIC" if bin_score > bin_thresh else "NON-TOXIC"
    color      = "#5c0000" if verdict == "TOXIC" else "#003300"

    st.markdown(
        f'<div style="background:{color};padding:14px 18px;border-radius:8px;margin:12px 0">'
        f'<b style="font-size:1.2rem">Verdict: {verdict}</b>'
        f'<span style="float:right;opacity:.8">Score: {bin_score:.3f} / '
        f'Threshold: {bin_thresh:.2f}</span></div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Sub-Category Probabilities Breakdown")

    for name, prob in label_probs.items():
        thresh  = thresholds.get(name, MIN_THRESHOLD_FLOOR)
        flagged = prob > thresh
        c1, c2  = st.columns([1, 3])
        with c1:
            st.markdown(f"**{name.replace('_',' ').title()}**")
            st.caption("⚠️ Flagged" if flagged else "✅ Safe")
        with c2:
            st.progress(min(prob, 1.0))
            st.caption(f"Score: {prob:.3f} / Threshold: {thresh:.2f}")

    with st.expander("🔍 Cleaned input sent to model"):
        st.code(clean(user_input))
