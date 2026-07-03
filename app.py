import streamlit as st, torch, json, re, os, urllib.request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# --- 1. MUST BE THE ABSOLUTE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Toxic Classifier", page_icon="🛡️")

# --- 2. CONFIGURATIONS & CONSTANTS ---
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_LEN    = 128
LOCAL_MODEL_DIR = "./toxic_bert_model"
BASE_MODEL = "microsoft/deberta-v3-base"

@st.cache_resource
def load_assets():
    """Downloads fine-tuned weights dynamically if missing, then loads all assets."""
    if not os.path.exists(LOCAL_MODEL_DIR):
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        
    # --- HANDLES LARGE WEIGHTS FILE ---
    # TODO: Upload your 'model.safetensors' to a cloud link and paste the direct download URL below:
    WEIGHTS_URL = https://huggingface.co/abishekw412001/toxic-deberta-v5/resolve/main/best_checkpoint.pt?download=true
    local_weights_path = os.path.join(LOCAL_MODEL_DIR, "model.safetensors")
    
    if not os.path.exists(local_weights_path) and "your-cloud-storage" not in WEIGHTS_URL:
        with st.spinner("Downloading your fine-tuned model weights from cloud storage... Please wait."):
            urllib.request.urlretrieve(WEIGHTS_URL, local_weights_path)

    # --- LOADING ALL MODEL COMPONENT GRAPH ARTIFACTS ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
    try:
        model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR).to("cpu")
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=6).to("cpu")
        
    model.eval()
    
    threshold_path = "thresholds.json"
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            thresholds = json.load(f)
    else:
        thresholds = {name: 0.5 for name in LABEL_COLS}
        thresholds["binary"] = 0.5
        
    return tokenizer, model, thresholds

tokenizer, model, thresholds = load_assets()

# --- Text Cleaning Engine ---
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
    enc = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=MAX_LEN).to("cpu")
    logits = model(**enc).logits[0]
    probs = torch.sigmoid(logits).numpy()
    bin_score = float(np.max(probs))
    return bin_score, {name: float(p) for name, p in zip(LABEL_COLS, probs)}

# --- Streamlit Presentation Layer ---
st.title("🛡️ Toxic Comment Classifier")
st.write("Enter a comment below to evaluate multi-label classification predictions.")

user_input = st.text_area("Comment Content Evaluation Window", height=120, placeholder="Type your text here...")
if st.button("Analyse Text Content", type="primary") and user_input.strip():
    bin_score, label_probs = predict(user_input)
    
    binary_thresh = thresholds.get("binary", 0.5)
    verdict = "TOXIC" if bin_score > binary_thresh else "NON-TOXIC"
    
    if verdict == "TOXIC":
        st.error(f"### Verdict: {verdict} (Score: {bin_score:.3f})")
    else:
        st.success(f"### Verdict: {verdict} (Score: {bin_score:.3f})")
        
    st.markdown("---")
    st.subheader("Sub-Category Probabilities Breakdown")
    for name, p in label_probs.items():
        tuned_t = thresholds.get(name, 0.5)
        flagged = p > tuned_t
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"**{name.title()}**")
            st.caption("⚠️ Flagged" if flagged else "✅ Safe")
        with col2:
            st.progress(p)
            st.caption(f"Score: {p:.3f} / Threshold: {tuned_t:.2f}")
