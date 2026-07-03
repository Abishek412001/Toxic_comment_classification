@st.cache_resource
def load_assets():
    """Downloads fine-tuned weights dynamically if missing, then loads all assets."""
    if not os.path.exists(LOCAL_MODEL_DIR):
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        
    # --- 1. DOWNLOAD THE WEIGHTS ---
    WEIGHTS_URL = "https://huggingface.co/abishekw412001/toxic-deberta-v5/resolve/main/best_checkpoint.pt?download=true"
    local_weights_path = os.path.join(LOCAL_MODEL_DIR, "best_checkpoint.pt")
    
    if not os.path.exists(local_weights_path):
        with st.spinner("Downloading your custom fine-tuned weights matrix... Please wait."):
            urllib.request.urlretrieve(WEIGHTS_URL, local_weights_path)

    # --- 2. LOAD TOKENIZER FROM HUGGING FACE DIRECTLY ---
    # This grabs the official microsoft config parameters straight from the web
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
    # --- 3. BUILD ARCHITECTURE & LOAD CUSTOM WEIGHTS ---
    # Create the baseline empty DeBERTa structure with 6 labels
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=6)
    
    # Load your custom fine-tuned brain parameters safely into the structure
    custom_state_dict = torch.load(local_weights_path, map_location="cpu")
    model.load_state_dict(custom_state_dict)
    model.to("cpu")
    model.eval()
    
    # --- 4. LOAD THRESHOLDS ---
    threshold_path = "thresholds.json"
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            thresholds = json.load(f)
    else:
        thresholds = {name: 0.5 for name in LABEL_COLS}
        thresholds["binary"] = 0.5
        
    return tokenizer, model, thresholds
