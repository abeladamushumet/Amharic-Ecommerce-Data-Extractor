# interpretability/lime_explainer.py

import os
import logging
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from lime.lime_text import LimeTextExplainer
import numpy as np

logging.basicConfig(level=logging.INFO)

# Force CPU usage to avoid CUDA memory issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "xlm-roberta-base")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "data", "processed")

# Label mapping (must match your model)
LABEL_LIST = [
    "O",
    "B-PRODUCT", "I-PRODUCT",
    "B-PRICE", "I-PRICE",
    "B-LOC", "I-LOC",
    "B-CONTACT-INFO", "I-CONTACT-INFO",
    "B-DELIVER-FEE", "I-DELIVER-FEE"
]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

# Global variables
tokenizer = None
model = None

# --- Functions ---
def model_predict(texts):
    """Return probabilities for LIME explainer"""
    global tokenizer, model
    
    if isinstance(texts, str):
        texts = [texts]
    texts = [str(t) for t in texts]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"]
        )
    
    # Get softmax probabilities (batch, seq_len, num_labels)
    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
    
    # For LIME, we need batch x num_labels -> average over tokens
    avg_probs = probs.mean(axis=1)
    
    return avg_probs

def analyze_sentence_with_lime(sentence):
    """Use LIME to explain a single sentence"""
    explainer = LimeTextExplainer(class_names=LABEL_LIST)
    
    try:
        exp = explainer.explain_instance(
            sentence,
            model_predict,
            num_features=20,
            labels=list(range(len(LABEL_LIST)))
        )
        return exp
    except Exception as e:
        logging.error(f"Error in LIME explanation: {e}")
        return None

def save_lime_explanation(exp, sentence, sentence_idx, output_dir):
    """Save LIME explanation to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        weights = exp.as_list()
        df_data = []
        for token, weight in weights:
            df_data.append({
                "token": token,
                "weight": weight,
                "sentence": sentence
            })
        df = pd.DataFrame(df_data)
        output_path = os.path.join(output_dir, f"lime_sentence_{sentence_idx+1}.csv")
        df.to_csv(output_path, index=False)
        logging.info(f"Saved LIME explanation to {output_path}")
    except Exception as e:
        logging.error(f"Error saving LIME CSV: {e}")

# --- Main ---
if __name__ == "__main__":
    logging.info("🔹 Loading tokenizer and model...")
    
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model directory not found: {MODEL_PATH}")
        raise FileNotFoundError(f"Could not find model directory. Please check the path.")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_PATH,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        local_files_only=True
    )
    model.to("cpu")
    model.eval()
    
    # Example sentences
    sentence_strings = [
        "እቃ ዋጋ 1500 ብር",
        "ማስቀመጫ ዋጋ 2000 ብር"
    ]
    
    logging.info("🔹 Running LIME explanations...")
    for i, sentence in enumerate(sentence_strings):
        exp = analyze_sentence_with_lime(sentence)
        if exp:
            save_lime_explanation(exp, sentence, i, OUTPUT_DIR)
    
    logging.info("✅ LIME analysis completed successfully!")
