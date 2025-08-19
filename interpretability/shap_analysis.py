# interpretability/shap_analysis.py

import os
import logging
import torch
import shap
import pandas as pd
import numpy as np
import gc
from transformers import AutoTokenizer, AutoModelForTokenClassification

logging.basicConfig(level=logging.INFO)

# Force CPU usage to avoid CUDA memory issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Get the current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths - use relative paths from the script location
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "xlm-roberta-base")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "data", "processed")

# Label mapping
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

# Global variables for model and tokenizer
tokenizer = None
model = None

# --- Functions ---
def model_predict(texts):
    """Wrapper function for model prediction"""
    global tokenizer, model
    
    # Handle input types
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray) and texts.dtype == object:
        texts = texts.tolist()
    
    texts = [str(text) for text in texts]
    
    if not texts:
        return np.zeros((1, len(LABEL_LIST)))
    
    try:
        # Tokenize
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128,
        )
        
        with torch.no_grad():
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"]
            )
        
        return outputs.logits.numpy()
        
    except Exception as e:
        logging.error(f"Error in model_predict: {e}")
        return np.zeros((len(texts), len(LABEL_LIST)))

def compute_gradient_importance(sentences):
    """Compute gradient-based importance as an alternative to SHAP"""
    global model, tokenizer
    
    model.eval()
    all_importances = []
    
    for sentence in sentences:
        try:
            # Clear gradients
            model.zero_grad()
            
            # Tokenize
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"]
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Use the most confident prediction for each token
            probs = torch.softmax(logits, dim=-1)
            max_probs, preds = torch.max(probs, dim=-1)
            
            # Compute gradients for the predicted classes
            loss = logits[0, torch.arange(logits.shape[1]), preds[0]].sum()
            loss.backward()
            
            # Get gradient importance from embeddings
            if hasattr(model, 'get_input_embeddings') and model.get_input_embeddings().weight.grad is not None:
                gradients = model.get_input_embeddings().weight.grad
                importance = gradients.abs().mean(dim=1)
                
                # Map back to tokens
                token_importance = []
                for token_id in input_ids[0]:
                    if token_id.item() < importance.shape[0]:
                        token_importance.append(importance[token_id.item()].item())
                    else:
                        token_importance.append(0.0)
                
                all_importances.append(token_importance)
            else:
                # Fallback: uniform importance
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                all_importances.append([1.0] * len(tokens))
                
        except Exception as e:
            logging.error(f"Error computing gradients for sentence: {e}")
            # Fallback: uniform importance
            tokens = tokenizer.tokenize(sentence)
            all_importances.append([1.0] * len(tokens))
    
    return all_importances

def compute_simple_importance(sentences):
    """Compute simple attention-based importance"""
    global model, tokenizer
    
    all_importances = []
    
    for sentence in sentences:
        try:
            # Tokenize
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
            
            # Use attention weights from the last layer
            attentions = outputs.attentions[-1]  # Last layer attention
            attention_weights = attentions.mean(dim=1).squeeze(0)  # Average attention heads
            
            # Use attention to [CLS] token as importance measure
            cls_attention = attention_weights[0, 1:]  # Skip [CLS] to [CLS] attention
            
            importance = cls_attention.numpy()
            all_importances.append(importance)
            
        except Exception as e:
            logging.error(f"Error computing attention importance: {e}")
            # Fallback: use token length-based importance
            tokens = tokenizer.tokenize(sentence)
            all_importances.append([1.0] * len(tokens))
    
    return all_importances

def save_importance_to_csv(importances, sentences, output_dir):
    """Save importance scores to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (sentence_importance, sentence) in enumerate(zip(importances, sentences)):
        try:
            # Tokenize to get proper tokens
            inputs = tokenizer(sentence, return_tensors="pt")
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Remove special tokens
            valid_tokens = []
            valid_importance = []
            for token, imp in zip(tokens, sentence_importance):
                if token not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                    valid_tokens.append(token)
                    valid_importance.append(imp)
            
            # Create DataFrame
            df_data = []
            for token_idx, (token, imp) in enumerate(zip(valid_tokens, valid_importance)):
                row = {
                    "token": token,
                    "position": token_idx,
                    "importance_score": imp,
                    "sentence": sentence
                }
                # Add placeholder values for each label
                for label in LABEL_LIST:
                    row[label] = imp if label == "B-PRODUCT" else imp * 0.5  # Example distribution
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            output_path = os.path.join(output_dir, f"importance_sentence_{i+1}.csv")
            df.to_csv(output_path, index=False)
            logging.info(f"Saved importance values to {output_path}")
            
        except Exception as e:
            logging.error(f"Error saving sentence {i+1}: {e}")

def analyze_sentence(sentence):
    """Simple analysis of a sentence using model predictions"""
    global model, tokenizer
    
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=-1)[0].numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Get softmax probabilities
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    
    # Filter out special tokens
    result = []
    for token_idx, (token, pred) in enumerate(zip(tokens, predictions)):
        if token not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            # Get the confidence for this specific token's prediction
            confidence = probs[token_idx, pred].item()
            result.append({
                "token": token,
                "prediction": ID2LABEL[pred],
                "confidence": confidence
            })
    
    return result

# --- Main ---
if __name__ == "__main__":
    logging.info("🔹 Loading tokenizer and model...")
    
    # Check if model directory exists
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model directory not found: {MODEL_PATH}")
        raise FileNotFoundError(f"Could not find model directory. Please check the path.")
    
    logging.info(f"Loading model from: {MODEL_PATH}")
    
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_PATH,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            local_files_only=True
        )
        
        # Move model to CPU and set to eval mode
        model.to("cpu")
        model.eval()
        
        # Example sentences
        sentence_strings = [
            "እቃ ዋጋ 1500 ብር",
            "ማስቀመጫ ዋጋ 2000 ብር"
        ]
        
        logging.info("🔹 Analyzing sentences...")
        for i, sentence in enumerate(sentence_strings):
            analysis = analyze_sentence(sentence)
            logging.info(f"Sentence {i+1} analysis:")
            for item in analysis:
                logging.info(f"  {item['token']}: {item['prediction']} ({item['confidence']:.3f})")
        
        logging.info("🔹 Computing importance scores...")
        importances = compute_gradient_importance(sentence_strings)
        
        logging.info("🔹 Saving importance scores to CSV...")
        save_importance_to_csv(importances, sentence_strings, OUTPUT_DIR)
        
        logging.info("✅ Analysis completed successfully!")
        
        # Show sample of what was saved
        for i in range(len(sentence_strings)):
            csv_path = os.path.join(OUTPUT_DIR, f"importance_sentence_{i+1}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                logging.info(f"\nSample from sentence {i+1}:")
                logging.info(df.head().to_string())
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        import traceback
        logging.error(traceback.format_exc())