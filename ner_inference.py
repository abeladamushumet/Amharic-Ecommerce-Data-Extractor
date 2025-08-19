# ner_inference.py
import os
import logging
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# 🔹 Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔹 Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "xlm-roberta-base")
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "processed", "labeled_data.conll")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "data", "processed", "ner_output.csv")

# 🔹 Allowed labels
LABEL_LIST = [
    "O",
    "B-PRODUCT", "I-PRODUCT",
    "B-PRICE", "I-PRICE",
    "B-LOC", "I-LOC",
    "B-CONTACT-INFO", "I-CONTACT-INFO",
    "B-DELIVER-FEE", "I-DELIVER-FEE"
]

def load_dataset(file_path):
    """Load labeled dataset for inference"""
    dataset = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            sentence = []
            for line in f:
                line = line.strip()
                if not line:
                    if sentence:
                        dataset.append(" ".join([t for t, l in sentence]))
                        sentence = []
                    continue
                parts = line.split()
                if len(parts) < 2:
                    logger.warning(f"Skipping malformed line: {line}")
                    continue
                token, label = parts[0], parts[-1]
                sentence.append((token, label))
            if sentence:
                dataset.append(" ".join([t for t, l in sentence]))
        logger.info(f"✅ Dataset loaded. Total sentences: {len(dataset)}")
    except FileNotFoundError:
        logger.error(f"Dataset not found at {file_path}")
    return dataset

def main():
    # 🔹 Load tokenizer & model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model directory not found: {MODEL_PATH}")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_PATH, local_files_only=True
    )

    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )

    # 🔹 Load dataset
    sentences = load_dataset(DATA_PATH)
    if not sentences:
        logger.error("No sentences to process. Exiting.")
        return

    # 🔹 Run inference
    all_predictions = []
    for i, sentence in enumerate(sentences):
        try:
            ner_results = ner_pipeline(sentence)
            if not ner_results:
                logger.info(f"No entities found in sentence {i}: '{sentence}'")
                continue

            logger.info(f"Sentence {i} processed. Entities: {len(ner_results)}")
            for res in ner_results:
                all_predictions.append({
                    "sentence_idx": i,
                    "word": res["word"],
                    "entity": res["entity_group"],
                    "score": res["score"]
                })
        except Exception as e:
            logger.error(f"Error processing sentence {i}: {e}")

    # 🔹 Save predictions
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if all_predictions:
        df = pd.DataFrame(all_predictions)
        df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
        logger.info(f"✅ NER predictions saved to {OUTPUT_PATH}")
    else:
        logger.warning("No predictions generated. CSV not created.")

if __name__ == "__main__":
    main()
