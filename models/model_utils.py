import logging
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

logging.basicConfig(level=logging.INFO)

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

def normalize_tag(tag: str) -> str:
    """Normalize tags to match LABEL_LIST."""
    if tag == "O":
        return "O"
    return tag.upper()

def read_conll(file_path: str) -> Dataset:
    """Read CoNLL file and return Hugging Face Dataset."""
    tokens, ner_tags = [], []
    all_tokens, all_tags = [], []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    tag_ids = []
                    for tag in ner_tags:
                        tag_norm = normalize_tag(tag)
                        if tag_norm not in LABEL2ID:
                            logging.warning(f"Unknown label found: {tag}. Skipping it.")
                            tag_ids.append(LABEL2ID["O"])
                        else:
                            tag_ids.append(LABEL2ID[tag_norm])
                    all_tokens.append(tokens)
                    all_tags.append(tag_ids)
                    tokens, ner_tags = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    ner_tags.append(parts[-1])
    if tokens:
        tag_ids = [LABEL2ID[normalize_tag(tag)] if normalize_tag(tag) in LABEL2ID else LABEL2ID["O"] for tag in ner_tags]
        all_tokens.append(tokens)
        all_tags.append(tag_ids)

    return Dataset.from_dict({"tokens": all_tokens, "ner_tags": all_tags})

def load_model_and_tokenizer(model_path: str):
    """Load tokenizer and model with proper label mapping."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    return tokenizer, model

def compute_metrics(pred):
    """Compute token-level metrics for evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Flatten lists
    true_labels, true_preds = [], []
    for l, p in zip(labels, preds):
        for tl, tp in zip(l, p):
            if tl != -100:
                true_labels.append(tl)
                true_preds.append(tp)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_preds, average="weighted")
    acc = accuracy_score(true_labels, true_preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
