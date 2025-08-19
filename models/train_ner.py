from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import logging
from huggingface_hub.utils import RepositoryNotFoundError

logging.basicConfig(level=logging.INFO)

# Paths
DATA_PATH = "/content/drive/MyDrive/Colab Notebooks/Amharic-Ecommerce-Data-Extractor/data/processed/labeled_data.conll"

# Label list
label_list = [
    "O",
    "B-PRODUCT", "I-PRODUCT",
    "B-PRICE", "I-PRICE",
    "B-LOC", "I-LOC",
    "B-CONTACT-INFO", "I-CONTACT-INFO",
    "B-DELIVER-FEE", "I-DELIVER-FEE"
]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# Models to train
MODELS = [
    "xlm-roberta-base",
    "Davlan/bert-base-multilingual-cased",
    "afroxmlr"
]

def load_conll_data(file_path):
    sentences, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        words, tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words, tags = [], []
            else:
                splits = line.split()
                if len(splits) < 2:
                    print(f"[WARNING] Skipping malformed line: {line}")
                    continue
                words.append(splits[0])
                tags.append(splits[1].upper().replace("_", "-"))
        if words:
            sentences.append(words)
            labels.append(tags)
    return Dataset.from_dict({
        "tokens": sentences,
        "ner_tags": [[label2id[tag] for tag in seq] for seq in labels]
    })

dataset = load_conll_data(DATA_PATH)
train_test = dataset.train_test_split(test_size=0.1)
train_dataset = train_test["train"]
val_dataset = train_test["test"]

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=True
    )
    aligned_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(labels[word_idx] if str(label_list[labels[word_idx]]).startswith("I-") else -100)
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)
    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

def train_model(model_name, output_dir):
    print(f"\n🔹 Training {model_name}...\n")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id
        )
    except (OSError, RepositoryNotFoundError) as e:
        logging.warning(f"Skipping {model_name}: {e}")
        return

    tokenized_train = train_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    tokenized_val = val_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

    tokenized_train = tokenized_train.remove_columns(['tokens', 'ner_tags'])
    tokenized_val = tokenized_val.remove_columns(['tokens', 'ner_tags'])

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"✅ Training completed for {model_name} and saved to {output_dir}")

for model_name in MODELS:
    model_folder = f"/content/drive/MyDrive/Colab Notebooks/Amharic-Ecommerce-Data-Extractor/models/{model_name.replace('/', '-')}"
    train_model(model_name, model_folder)
