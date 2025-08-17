# models/train_ner.py

"""
Fine-tune a multilingual transformer model (XLM-RoBERTa) for Amharic Named Entity Recognition (NER)
using your manually-labeled CoNLL file.

Before running:
1) Install dependencies with:
   pip install transformers datasets seqeval accelerate

2) Make sure you have ~1000 manually-labeled lines stored as: data/processed/labeled_data.conll
"""

from datasets import load_dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification,
                          TrainingArguments, Trainer)
from transformers import XLMRobertaTokenizerFast
import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

MODEL_NAME = "xlm-roberta-base"
DATA_PATH = "data/processed/labeled_data.conll"


def get_label_list():
    # Make sure the order matches all entity types used in labeling!
    return ["O", "B-PRODUCT", "I-PRODUCT", "B-PRICE", "I-PRICE",
            "B-PCONTACT", "I-CONTACT", "B-LINK", "I-LINK",
            "B-ADDRESS", "I-ADDRESS", "B-PHONE", "I-PHONE"]


def encode_labels(labels, label2id):
    encoded = [[label2id[t] for t in example] for example in labels]
    return encoded


def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[label_list[l] for l in label] for label in labels]
    true_preds = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds)
    }


if __name__ == "__main__":
    label_list = get_label_list()
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}

    raw_dataset = load_dataset("conll2003", data_files={'train': DATA_PATH, 'validation': DATA_PATH, 'test': DATA_PATH},
                               task="ner")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_and_align(batch):
        tokenized = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True)
        new_labels = []
        for i, labels in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned = []
            prev_word = None
            for word_id in word_ids:
                if word_id is None:
                    aligned.append(-100)
                elif word_id != prev_word:
                    aligned.append(label2id[batch["ner_tags"][i][word_id]])
                else:
                    aligned.append(label2id[batch["ner_tags"][i][word_id]] if batch["ner_tags"][i][word_id].startswith("I") else -100)
                prev_word = word_id
            new_labels.append(aligned)
        tokenized["labels"] = new_labels
        return tokenized

    encoded_dataset = raw_dataset.map(lambda x: {"tokens": x["tokens"], "ner_tags": x["ner_tags"]})
    tokenized_dataset = encoded_dataset.map(tokenize_and_align, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir="models/checkpoints",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("models/final_ner_model")
    print("Model saved → models/final_ner_model")
