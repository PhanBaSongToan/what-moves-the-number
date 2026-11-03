import json
import uuid
from pathlib import Path

OUT = Path(r"C:\Users\ACER\Code\TMG301\SST2_DistilBERT_Sentiment_Analysis.ipynb")

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "id": uuid.uuid4().hex[:8], "source": text.splitlines(True)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "id": uuid.uuid4().hex[:8], "outputs": [], "source": text.splitlines(True)}

cells = [
md(r'''
# Fine-Tuning and Evaluating DistilBERT for SST-2 Sentiment Classification

Student name: ____________________  
Student ID: ____________________

This notebook follows the practical exam specification:

- Dataset: stanfordnlp/sst2
- Split: 1,000 train, 200 development, 200 final evaluation
- Baseline: TF-IDF with max_features=5000 plus Logistic Regression
- Transformer: distilbert/distilbert-base-uncased, fine-tuned for one epoch
- Final metrics: Accuracy, Macro Precision, Macro Recall, Macro F1, confusion matrices
- Tokenization output, at least three errors, and a 100-150 word conclusion

Leakage control: the baseline is fit only on the training subset, development is used for monitoring, and final evaluation is used only after the models are fixed.
'''),
md(r'''
## 0. Environment setup

This cell installs only packages that are missing. On Colab it may take a short time on the first run.
'''),
code(r'''
import importlib.util
import subprocess
import sys

required = {
    "datasets": "datasets",
    "transformers": "transformers",
    "torch": "torch",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "accelerate": "accelerate",
}
missing = [package for module, package in required.items()
           if importlib.util.find_spec(module) is None]
if missing:
    print("Installing missing packages:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
else:
    print("All required packages are already installed.")
'''),
code(r'''
import inspect
import os
import random
import re
import warnings

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 42
CHECKPOINT = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Environment ready")
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
'''),
md(r'''
## 1. Dataset preparation

SST-2 contains English movie-review sentences with binary labels:

- 0 = negative
- 1 = positive

The official test split has no usable gold labels for this exam, so the official validation split supplies the final evaluation subset.
'''),
code(r'''
ds = load_dataset("stanfordnlp/sst2")
print(ds)
print("Official splits:", {split: len(ds[split]) for split in ds})
'''),
code(r'''
train_pool = ds["train"].shuffle(seed=SEED).select(range(1200))
train_ds = train_pool.select(range(1000))
dev_ds = train_pool.select(range(1000, 1200))
eval_ds = ds["validation"].shuffle(seed=SEED).select(range(200))

assert len(train_ds) == 1000
assert len(dev_ds) == 200
assert len(eval_ds) == 200

def distribution(dataset):
    counts = pd.Series(dataset["label"]).value_counts().sort_index()
    return {
        "negative_0": int(counts.get(0, 0)),
        "positive_1": int(counts.get(1, 0)),
    }

summary = pd.DataFrame([
    {"subset": "train", "n": len(train_ds), **distribution(train_ds)},
    {"subset": "development", "n": len(dev_ds), **distribution(dev_ds)},
    {"subset": "final evaluation", "n": len(eval_ds), **distribution(eval_ds)},
])
display(summary)
print("Five training examples:")
display(pd.DataFrame(train_ds.select(range(5))))
'''),
md(r'''
### Split protocol

The development subset supports model monitoring and checkpoint decisions. The final evaluation subset stays untouched until both models and all hyperparameters are fixed. This prevents optimistic evaluation.
'''),
md(r'''
## 2. Task 1 - Tokenization

DistilBERT uses subword tokenization. The tokenizer creates input_ids and attention_mask; truncation limits sequences to 128 tokens, while dynamic padding pads only within each batch.
'''),
code(r'''
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

def tokenize(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

tokenized_train = train_ds.map(tokenize, batched=True)
tokenized_dev = dev_ds.map(tokenize, batched=True)
tokenized_eval = eval_ds.map(tokenize, batched=True)

if "label" in tokenized_train.column_names:
    tokenized_train = tokenized_train.rename_column("label", "labels")
if "label" in tokenized_dev.column_names:
    tokenized_dev = tokenized_dev.rename_column("label", "labels")
if "label" in tokenized_eval.column_names:
    tokenized_eval = tokenized_eval.rename_column("label", "labels")

sample_text = train_ds[0]["sentence"]
sample_encoded = tokenizer(sample_text, truncation=True, max_length=MAX_LENGTH)

print("Original text:")
print(sample_text)
print("\nSubword tokens:")
print(tokenizer.convert_ids_to_tokens(sample_encoded["input_ids"]))
print("\ninput_ids:")
print(sample_encoded["input_ids"])
print("\nattention_mask:")
print(sample_encoded["attention_mask"])
'''),
md(r'''
### Tokenization explanation

Subword tokenization breaks uncommon or unseen words into smaller pieces, reducing the unknown-word problem. input_ids stores the integer vocabulary IDs that DistilBERT consumes. attention_mask marks real tokens with 1 and padding positions with 0, so attention ignores padding. Truncation cuts sentences longer than 128 tokens to satisfy the model maximum, while the dynamic collator pads examples only to the longest sequence in each batch.
'''),
md(r'''
## 3. Task 2 - Traditional baseline

The baseline uses the required pipeline: TfidfVectorizer(max_features=5000) followed by LogisticRegression(max_iter=1000). The vectorizer learns its vocabulary from the training subset only.
'''),
code(r'''
baseline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
])

baseline.fit(train_ds["sentence"], train_ds["label"])
dev_base_pred = baseline.predict(dev_ds["sentence"])
dev_base_accuracy = accuracy_score(dev_ds["label"], dev_base_pred)
dev_base_macro_f1 = f1_score(dev_ds["label"], dev_base_pred, average="macro")

baseline_dev_metrics = pd.DataFrame([{
    "model": "TF-IDF + Logistic Regression",
    "development_accuracy": dev_base_accuracy,
    "development_macro_f1": dev_base_macro_f1,
}])
display(baseline_dev_metrics.round(4))
print("Limitation: TF-IDF emphasizes lexical frequency and overlap, but does not model context and word order as deeply as a Transformer.")
'''),
md(r'''
## 4. Task 3 - Fine-tune DistilBERT

The classification head maps the final DistilBERT representation to two logits: negative and positive. The head can start randomly initialized; fine-tuning learns it together with the pre-trained encoder.
'''),
code(r'''
model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT,
    num_labels=2,
)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    guesses = np.argmax(predictions, axis=-1)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels, guesses, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, guesses),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
    }

ta_params = inspect.signature(TrainingArguments.__init__).parameters
eval_argument = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"

training_kwargs = {
    "output_dir": "./sst2_distilbert_exam",
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 1,
    "save_strategy": "no",
    "report_to": "none",
    "seed": SEED,
    "data_seed": SEED,
    "logging_strategy": "steps",
    "logging_steps": 50,
    "fp16": bool(torch.cuda.is_available()),
}
training_kwargs[eval_argument] = "epoch"
args = TrainingArguments(**training_kwargs)

trainer_kwargs = {
    "model": model,
    "args": args,
    "train_dataset": tokenized_train,
    "eval_dataset": tokenized_dev,
    "data_collator": collator,
    "compute_metrics": compute_metrics,
}
trainer_params = inspect.signature(Trainer.__init__).parameters
if "processing_class" in trainer_params:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = Trainer(**trainer_kwargs)
train_output = trainer.train()

print("Training metrics:")
print(train_output.metrics)
print("\nDevelopment metrics:")
dev_transformer_metrics = trainer.evaluate()
print(dev_transformer_metrics)

trainer.save_model("./final_distilbert")
tokenizer.save_pretrained("./final_distilbert")
print("Saved model and tokenizer to ./final_distilbert")
'''),
md(r'''
### Short-answer responses

Pre-training learns general language representations from a large corpus; fine-tuning updates the model on a labeled task-specific dataset.

DistilBERT produces contextual representations, but the classification head converts the representation into two task-specific logits.

Self-attention lets each token weigh other tokens in the sentence, allowing the model to represent context such as negation and long-range relationships.
'''),
md(r'''
## 5. Task 4 - Final evaluation

The following cell evaluates both trained models exactly once on the held-out final evaluation subset. Precision, recall and F1 all use macro averaging.
'''),
code(r'''
def macro_metrics(y_true, y_pred, model_name):
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": macro_f1,
    }

base_eval_pred = baseline.predict(eval_ds["sentence"])
transformer_output = trainer.predict(tokenized_eval)
transformer_logits = transformer_output.predictions
if isinstance(transformer_logits, tuple):
    transformer_logits = transformer_logits[0]
bert_eval_pred = np.argmax(transformer_logits, axis=-1)

y_eval = np.asarray(eval_ds["label"])
final_results = pd.DataFrame([
    macro_metrics(y_eval, base_eval_pred, "TF-IDF + Logistic Regression"),
    macro_metrics(y_eval, bert_eval_pred, "Fine-tuned DistilBERT"),
])
display(final_results.round(4))

print("Baseline classification report")
print(classification_report(
    y_eval, base_eval_pred,
    target_names=["Negative", "Positive"],
    digits=3,
    zero_division=0,
))
print("DistilBERT classification report")
print(classification_report(
    y_eval, bert_eval_pred,
    target_names=["Negative", "Positive"],
    digits=3,
    zero_division=0,
))
'''),
code(r'''
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, name, pred in [
    (axes[0], "TF-IDF + Logistic Regression", base_eval_pred),
    (axes[1], "Fine-tuned DistilBERT", bert_eval_pred),
]:
    cm = confusion_matrix(y_eval, pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"], rotation=0)
plt.tight_layout()
plt.show()
'''),
md(r'''
## 6. Error analysis

At least three misclassified final-evaluation examples are displayed. The explanation is a hypothesis based on the sentence, not an additional model label.
'''),
code(r'''
def plausible_explanation(sentence):
    text = sentence.lower()
    if re.search(r"\b(not|no|never|neither|nor|n't)\b", text):
        return "Negation may reverse or weaken the sentiment cue."
    if any(word in text for word in ["but", "although", "however", "despite"]):
        return "The sentence contains contrast or mixed sentiment."
    if "?" in sentence:
        return "The sentence is phrased as a question and may lack explicit sentiment."
    if len(text.split()) <= 6:
        return "The sentence is short, so it provides limited contextual evidence."
    return "The wording may be ambiguous or require movie context beyond the sentence."

error_frame = pd.DataFrame({
    "sentence": eval_ds["sentence"],
    "true_label": y_eval,
    "baseline_prediction": base_eval_pred,
    "distilbert_prediction": bert_eval_pred,
})
error_frame["baseline_wrong"] = error_frame.true_label != error_frame.baseline_prediction
error_frame["distilbert_wrong"] = error_frame.true_label != error_frame.distilbert_prediction

all_errors = error_frame[
    error_frame.baseline_wrong | error_frame.distilbert_wrong
].copy()
all_errors["explanation"] = all_errors.sentence.map(plausible_explanation)
assert len(all_errors) >= 3, "At least three errors are required for the analysis."

print("Three or more misclassified examples:")
display(all_errors.head(3)[[
    "sentence",
    "true_label",
    "baseline_prediction",
    "distilbert_prediction",
    "explanation",
]])

unique_baseline_error = error_frame[
    error_frame.baseline_wrong & ~error_frame.distilbert_wrong
]
unique_distilbert_error = error_frame[
    error_frame.distilbert_wrong & ~error_frame.baseline_wrong
]

if len(unique_baseline_error):
    print("Example: baseline wrong, DistilBERT correct")
    display(unique_baseline_error.head(1)[[
        "sentence", "true_label",
        "baseline_prediction", "distilbert_prediction",
    ]])
elif len(unique_distilbert_error):
    print("Example: DistilBERT wrong, baseline correct")
    display(unique_distilbert_error.head(1)[[
        "sentence", "true_label",
        "baseline_prediction", "distilbert_prediction",
    ]])
else:
    print("No unique-error example was found in this final subset.")
'''),
md(r'''
### Discussion

A pre-trained Transformer may outperform TF-IDF because self-attention and pre-training provide contextual representations before task-specific training. However, higher accuracy alone does not prove that a model is better: macro precision, macro recall, Macro F1, the confusion matrices and qualitative error analysis must also be considered. The evaluation subset contains only 200 examples, so small metric differences should be interpreted cautiously.
'''),
md(r'''
## 7. Conclusion

The next cell generates a 100-150 word English conclusion from the actual final metrics.
'''),
code(r'''
base_row = final_results.iloc[0]
bert_row = final_results.iloc[1]

conclusion = f"""This experiment compared a traditional TF-IDF plus Logistic Regression baseline with a fine-tuned DistilBERT classifier for binary sentiment analysis on SST-2. The baseline achieved {base_row["Accuracy"]:.3f} accuracy and {base_row["Macro F1"]:.3f} Macro F1 on the held-out evaluation subset, while DistilBERT achieved {bert_row["Accuracy"]:.3f} accuracy and {bert_row["Macro F1"]:.3f} Macro F1. The Transformer benefits from transfer learning because it starts with language representations learned from a large corpus and then adapts them to sentiment classification. Its self-attention mechanism also represents relationships between words more effectively than TF-IDF, especially when context or negation matters. Nevertheless, fine-tuning requires downloading a large checkpoint, compatible libraries, and preferably GPU resources. The small evaluation subset makes small metric differences uncertain. Error analysis remains important because a higher aggregate score may hide mistakes on ambiguous or mixed-sentiment sentences. Overall, the comparison demonstrates the practical trade-off between an efficient lexical baseline and a more context-aware pre-trained language model."""
print(conclusion)
print("Word count:", len(conclusion.split()))
assert 100 <= len(conclusion.split()) <= 150
'''),
md(r'''
## 8. Submission checklist

- [x] SST-2 loaded from the required Hugging Face dataset.
- [x] Seed 42 and prescribed 1,000/200/200 split.
- [x] Five training examples and label distributions shown.
- [x] DistilBERT tokenizer output shown: tokens, input_ids, attention_mask.
- [x] Explanation of subword tokenization, padding and truncation included.
- [x] Required TF-IDF + Logistic Regression baseline trained only on train.
- [x] DistilBERT loaded from the required base checkpoint.
- [x] Dynamic padding, learning rate 2e-5, batch size 8, one epoch.
- [x] Development metrics reported.
- [x] Both models evaluated on the held-out final subset.
- [x] Accuracy, Macro Precision, Macro Recall, Macro F1 and confusion matrices included.
- [x] At least three misclassified examples analyzed.
- [x] A cross-model error comparison included when available.
- [x] English conclusion contains 100-150 words and uses actual output metrics.
- [x] Fine-tuned model and tokenizer saved under ./final_distilbert.
'''),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT)

