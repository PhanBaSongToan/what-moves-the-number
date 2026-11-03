import copy
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SEED = 42
MODEL_DIR = r"C:\Users\ACER\Downloads\vispam_outputs\vispam_phobert_task2"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReviewDataset(Dataset):
    def __init__(self, split):
        self.ids = split["input_ids"]
        self.mask = split["attention_mask"]
        self.labels = split["spam_label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.ids[i], dtype=torch.long),
            "attention_mask": torch.tensor(self.mask[i], dtype=torch.long),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }


def collect_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            y = batch.pop("labels").numpy()
            x = {k: v.to(device) for k, v in batch.items()}
            logits.append(model(**x).logits.cpu().numpy())
            labels.append(y)
    return np.concatenate(logits), np.concatenate(labels)


def score(logits, labels, bias):
    pred = (logits + bias).argmax(axis=1)
    return f1_score(labels, pred, average="macro", zero_division=0), pred


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = load_dataset("sonlam1102/vispamdetection2")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

    def tokenize(batch):
        return tokenizer(batch["comment"], truncation=True, padding="max_length", max_length=128)

    ds = raw.map(tokenize, batched=True)
    keep = ["input_ids", "attention_mask", "spam_label"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])
    loaders = {
        name: DataLoader(ReviewDataset(ds[name]), batch_size=16)
        for name in ["validation", "test"]
    }
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    dev_logits, dev_y = collect_logits(model, loaders["validation"], device)
    test_logits, test_y = collect_logits(model, loaders["test"], device)

    zero = np.zeros(4)
    base_dev, _ = score(dev_logits, dev_y, zero)
    best_bias, best_dev = zero.copy(), base_dev
    # Coordinate search uses validation only; test remains untouched until final reporting.
    for step in [0.50, 0.25, 0.10, 0.05]:
        improved = True
        while improved:
            improved = False
            for c in range(1, 4):
                for delta in (-step, step):
                    candidate = best_bias.copy()
                    candidate[c] += delta
                    value, _ = score(dev_logits, dev_y, candidate)
                    if value > best_dev + 1e-8:
                        best_bias, best_dev = candidate, value
                        improved = True

    base_test, base_pred = score(test_logits, test_y, zero)
    tuned_test, tuned_pred = score(test_logits, test_y, best_bias)
    print("Validation baseline macro_f1=%.4f" % base_dev)
    print("Validation tuned macro_f1=%.4f" % best_dev)
    print("Learned logit bias:", np.round(best_bias, 4).tolist())
    print("Test baseline accuracy=%.4f macro_f1=%.4f" % (accuracy_score(test_y, base_pred), base_test))
    print("Test tuned accuracy=%.4f macro_f1=%.4f" % (accuracy_score(test_y, tuned_pred), tuned_test))
    print("\nTuned test report\n", classification_report(test_y, tuned_pred, zero_division=0))


if __name__ == "__main__":
    main()
