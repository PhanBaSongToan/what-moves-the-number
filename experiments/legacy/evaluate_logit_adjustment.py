import random
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
MODEL_DIR = r"C:\Users\ACER\Downloads\vispam_outputs\reproduction_official_v2_task2_ce_seed42"


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class Reviews(Dataset):
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


def collect(model, loader, device):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            y = batch.pop("labels").numpy()
            x = {k: v.to(device) for k, v in batch.items()}
            logits.append(model(**x).logits.cpu().numpy())
            labels.append(y)
    return np.concatenate(logits), np.concatenate(labels)


def metrics(logits, labels, adjustment):
    pred = (logits + adjustment).argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, pred),
        "macro_f1": f1_score(labels, pred, average="macro", zero_division=0),
        "pred": pred,
    }


def main():
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    raw = load_dataset("sonlam1102/vispamdetection2")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

    def tokenize(batch):
        return tokenizer(batch["comment"], truncation=True, padding="max_length", max_length=128)

    ds = raw.map(tokenize, batched=True)
    keep = ["input_ids", "attention_mask", "spam_label"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])
    dev_loader = DataLoader(Reviews(ds["validation"]), batch_size=32)
    test_loader = DataLoader(Reviews(ds["test"]), batch_size=32)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    dev_logits, dev_y = collect(model, dev_loader, device)
    test_logits, test_y = collect(model, test_loader, device)

    counts = np.bincount(np.asarray(raw["train"]["spam_label"]), minlength=4).astype(float)
    priors = counts / counts.sum()
    print("Train counts:", counts.astype(int).tolist())
    print("Train priors:", np.round(priors, 6).tolist())

    best_tau, best_dev = 0.0, -1.0
    for tau in np.arange(0.0, 1.01, 0.05):
        # Rare classes receive a positive correction relative to frequent classes.
        adjustment = -tau * np.log(priors)
        adjustment -= adjustment[0]
        result = metrics(dev_logits, dev_y, adjustment)
        print(f"tau={tau:.2f} dev_accuracy={result['accuracy']:.4f} dev_macro_f1={result['macro_f1']:.4f}")
        if result["macro_f1"] > best_dev:
            best_dev, best_tau = result["macro_f1"], float(tau)

    base = metrics(test_logits, test_y, np.zeros(4))
    adjustment = -best_tau * np.log(priors)
    adjustment -= adjustment[0]
    tuned = metrics(test_logits, test_y, adjustment)
    print("\nSelected tau=%.2f by validation Macro-F1=%.4f" % (best_tau, best_dev))
    print("Test baseline accuracy=%.4f macro_f1=%.4f" % (base["accuracy"], base["macro_f1"]))
    print("Test adjusted accuracy=%.4f macro_f1=%.4f" % (tuned["accuracy"], tuned["macro_f1"]))
    print("Adjustment:", np.round(adjustment, 4).tolist())
    print("\nAdjusted test report\n", classification_report(test_y, tuned["pred"], zero_division=0))


if __name__ == "__main__":
    main()
