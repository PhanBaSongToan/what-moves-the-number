import random
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
WEIGHTED_DIR = r"C:\Users\ACER\Downloads\vispam_outputs\vispam_phobert_task2"
CE_DIR = r"C:\Users\ACER\Downloads\vispam_outputs\vispam_phobert_task2_ce"


class Reviews(Dataset):
    def __init__(self, split):
        self.ids, self.mask, self.labels = split["input_ids"], split["attention_mask"], split["spam_label"]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {"input_ids": torch.tensor(self.ids[i], dtype=torch.long),
                "attention_mask": torch.tensor(self.mask[i], dtype=torch.long),
                "labels": torch.tensor(self.labels[i], dtype=torch.long)}


@torch.no_grad()
def collect(model, loader, device):
    model.eval(); logits, labels = [], []
    for batch in loader:
        y = batch.pop("labels").numpy()
        x = {k: v.to(device) for k, v in batch.items()}
        logits.append(model(**x).logits.cpu().numpy()); labels.append(y)
    return np.concatenate(logits), np.concatenate(labels)


def predict(logits, bias): return (logits + bias).argmax(1)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = load_dataset("sonlam1102/vispamdetection2")
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTED_DIR, use_fast=False)
    def tok(batch): return tokenizer(batch["comment"], truncation=True, padding="max_length", max_length=128)
    ds = raw.map(tok, batched=True)
    keep = ["input_ids", "attention_mask", "spam_label"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])
    dev_loader = DataLoader(Reviews(ds["validation"]), batch_size=32)
    test_loader = DataLoader(Reviews(ds["test"]), batch_size=32)
    weighted = AutoModelForSequenceClassification.from_pretrained(WEIGHTED_DIR).to(device)
    ce = AutoModelForSequenceClassification.from_pretrained(CE_DIR).to(device)
    wl_dev, y_dev = collect(weighted, dev_loader, device); ce_dev, _ = collect(ce, dev_loader, device)
    wl_test, y_test = collect(weighted, test_loader, device); ce_test, _ = collect(ce, test_loader, device)
    counts = np.bincount(np.asarray(raw["train"]["spam_label"]), minlength=4).astype(float)
    priors = counts / counts.sum()
    best = (-1, None)
    for alpha in np.arange(0, 1.01, .05):
        mix_dev = alpha * wl_dev + (1 - alpha) * ce_dev
        for tau in np.arange(0, .51, .05):
            bias = -tau * np.log(priors); bias -= bias[0]
            pred = predict(mix_dev, bias)
            f = f1_score(y_dev, pred, average="macro", zero_division=0)
            if f > best[0]: best = (f, float(alpha), float(tau))
    _, alpha, tau = best
    bias = -tau * np.log(priors); bias -= bias[0]
    dev_pred = predict(alpha * wl_dev + (1-alpha) * ce_dev, bias)
    test_pred = predict(alpha * wl_test + (1-alpha) * ce_test, bias)
    print("Selected alpha=%.2f tau=%.2f" % (alpha, tau))
    print("Validation accuracy=%.4f macro_f1=%.4f" % (accuracy_score(y_dev, dev_pred), f1_score(y_dev, dev_pred, average="macro", zero_division=0)))
    print("Test accuracy=%.4f macro_f1=%.4f" % (accuracy_score(y_test, test_pred), f1_score(y_test, test_pred, average="macro", zero_division=0)))
    print(classification_report(y_test, test_pred, zero_division=0))


if __name__ == "__main__": main()
