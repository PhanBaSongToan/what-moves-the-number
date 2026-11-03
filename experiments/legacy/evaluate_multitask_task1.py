import json
import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader

from train_multitask import (
    OUTPUT_DIR,
    Reviews,
    MultiTaskPhoBERT,
    load_official,
    seed_everything,
)
from transformers import AutoTokenizer


def main():
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = load_official()
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)

    def tok(batch):
        return tokenizer(batch["comment"], truncation=True, padding="max_length", max_length=100)

    ds = raw.map(tok, batched=True)
    keep = ["input_ids", "attention_mask", "spam_label"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])
    model = MultiTaskPhoBERT().to(device)
    state_path = os.path.join(OUTPUT_DIR, "best_model_state.pt")
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()

    results = {}
    for split_name in ["validation", "test"]:
        loader = DataLoader(Reviews(ds[split_name]), batch_size=16)
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in loader:
                binary_y = batch.pop("binary_labels")
                batch.pop("type_labels")
                batch.pop("labels")
                x = {k: v.to(device) for k, v in batch.items()}
                binary_logits, _ = model(**x)
                y_pred.extend(binary_logits.argmax(1).cpu().tolist())
                y_true.extend(binary_y.tolist())
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "report": classification_report(y_true, y_pred, zero_division=0),
        }
        results[split_name] = metrics
        print(split_name, "accuracy=%.4f macro_f1=%.4f" % (metrics["accuracy"], metrics["macro_f1"]))
        print(metrics["report"])

    out_path = os.path.join(OUTPUT_DIR, "task1_binary_head_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
