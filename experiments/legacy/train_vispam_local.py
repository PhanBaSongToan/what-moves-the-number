import argparse
import os
import random
import zipfile

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=[1, 2], default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--data-zip", type=str, default=None)
    parser.add_argument(
        "--loss",
        choices=["ce", "weighted", "focal", "cb"],
        default="weighted",
    )
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_column = "label" if args.task == 1 else "spam_label"
    num_labels = 2 if args.task == 1 else 4
    checkpoint = "vinai/phobert-base-v2"

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if args.data_zip:
        from datasets import Dataset, DatasetDict
        with zipfile.ZipFile(args.data_zip) as z:
            ds = DatasetDict({
                "train": Dataset.from_pandas(pd.read_csv(z.open("train.csv")), preserve_index=False),
                "validation": Dataset.from_pandas(pd.read_csv(z.open("dev.csv")), preserve_index=False),
                "test": Dataset.from_pandas(pd.read_csv(z.open("test.csv")), preserve_index=False),
            })
    else:
        ds = load_dataset("sonlam1102/vispamdetection2")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)

    def tokenize(batch):
        return tokenizer(
            batch["comment"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    tokenized = ds.map(tokenize, batched=True)
    keep = ["input_ids", "attention_mask", label_column]
    tokenized = tokenized.remove_columns(
        [c for c in tokenized["train"].column_names if c not in keep]
    )
    tokenized = tokenized.rename_column(label_column, "labels")

    # Convert only the required fields ourselves. This avoids the datasets
    # torch formatter importing an incompatible torchvision VideoReader.
    class TorchReviewDataset(torch.utils.data.Dataset):
        def __init__(self, split):
            self.input_ids = split["input_ids"]
            self.attention_mask = split["attention_mask"]
            self.labels = split["labels"]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return {
                "input_ids": torch.tensor(self.input_ids[index], dtype=torch.long),
                "attention_mask": torch.tensor(self.attention_mask[index], dtype=torch.long),
                "labels": torch.tensor(self.labels[index], dtype=torch.long),
            }

    train_loader = DataLoader(TorchReviewDataset(tokenized["train"]), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(TorchReviewDataset(tokenized["validation"]), batch_size=args.batch_size)
    test_loader = DataLoader(TorchReviewDataset(tokenized["test"]), batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels
    ).to(device)

    counts = np.bincount(np.array(ds["train"][label_column]), minlength=num_labels)
    weights = len(ds["train"]) / (num_labels * np.maximum(counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    beta = 0.9999
    effective = 1.0 - np.power(beta, counts)
    cb_weights = (1.0 - beta) / np.maximum(effective, 1e-12)
    cb_weights = cb_weights / cb_weights.mean()
    cb_class_weights = torch.tensor(cb_weights, dtype=torch.float32, device=device)
    def loss_fn(logits, labels):
        # Focal loss already down-weights easy examples; combining it with
        # inverse-frequency weights can over-correct the minority classes.
        weights_arg = class_weights if args.loss == "weighted" else None
        if args.loss == "cb":
            weights_arg = cb_class_weights
        ce = torch.nn.functional.cross_entropy(
            logits, labels, weight=weights_arg, reduction="none"
        )
        if args.loss == "focal":
            pt = torch.exp(-ce)
            return (((1.0 - pt) ** 2.0) * ce).mean()
        return ce.mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    def evaluate(loader):
        model.eval()
        all_pred, all_true = [], []
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                loss = loss_fn(logits, batch["labels"])
                total_loss += loss.item() * batch["labels"].size(0)
                all_pred.extend(logits.argmax(dim=-1).cpu().tolist())
                all_true.extend(batch["labels"].cpu().tolist())
        return {
            "loss": total_loss / len(loader.dataset),
            "accuracy": accuracy_score(all_true, all_pred),
            "macro_f1": f1_score(all_true, all_pred, average="macro", zero_division=0),
            "report": classification_report(all_true, all_pred, zero_division=0),
        }

    best_f1 = -1.0
    epochs_without_improvement = 0
    # Save beside this script, not relative to the current PowerShell folder.
    # This avoids writing to C:\Windows\System32 when the command is launched there.
    output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vispam_outputs")
    loss_name = args.loss
    output_name = args.output_name or f"vispam_phobert_task{args.task}_{loss_name}"
    output_dir = os.path.join(output_root, output_name)
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
            if step % 100 == 0:
                print(f"epoch={epoch} step={step} loss={running / 100:.4f}")
                running = 0.0

        dev_metrics = evaluate(dev_loader)
        print(f"epoch={epoch} dev={dev_metrics['accuracy']:.4f} "
              f"macro_f1={dev_metrics['macro_f1']:.4f}")
        if dev_metrics["macro_f1"] > best_f1:
            best_f1 = dev_metrics["macro_f1"]
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early_stopping epoch={epoch} best_dev_macro_f1={best_f1:.4f}")
                break

    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    test_metrics = evaluate(test_loader)
    print("\nFinal test metrics")
    print(f"accuracy={test_metrics['accuracy']:.4f}")
    print(f"macro_f1={test_metrics['macro_f1']:.4f}")
    print(test_metrics["report"])


if __name__ == "__main__":
    main()
