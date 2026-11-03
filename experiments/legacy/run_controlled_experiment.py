import argparse
import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TorchReviewDataset(Dataset):
    def __init__(self, split):
        self.input_ids = split["input_ids"]
        self.attention_mask = split["attention_mask"]
        self.labels = split["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.input_ids[i], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[i], dtype=torch.long),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, choices=[1, 2], default=1)
    p.add_argument(
        "--variant",
        choices=["comment", "category", "description", "metadata"],
        default="comment",
    )
    p.add_argument("--loss", choices=["ce", "weighted"], default="weighted")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_col = "label" if args.task == 1 else "spam_label"
    num_labels = 2 if args.task == 1 else 4
    checkpoint = "vinai/phobert-base-v2"
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    ds = load_dataset("sonlam1102/vispamdetection2")
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()

    def make_text(row):
        comment = str(row.get("comment") or "")
        if args.variant == "comment":
            return comment
        category = str(row.get("category") or "")
        description = str(row.get("description") or "")
        if args.variant == "category":
            return f"{comment} [CATEGORY] {category}"
        if args.variant == "description":
            return f"{comment} [DESCRIPTION] {description}"
        return f"{comment} [CATEGORY] {category} [DESCRIPTION] {description}"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)

    def tokenize_split(split):
        texts = [make_text(row) for row in split]
        encoded = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )
        encoded["labels"] = split[label_col]
        return encoded

    tokenized = {
        name: tokenize_split(split)
        for name, split in ds.items()
    }

    train_loader = DataLoader(TorchReviewDataset(tokenized["train"]), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(TorchReviewDataset(tokenized["validation"]), batch_size=args.batch_size)
    test_loader = DataLoader(TorchReviewDataset(tokenized["test"]), batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels
    ).to(device)
    counts = np.bincount(np.asarray(train_df[label_col]), minlength=num_labels)
    weights = len(train_df) / (num_labels * np.maximum(counts, 1))
    loss_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=loss_weights if args.loss == "weighted" else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    def evaluate(loader):
        model.eval()
        true, pred = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                true.extend(batch["labels"].cpu().tolist())
                pred.extend(logits.argmax(-1).cpu().tolist())
        return {
            "accuracy": accuracy_score(true, pred),
            "macro_f1": f1_score(true, pred, average="macro", zero_division=0),
            "report": classification_report(true, pred, zero_division=0, output_dict=True),
        }

    output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vispam_outputs")
    output_dir = os.path.join(
        output_root, f"controlled_t{args.task}_{args.variant}_{args.loss}_seed{args.seed}"
    )
    os.makedirs(output_dir, exist_ok=True)
    best_f1 = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
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
                print(f"epoch={epoch} step={step} loss={running/100:.4f}")
                running = 0.0

        dev_metrics = evaluate(dev_loader)
        print("dev_accuracy=", round(dev_metrics["accuracy"], 4),
              "dev_macro_f1=", round(dev_metrics["macro_f1"], 4))
        if dev_metrics["macro_f1"] > best_f1:
            best_f1 = dev_metrics["macro_f1"]
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    test_metrics = evaluate(test_loader)
    result = {
        "task": args.task,
        "variant": args.variant,
        "loss": args.loss,
        "seed": args.seed,
        "max_length": args.max_length,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_report": test_metrics["report"],
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("test_accuracy=", round(result["test_accuracy"], 4))
    print("test_macro_f1=", round(result["test_macro_f1"], 4))
    print(classification_report(
        list(test_df[label_col]),
        [max(range(num_labels), key=lambda j: 0) for _ in range(0)],
    ) if False else "Metrics saved to " + os.path.join(output_dir, "metrics.json"))


if __name__ == "__main__":
    main()
