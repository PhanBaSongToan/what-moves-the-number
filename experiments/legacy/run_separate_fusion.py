import argparse
import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class FusionClassifier(torch.nn.Module):
    def __init__(self, checkpoint, num_categories, num_labels, category_dim=32):
        super().__init__()
        self.comment_encoder = AutoModel.from_pretrained(checkpoint)
        hidden = self.comment_encoder.config.hidden_size
        self.category_embedding = torch.nn.Embedding(num_categories, category_dim)
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(hidden * 4 + 1 + category_dim, hidden),
            torch.nn.Sigmoid(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden + category_dim),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden + category_dim, num_labels),
        )

    def forward(self, comment_ids, comment_mask, description_emb, category_ids):
        comment_out = self.comment_encoder(
            input_ids=comment_ids,
            attention_mask=comment_mask,
        )
        review = mean_pool(comment_out.last_hidden_state, comment_mask)
        diff = torch.abs(review - description_emb)
        product = review * description_emb
        cosine = torch.nn.functional.cosine_similarity(
            review, description_emb, dim=-1
        ).unsqueeze(-1)
        category = self.category_embedding(category_ids)
        fusion_input = torch.cat([review, description_emb, diff, product, cosine, category], dim=-1)
        gate = self.gate(fusion_input)
        fused = gate * review + (1.0 - gate) * description_emb
        return self.classifier(torch.cat([fused, category], dim=-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=[1, 2], default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--description-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = "vinai/phobert-base-v2"
    label_col = "label" if args.task == 1 else "spam_label"
    num_labels = 2 if args.task == 1 else 4
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    ds = load_dataset("sonlam1102/vispamdetection2")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)

    categories = sorted(set(ds["train"]["category"]))
    category_to_id = {value: i for i, value in enumerate(categories)}

    def tokenize_column(split, column):
        return tokenizer(
            split[column],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt",
        )

    # Precompute frozen description embeddings once; this prevents duplicate
    # description encoding during every classifier training step.
    desc_encoder = AutoModel.from_pretrained(checkpoint).to(device).eval()
    description_embeddings = {}
    with torch.no_grad():
        for split_name in ["train", "validation", "test"]:
            tokenized_desc = tokenize_column(ds[split_name], "description")
            vectors = []
            for start in range(0, len(ds[split_name]), args.description_batch_size):
                ids = tokenized_desc["input_ids"][start:start + args.description_batch_size].to(device)
                mask = tokenized_desc["attention_mask"][start:start + args.description_batch_size].to(device)
                out = desc_encoder(input_ids=ids, attention_mask=mask)
                vectors.append(mean_pool(out.last_hidden_state, mask).cpu())
            description_embeddings[split_name] = torch.cat(vectors)
            print(split_name, "description embeddings:", description_embeddings[split_name].shape)
    del desc_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def make_tensor_dataset(split_name):
        split = ds[split_name]
        comment = tokenize_column(split, "comment")
        category_ids = torch.tensor([category_to_id.get(x, 0) for x in split["category"]], dtype=torch.long)
        labels = torch.tensor(split[label_col], dtype=torch.long)
        return TensorDataset(
            comment["input_ids"],
            comment["attention_mask"],
            description_embeddings[split_name],
            category_ids,
            labels,
        )

    train_data = make_tensor_dataset("train")
    dev_data = make_tensor_dataset("validation")
    test_data = make_tensor_dataset("test")
    loaders = {
        "train": DataLoader(train_data, batch_size=args.batch_size, shuffle=True),
        "dev": DataLoader(dev_data, batch_size=args.batch_size),
        "test": DataLoader(test_data, batch_size=args.batch_size),
    }

    model = FusionClassifier(
        checkpoint, len(categories), num_labels
    ).to(device)
    counts = np.bincount(np.asarray(ds["train"][label_col]), minlength=num_labels)
    weights = len(ds["train"]) / (num_labels * np.maximum(counts, 1))
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    def evaluate(loader):
        model.eval()
        true, pred = [], []
        with torch.no_grad():
            for batch in loader:
                comment_ids, comment_mask, desc, cat, labels = [x.to(device) for x in batch]
                logits = model(comment_ids, comment_mask, desc, cat)
                true.extend(labels.cpu().tolist())
                pred.extend(logits.argmax(-1).cpu().tolist())
        return {
            "accuracy": accuracy_score(true, pred),
            "macro_f1": f1_score(true, pred, average="macro", zero_division=0),
            "report": classification_report(true, pred, zero_division=0, output_dict=True),
        }

    best_f1 = -1.0
    output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vispam_outputs")
    output_dir = os.path.join(output_root, f"separate_fusion_t{args.task}_seed{args.seed}")
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(loaders["train"], 1):
            comment_ids, comment_mask, desc, cat, labels = [x.to(device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            logits = model(comment_ids, comment_mask, desc, cat)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
            if step % 100 == 0:
                print(f"epoch={epoch} step={step} loss={running/100:.4f}")
                running = 0.0
        metrics = evaluate(loaders["dev"])
        print("dev_accuracy=", round(metrics["accuracy"], 4), "dev_macro_f1=", round(metrics["macro_f1"], 4))
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    model.load_state_dict(torch.load(os.path.join(output_dir, "model.pt"), map_location=device))
    test_metrics = evaluate(loaders["test"])
    result = {
        "task": args.task,
        "seed": args.seed,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_report": test_metrics["report"],
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("test_accuracy=", round(result["test_accuracy"], 4))
    print("test_macro_f1=", round(result["test_macro_f1"], 4))


if __name__ == "__main__":
    main()
