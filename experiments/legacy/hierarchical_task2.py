import os
import random

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SEED = 42
CHECKPOINT = "vinai/phobert-base-v2"
OUTPUT_ROOT = r"C:\Users\ACER\Downloads\vispam_outputs\hierarchical_task2"
BATCH_SIZE = 32
EPOCHS = 1


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EncodedReviews(Dataset):
    def __init__(self, split, indices, label_key, label_map):
        self.ids = [split["input_ids"][i] for i in indices]
        self.mask = [split["attention_mask"][i] for i in indices]
        self.labels = [label_map[int(split[label_key][i])] for i in indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.ids[i], dtype=torch.long),
            "attention_mask": torch.tensor(self.mask[i], dtype=torch.long),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }


def make_loader(split, indices, label_map, batch_size=BATCH_SIZE, shuffle=False):
    return DataLoader(
        EncodedReviews(split, indices, "spam_label", label_map),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_model(train_loader, dev_loader, num_labels, class_weights, device, name):
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=num_labels
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    best_f1 = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
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
                print(f"{name} epoch={epoch} step={step} loss={running / 100:.4f}")
                running = 0.0

        dev_y, dev_pred = predict(model, dev_loader, device)
        dev_f1 = f1_score(dev_y, dev_pred, average="macro", zero_division=0)
        dev_acc = accuracy_score(dev_y, dev_pred)
        print(f"{name} epoch={epoch} dev_accuracy={dev_acc:.4f} dev_macro_f1={dev_f1:.4f}")
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"{name} selected_dev_macro_f1={best_f1:.4f}")
    return model


def predict(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            x = {k: v.to(device) for k, v in batch.items()}
            pred = model(**x).logits.argmax(dim=-1).cpu().tolist()
            preds.extend(pred)
            ys.extend(labels.tolist())
    return np.array(ys), np.array(preds)


def hierarchical_predict(stage1, stage2, split, device, batch_size=BATCH_SIZE):
    all_indices = list(range(len(split["spam_label"])))
    loader = make_loader(split, all_indices, {0: 0, 1: 1, 2: 1, 3: 1}, batch_size)
    stage1_y, stage1_pred = predict(stage1, loader, device)

    # Run stage 2 on every item; only its prediction is used for stage-1 Spam.
    spam3_loader = DataLoader(
        EncodedReviews(split, all_indices, "spam_label", {0: 0, 1: 0, 2: 1, 3: 2}),
        batch_size=batch_size,
    )
    _, stage2_pred = predict(stage2, spam3_loader, device)
    combined = np.where(stage1_pred == 0, 0, stage2_pred + 1)
    true = np.array(split["spam_label"])
    return true, combined


def main():
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    raw = load_dataset("sonlam1102/vispamdetection2")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=False)

    def tokenize(batch):
        return tokenizer(batch["comment"], truncation=True, padding="max_length", max_length=128)

    ds = raw.map(tokenize, batched=True)
    keep = ["input_ids", "attention_mask", "spam_label"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])

    train_spam = np.array(ds["train"]["spam_label"])
    dev_spam = np.array(ds["validation"]["spam_label"])
    train_all = list(range(len(train_spam)))
    dev_all = list(range(len(dev_spam)))
    train_spam_idx = [i for i, y in enumerate(train_spam) if y != 0]
    dev_spam_idx = [i for i, y in enumerate(dev_spam) if y != 0]

    # Stage 1: normal (0) vs spam (1,2,3).
    s1_map = {0: 0, 1: 1, 2: 1, 3: 1}
    s1_train = make_loader(ds["train"], train_all, s1_map, shuffle=True)
    s1_dev = make_loader(ds["validation"], dev_all, s1_map)
    s1_counts = np.bincount([s1_map[int(y)] for y in train_spam], minlength=2)
    s1_weights = torch.tensor(len(train_spam) / (2 * np.maximum(s1_counts, 1)), dtype=torch.float32)
    stage1 = train_model(s1_train, s1_dev, 2, s1_weights, device, "stage1")

    # Stage 2: spam type 1/2/3 only.
    s2_map = {1: 0, 2: 1, 3: 2}
    s2_train = make_loader(ds["train"], train_spam_idx, s2_map, shuffle=True)
    s2_dev = make_loader(ds["validation"], dev_spam_idx, s2_map)
    s2_labels = [s2_map[int(train_spam[i])] for i in train_spam_idx]
    s2_counts = np.bincount(s2_labels, minlength=3)
    s2_weights = torch.tensor(len(s2_labels) / (3 * np.maximum(s2_counts, 1)), dtype=torch.float32)
    stage2 = train_model(s2_train, s2_dev, 3, s2_weights, device, "stage2")

    dev_y, dev_pred = hierarchical_predict(stage1, stage2, ds["validation"], device)
    test_y, test_pred = hierarchical_predict(stage1, stage2, ds["test"], device)
    print("\nHierarchical validation")
    print("accuracy=%.4f macro_f1=%.4f" % (accuracy_score(dev_y, dev_pred), f1_score(dev_y, dev_pred, average="macro", zero_division=0)))
    print("\nHierarchical test")
    print("accuracy=%.4f macro_f1=%.4f" % (accuracy_score(test_y, test_pred), f1_score(test_y, test_pred, average="macro", zero_division=0)))
    print(classification_report(test_y, test_pred, zero_division=0))


if __name__ == "__main__":
    main()
