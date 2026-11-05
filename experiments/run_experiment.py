"""Locked-protocol experiments for the ViSpamReviews v2 study.

Every model in the paper is trained by this single script so that the
single-task baseline and the multi-task model share one protocol:

    encoder lr 1e-5, head lr 5e-5, max_len 100, batch 32, AdamW, bf16,
    grad-clip 1.0, early stopping on dev macro-F1 (patience 3), 10 epochs.

One deliberate change with respect to the exploratory runs is dynamic
per-batch padding instead of padding every sequence to max_len.  This only
touches compute, not the optimisation problem: padding tokens are masked out
of the self-attention anyway.  The change is recorded in every config.json.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import zipfile
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

import augment
import config

CHECKPOINT = "vinai/phobert-base-v2"
DEFAULT_ZIP = config.dataset_zip
DEFAULT_DISJOINT = config.disjoint_dir
RESULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

MAX_LEN = 100
BATCH_SIZE = 32
EPOCHS = 10
ENCODER_LR = 1e-5
HEAD_LR = 5e-5
PATIENCE = 3
PAD_TO = 8  # pad every batch to a multiple of 8 tokens for tensor-core kernels

_CACHE: dict[str, pd.DataFrame] = {}
_TOKEN_CACHE: dict[tuple[str, str], dict] = {}


# --------------------------------------------------------------------------- data
def _read_official(zip_path: str) -> dict[str, pd.DataFrame]:
    with zipfile.ZipFile(zip_path) as handle:
        frames = {}
        for name in ("train", "dev", "test"):
            frames[name] = pd.read_csv(handle.open(f"{name}.csv"))
    return frames


def _read_disjoint(data_dir: str) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(os.path.join(data_dir, f"{name}.csv"))
        for name in ("train", "dev", "test")
    }


def load_split(split: str) -> dict[str, pd.DataFrame]:
    """``official``, ``disjoint``, ``disjoint_<seed>`` or ``category_<seed>``."""
    key = f"split::{split}"
    if key not in _CACHE:
        if split == "official":
            _CACHE[key] = _read_official(DEFAULT_ZIP())
        elif split.startswith("disjoint_"):
            seed = split.split("_", 1)[1]
            _CACHE[key] = _read_disjoint(
                os.path.join(config.DATA_DIR, f"product_disjoint_split_seed{seed}")
            )
        elif split.startswith("category_"):
            seed = split.split("_", 1)[1]
            _CACHE[key] = _read_disjoint(
                os.path.join(config.DATA_DIR, f"category_disjoint_split_seed{seed}")
            )
        elif split.startswith("ott_"):
            _CACHE[key] = _read_disjoint(os.path.join(config.DATA_DIR, "ott2011", split))
        else:
            _CACHE[key] = _read_disjoint(DEFAULT_DISJOINT())
    return _CACHE[key]



_DESC_CACHE: dict[str, dict] = {}


def description_embeddings(split: str, backbone: str, max_len: int) -> dict:
    """Mean-pooled, frozen PhoBERT vectors for every product description."""
    key = f"{split}::{backbone}::{max_len}"
    if key in _DESC_CACHE:
        return _DESC_CACHE[key]

    cache_dir = os.path.join(config.DATA_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    safe = backbone.replace("/", "_")
    path = os.path.join(cache_dir, f"desc_{split}_{safe}_{max_len}.npz")

    if os.path.exists(path):
        loaded = np.load(path)
        out = {name: loaded[name] for name in loaded.files}
    else:
        frames = load_split(split)
        tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = AutoModel.from_pretrained(backbone).to(device).eval()
        out = {}
        for name, frame in frames.items():
            texts = frame["description"].fillna("").astype(str).tolist()
            chunks = []
            with torch.no_grad():
                for start in range(0, len(texts), 64):
                    batch = tokenizer(
                        texts[start : start + 64],
                        truncation=True,
                        padding=True,
                        max_length=max_len,
                        return_tensors="pt",
                    )
                    batch = {k: v.to(device) for k, v in batch.items()}
                    hidden = encoder(**batch).last_hidden_state
                    mask = batch["attention_mask"].unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                    chunks.append(pooled.cpu().numpy())
            out[name] = np.concatenate(chunks).astype(np.float32)
        np.savez(path, **out)
        del encoder
        torch.cuda.empty_cache()

    _DESC_CACHE[key] = out
    return out


def build_text(row, text_mode: str) -> str:
    comment = str(row["comment"] or "")
    if text_mode == "comment":
        return comment
    category = str(row["category"] or "")
    description = str(row["description"] or "")
    return f"{comment} [CATEGORY] {category} [DESCRIPTION] {description}"


def encode_split(
    split: str,
    text_mode: str,
    backbone: str = CHECKPOINT,
    max_len: int = MAX_LEN,
    normalize: bool = False,
) -> dict[str, dict]:
    key = f"{split}::{text_mode}::{backbone}::{max_len}::{normalize}"
    if key in _TOKEN_CACHE:
        return _TOKEN_CACHE[key]

    frames = load_split(split)
    tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=False)
    encoded = {}
    for name, frame in frames.items():
        if text_mode == "fusion":
            texts = frame["comment"].astype(str).tolist()
            descriptions = frame["description"].fillna("").astype(str).tolist()
        else:
            texts = [build_text(row, text_mode) for _, row in frame.iterrows()]
            descriptions = None
        if normalize:
            texts = [augment.normalize_text(item) for item in texts]
        batch = tokenizer(texts, truncation=True, max_length=max_len)
        encoded[name] = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "spam_label": frame["spam_label"].astype(int).tolist(),
            "binary_label": frame["label"].astype(int).tolist(),
            "comment": frame["comment"].astype(str).tolist(),
        }
        if descriptions is not None:
            encoded[name]["desc_input_ids"] = tokenizer(
                descriptions, truncation=True, max_length=max_len
            )["input_ids"]
            encoded[name]["desc_mask"] = tokenizer(
                descriptions, truncation=True, max_length=max_len
            )["attention_mask"]
    _TOKEN_CACHE[key] = encoded
    return encoded


def apply_dedupe(
    data: dict[str, dict], mode: str = "exact", threshold: float = 0.8
) -> tuple[dict[str, dict], int]:
    """Drop training rows that duplicate a test review.

    ``mode="exact"`` removes verbatim copies (the leakage measured in the paper);
    ``mode="near"`` also removes rows whose character 3-gram Jaccard similarity
    with a test review is at least ``threshold``.
    """

    def normalise(text: str) -> str:
        return re.sub(r"\s+", " ", str(text)).strip().lower()

    if mode == "near":
        flagged = augment.near_duplicate_indices(
            [str(item) for item in data["train"]["comment"]],
            [str(item) for item in data["test"]["comment"]],
            threshold=threshold,
        )
        keep = [i for i in range(len(data["train"]["comment"])) if i not in flagged]
    else:
        test_texts = {normalise(text) for text in data["test"]["comment"]}
        keep = [i for i, text in enumerate(data["train"]["comment"]) if normalise(text) not in test_texts]
    filtered = dict(data)
    filtered["train"] = {
        key: [value[i] for i in keep] for key, value in data["train"].items()
    }
    return filtered, len(data["train"]["comment"]) - len(keep)


class ReviewDataset(Dataset):
    def __init__(self, split_data: dict, task: int):
        self.ids = split_data["input_ids"]
        self.mask = split_data["attention_mask"]
        self.desc_ids = split_data.get("desc_input_ids")
        self.desc_mask = split_data.get("desc_mask")
        self.desc_emb = split_data.get("desc_emb")
        if task == 1:
            self.labels = split_data["binary_label"]
            self.binary = self.labels
            self.spam_type = [0] * len(self.labels)
        else:
            self.labels = split_data["spam_label"]
            self.binary = [int(y != 0) for y in self.labels]
            self.spam_type = [max(y - 1, 0) for y in self.labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        item = {
            "input_ids": self.ids[index],
            "attention_mask": self.mask[index],
            "labels": self.labels[index],
            "binary": self.binary[index],
            "spam_type": self.spam_type[index],
        }
        if self.desc_ids is not None:
            item["desc_input_ids"] = self.desc_ids[index]
            item["desc_mask"] = self.desc_mask[index]
        if self.desc_emb is not None:
            item["desc_emb"] = self.desc_emb[index]
        return item


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    longest = max(len(item["input_ids"]) for item in batch)
    width = ((longest + PAD_TO - 1) // PAD_TO) * PAD_TO
    ids = torch.zeros(len(batch), width, dtype=torch.long)
    mask = torch.zeros(len(batch), width, dtype=torch.long)
    for row, item in enumerate(batch):
        length = len(item["input_ids"])
        ids[row, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
        mask[row, :length] = torch.tensor(item["attention_mask"], dtype=torch.long)
    collated = {
        "input_ids": ids,
        "attention_mask": mask,
        "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.long),
        "binary": torch.tensor([item["binary"] for item in batch], dtype=torch.long),
        "spam_type": torch.tensor([item["spam_type"] for item in batch], dtype=torch.long),
    }
    if "desc_input_ids" in batch[0]:
        desc_width = ((max(len(item["desc_input_ids"]) for item in batch) + PAD_TO - 1) // PAD_TO) * PAD_TO
        desc_ids = torch.zeros(len(batch), desc_width, dtype=torch.long)
        desc_mask = torch.zeros(len(batch), desc_width, dtype=torch.long)
        for row, item in enumerate(batch):
            length = len(item["desc_input_ids"])
            desc_ids[row, :length] = torch.tensor(item["desc_input_ids"], dtype=torch.long)
            desc_mask[row, :length] = torch.tensor(item["desc_mask"], dtype=torch.long)
        collated["desc_input_ids"] = desc_ids
        collated["desc_mask"] = desc_mask
    if "desc_emb" in batch[0]:
        collated["desc_emb"] = torch.tensor(
            [item["desc_emb"] for item in batch], dtype=torch.float32
        )
    return collated


# --------------------------------------------------------------------------- models
class MultiTaskPhoBERT(nn.Module):
    def __init__(self, dropout: float = 0.2, backbone: str = CHECKPOINT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        hidden = self.encoder.config.hidden_size
        self.binary_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 2))
        self.type_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 3))

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.binary_head(hidden), self.type_head(hidden)


class FusionPhoBERT(nn.Module):
    """Shared encoder, description encoded separately and concatenated to the review vector."""

    def __init__(self, dropout: float = 0.2, backbone: str = CHECKPOINT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 4))

    def forward(
        self, input_ids, attention_mask, desc_input_ids=None, desc_mask=None, desc_emb=None
    ):
        comment = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        if desc_emb is not None:
            pooled = desc_emb
        else:
            description = self.encoder(
                input_ids=desc_input_ids, attention_mask=desc_mask
            ).last_hidden_state
            weight = desc_mask.unsqueeze(-1).float()
            pooled = (description * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1e-6)
        return self.head(torch.cat([comment, pooled], dim=1))


def combined(binary_logits: torch.Tensor, type_logits: torch.Tensor) -> torch.Tensor:
    binary = binary_logits.argmax(dim=1)
    spam_type = type_logits.argmax(dim=1) + 1
    return torch.where(binary == 0, torch.zeros_like(spam_type), spam_type)


@dataclass
class Config:
    name: str
    task: int = 2
    method: str = "single"          # single | multitask
    split: str = "official"         # official | disjoint
    text_mode: str = "comment"      # comment | metadata
    type_loss: str = "ce"           # ce | weighted
    seed: int = 42
    max_len: int = MAX_LEN
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    encoder_lr: float = ENCODER_LR
    head_lr: float = HEAD_LR
    patience: int = PATIENCE
    dropout: float = 0.2
    padding: str = "dynamic"
    precision: str = "bf16"
    backbone: str = CHECKPOINT
    dedupe: bool | str = False
    normalize: bool = False
    extra: dict = field(default_factory=dict)


def autocast(enabled: bool):
    if not enabled:
        return torch.autocast(device_type="cuda", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config, method: str) -> torch.optim.Optimizer:
    if method in ("multitask", "fusion", "fusion2"):
        encoder = [p for p in model.encoder.parameters()]
        encoder_ids = {id(p) for p in encoder}
        heads = [p for p in model.parameters() if id(p) not in encoder_ids]
        if method == "multitask":
            head_named = list(model.binary_head.named_parameters()) + list(model.type_head.named_parameters())
            heads = [p for _, p in head_named]
    else:
        head_ids = {id(p) for n, p in model.named_parameters() if n.startswith(("classifier", "pooler"))}
        encoder = [p for n, p in model.named_parameters() if id(p) not in head_ids]
        heads = [p for n, p in model.named_parameters() if id(p) in head_ids]
    groups = [
        {"params": encoder, "lr": cfg.encoder_lr},
        {"params": heads, "lr": cfg.head_lr},
    ]
    return torch.optim.AdamW(groups)


@torch.no_grad()
def evaluate(model, loader, device, task: int, method: str, precision: str = "bf16") -> dict:
    model.eval()
    use_amp = precision != "fp32" and device.type == "cuda"
    true, pred, probs = [], [], []
    binary_true, binary_pred = [], []
    total_loss, seen = 0.0, 0
    ce = nn.CrossEntropyLoss()
    for batch in loader:
        labels = batch["labels"].to(device)
        inputs = {k: batch[k].to(device) for k in ("input_ids", "attention_mask")}
        if "desc_input_ids" in batch:
            inputs["desc_input_ids"] = batch["desc_input_ids"].to(device)
            inputs["desc_mask"] = batch["desc_mask"].to(device)
        if "desc_emb" in batch:
            inputs["desc_emb"] = batch["desc_emb"].to(device)
        with autocast(use_amp):
            if method == "multitask":
                binary_logits, type_logits = model(**inputs)
            elif method in ("fusion", "fusion2"):
                logits = model(**inputs)
            else:
                logits = model(**inputs).logits
        if method == "multitask":
            if task == 1:
                combined_pred = binary_logits.argmax(dim=1)
            else:
                combined_pred = combined(binary_logits, type_logits)
                four = torch.full((len(labels), 4), 1e-9, device=device)
                binary_prob = binary_logits.float().softmax(dim=1)[:, 1:]
                type_prob = type_logits.float().softmax(dim=1)
                four[:, 0] = binary_logits.float().softmax(dim=1)[:, 0]
                for cls in range(3):
                    four[:, cls + 1] = binary_prob[:, 0] * type_prob[:, cls]
                probs.extend(four.cpu().tolist())
            loss = ce(binary_logits.float(), batch["binary"].to(device))
            binary_true.extend(batch["binary"].tolist())
            binary_pred.extend(binary_logits.argmax(dim=1).cpu().tolist())
        else:
            combined_pred = logits.argmax(dim=1)
            if task == 2:
                probs.extend(logits.float().softmax(dim=1).cpu().tolist())
            loss = ce(logits.float(), labels)
        total_loss += loss.item() * len(labels)
        seen += len(labels)
        true.extend(labels.cpu().tolist())
        pred.extend(combined_pred.cpu().tolist())

    true_arr = np.asarray(true)
    pred_arr = np.asarray(pred)
    metrics = {
        "loss": total_loss / max(seen, 1),
        "accuracy": float(accuracy_score(true_arr, pred_arr)),
        "macro_f1": float(f1_score(true_arr, pred_arr, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true_arr, pred_arr, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            true_arr, pred_arr, labels=[0, 1, 2, 3] if task == 2 else [0, 1]
        ).tolist(),
        "support": {int(k): int(v) for k, v in zip(*np.unique(true_arr, return_counts=True))},
    }
    if task == 2:
        per_class = {}
        for cls in range(4):
            mask_t = true_arr == cls
            tp = int(((pred_arr == cls) & mask_t).sum())
            fp = int(((pred_arr == cls) & ~mask_t).sum())
            fn = int(((pred_arr != cls) & mask_t).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[str(cls)] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(mask_t.sum()),
            }
        metrics["per_class"] = per_class
    if method == "multitask":
        metrics["binary"] = {
            "accuracy": float(accuracy_score(binary_true, binary_pred)),
            "macro_f1": float(f1_score(binary_true, binary_pred, average="macro", zero_division=0)),
        }
    metrics["_arrays"] = {"true": true, "pred": pred, "probs": probs}
    return metrics


def run(cfg: Config) -> dict:
    out_dir = os.path.join(RESULT_ROOT, cfg.name)
    metrics_path = os.path.join(out_dir, "final_test_metrics.json")
    if os.path.exists(metrics_path):
        print(f"[skip] {cfg.name} already finished")
        with open(metrics_path, encoding="utf-8") as handle:
            return json.load(handle)

    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.precision != "fp32" and device.type == "cuda"
    print(f"\n=== {cfg.name} ===")
    print(json.dumps(asdict(cfg), ensure_ascii=False))

    data = encode_split(cfg.split, cfg.text_mode, cfg.backbone, cfg.max_len, cfg.normalize)
    if cfg.method == "fusion2":
        embeddings = description_embeddings(cfg.split, cfg.backbone, cfg.max_len)
        data = {
            name: {**part, "desc_emb": [row.tolist() for row in embeddings[name]]}
            for name, part in data.items()
        }
    removed = 0
    if cfg.dedupe:
        mode = "near" if cfg.dedupe == "near" else "exact"
        data, removed = apply_dedupe(data, mode=mode)

    if cfg.method == "multitask":
        model = MultiTaskPhoBERT(cfg.dropout, cfg.backbone).to(device)
    elif cfg.method in ("fusion", "fusion2"):
        model = FusionPhoBERT(cfg.dropout, cfg.backbone).to(device)
    else:
        num_labels = 2 if cfg.task == 1 else 4
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.backbone, num_labels=num_labels
        ).to(device)
    train_loader = DataLoader(
        ReviewDataset(data["train"], cfg.task),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        ReviewDataset(data["dev"], cfg.task), batch_size=cfg.batch_size, collate_fn=collate
    )
    test_loader = DataLoader(
        ReviewDataset(data["test"], cfg.task), batch_size=cfg.batch_size, collate_fn=collate
    )
    print(f"rows: train={len(train_loader.dataset)} dev={len(dev_loader.dataset)} test={len(test_loader.dataset)}")


    optimizer = build_optimizer(model, cfg, cfg.method)
    ce = nn.CrossEntropyLoss()

    type_weights = None
    if cfg.method == "multitask" and cfg.type_loss == "weighted":
        spam_labels = [max(y - 1, 0) for y in data["train"]["spam_label"] if y != 0]
        counts = np.bincount(spam_labels, minlength=3).astype(float)
        weights = len(spam_labels) / (3 * np.maximum(counts, 1))
        type_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print("type weights:", weights.round(4).tolist())
    type_ce = nn.CrossEntropyLoss(weight=type_weights)

    log_path = os.path.join(out_dir, "training_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        handle.write("epoch,train_loss,dev_loss,dev_accuracy,dev_macro_f1,seconds\n")

    best_f1, best_epoch, best_state, wait = -1.0, 0, None, 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        started = time.time()
        model.train()
        running, steps = 0.0, 0
        for batch in train_loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast(use_amp):
                if cfg.method == "multitask":
                    binary_logits, type_logits = model(
                        batch_gpu["input_ids"], batch_gpu["attention_mask"]
                    )
                    binary_y = batch_gpu["binary"]
                    mask = binary_y == 1
                    loss = ce(binary_logits.float(), binary_y)
                    if mask.any():
                        loss = loss + type_ce(type_logits.float()[mask], batch_gpu["spam_type"][mask])
                elif cfg.method in ("fusion", "fusion2"):
                    kwargs = {
                        "input_ids": batch_gpu["input_ids"],
                        "attention_mask": batch_gpu["attention_mask"],
                    }
                    if "desc_emb" in batch_gpu:
                        kwargs["desc_emb"] = batch_gpu["desc_emb"]
                    else:
                        kwargs["desc_input_ids"] = batch_gpu["desc_input_ids"]
                        kwargs["desc_mask"] = batch_gpu["desc_mask"]
                    logits = model(**kwargs)
                    loss = ce(logits.float(), batch_gpu["labels"])
                else:
                    logits = model(
                        input_ids=batch_gpu["input_ids"],
                        attention_mask=batch_gpu["attention_mask"],
                    ).logits
                    loss = ce(logits.float(), batch_gpu["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
            steps += 1

        dev = evaluate(model, dev_loader, device, cfg.task, cfg.method, cfg.precision)
        elapsed = time.time() - started
        print(
            f"epoch={epoch} loss={running / max(steps, 1):.4f} "
            f"dev_acc={dev['accuracy']:.4f} dev_f1={dev['macro_f1']:.4f} ({elapsed:.0f}s)"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(steps, 1),
                "dev_loss": dev["loss"],
                "dev_accuracy": dev["accuracy"],
                "dev_macro_f1": dev["macro_f1"],
                "seconds": elapsed,
            }
        )
        with open(log_path, "a", newline="", encoding="utf-8") as handle:
            handle.write(
                f"{epoch},{history[-1]['train_loss']:.6f},{dev['loss']:.6f},"
                f"{dev['accuracy']:.6f},{dev['macro_f1']:.6f},{elapsed:.1f}\n"
            )
        if dev["macro_f1"] > best_f1:
            best_f1 = dev["macro_f1"]
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(out_dir, "best_model_state.pt"))
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"early stop at epoch {epoch} (best {best_epoch}, dev_f1={best_f1:.4f})")
                break

    model.load_state_dict(best_state)
    dev_final = evaluate(model, dev_loader, device, cfg.task, cfg.method, cfg.precision)
    dev_arrays = dev_final.pop("_arrays")
    test = evaluate(model, test_loader, device, cfg.task, cfg.method, cfg.precision)
    arrays = test.pop("_arrays")
    summary = {
        "name": cfg.name,
        "config": asdict(cfg),
        "best_epoch": best_epoch,
        "best_dev_macro_f1": best_f1,
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "test_weighted_f1": test["weighted_f1"],
        "test_loss": test["loss"],
        "test_per_class": test.get("per_class"),
        "confusion_matrix": test["confusion_matrix"],
        "support": test["support"],
        "test_binary": test.get("binary"),
        "history": history,
        "train_rows": len(train_loader.dataset),
        "removed_train_rows": removed,
        "dev_rows": len(dev_loader.dataset),
        "test_rows": len(test_loader.dataset),
        "device": str(device),
        "torch": torch.__version__,
        "elapsed_seconds": sum(row["seconds"] for row in history),
    }
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    frame = pd.DataFrame(
        {
            "y_true": arrays["true"],
            "y_pred": arrays["pred"],
            "comment": [c[:400] for c in data["test"]["comment"]],
        }
    )
    if arrays["probs"]:
        frame["p0"] = [row[0] for row in arrays["probs"]]
        frame["p1"] = [row[1] for row in arrays["probs"]]
        frame["p2"] = [row[2] for row in arrays["probs"]]
        frame["p3"] = [row[3] for row in arrays["probs"]]
    frame.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False, encoding="utf-8-sig")

    dev_frame = pd.DataFrame(
        {
            "y_true": dev_arrays["true"],
            "y_pred": dev_arrays["pred"],
            "comment": [c[:400] for c in data["dev"]["comment"]],
        }
    )
    if dev_arrays["probs"]:
        for index in range(4):
            dev_frame[f"p{index}"] = [row[index] for row in dev_arrays["probs"]]
    dev_frame.to_csv(os.path.join(out_dir, "dev_predictions.csv"), index=False, encoding="utf-8-sig")

    print(
        f"test accuracy={test['accuracy']:.4f} macro_f1={test['macro_f1']:.4f}"
    )
    return summary
