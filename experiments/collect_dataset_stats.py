"""Collect the dataset statistics and qualitative examples used in the paper."""

from __future__ import annotations

import json
import re
import unicodedata
import zipfile

import pandas as pd

ZIP = config.dataset_zip()

with zipfile.ZipFile(ZIP) as handle:
    parts = {name: pd.read_csv(handle.open(f"{name}.csv")) for name in ("train", "dev", "test")}


def norm(value) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    return re.sub(r"\s+", " ", text)


stats = {}
for name, frame in parts.items():
    tokens = frame["comment"].astype(str).str.split()
    stats[name] = {
        "rows": int(len(frame)),
        "vocab": int(len(set(" ".join(frame["comment"].astype(str)).split()))),
        "mean_tokens": round(float(tokens.str.len().mean()), 2),
        "median_tokens": int(tokens.str.len().median()),
        "p95_tokens": int(tokens.str.len().quantile(0.95)),
        "max_tokens": int(tokens.str.len().max()),
        "labels": {int(k): int(v) for k, v in frame["spam_label"].value_counts().sort_index().items()},
    }

train_groups = set(
    parts["train"]["product_name"].map(norm)
    + " || "
    + parts["train"]["category"].map(norm)
    + " || "
    + parts["train"]["description"].map(norm)
)
test_groups = set(
    parts["test"]["product_name"].map(norm)
    + " || "
    + parts["test"]["category"].map(norm)
    + " || "
    + parts["test"]["description"].map(norm)
)
dev_groups = set(
    parts["dev"]["product_name"].map(norm)
    + " || "
    + parts["dev"]["category"].map(norm)
    + " || "
    + parts["dev"]["description"].map(norm)
)

train_comments = set(parts["train"]["comment"].astype(str).str.strip().str.lower())
test_comments = set(parts["test"]["comment"].astype(str).str.strip().str.lower())

label_names = {0: "NO-SPAM", 1: "SPAM-1", 2: "SPAM-2", 3: "SPAM-3"}
examples = {}
for label in range(4):
    subset = parts["train"][parts["train"]["spam_label"] == label].copy()
    subset["length"] = subset["comment"].astype(str).str.len()
    subset = subset[(subset["length"] > 70) & (subset["length"] < 135)]
    picked = subset.drop_duplicates("comment").head(3)
    examples[label_names[label]] = picked["comment"].astype(str).tolist()

report = {
    "stats": stats,
    "unique_groups": {
        "train": len(train_groups),
        "dev": len(dev_groups),
        "test": len(test_groups),
    },
    "overlap": {
        "train_dev": len(train_groups & dev_groups),
        "train_test": len(train_groups & test_groups),
        "dev_test": len(dev_groups & test_groups),
    },
    "comment_overlap": {
        "test_comments_in_train": len(test_comments & train_comments),
        "test_rows_matching_train": int(
            parts["test"]["comment"].astype(str).str.strip().str.lower().isin(train_comments).sum()
        ),
        "all_rows": int(sum(len(f) for f in parts.values())),
        "unique_all": int(
            len(set(pd.concat([f["comment"].astype(str).str.strip().str.lower() for f in parts.values()])))
        ),
    },
    "examples": examples,
}

with open("experiments/dataset_report.json", "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, ensure_ascii=False)

print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2, ensure_ascii=False))
print("\n=== EXAMPLES ===")
for name, texts in examples.items():
    print(f"\n[{name}]")
    for text in texts:
        print(" -", text.replace("\n", " ")[:220])
import config
