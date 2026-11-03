"""Extra descriptive statistics quoted in the paper (metadata length, duplicates)."""

from __future__ import annotations

import json
import zipfile

import pandas as pd

ZIP = r"C:\Users\ACER\Downloads\vispamdetectionv2_official.zip"

with zipfile.ZipFile(ZIP) as handle:
    train = pd.read_csv(handle.open("train.csv"))
    test = pd.read_csv(handle.open("test.csv"))

comment_tokens = train["comment"].astype(str).str.split().str.len()
description_tokens = train["description"].astype(str).str.split().str.len()

over_100 = int(
    (
        (train["comment"].astype(str).str.split().str.len() + train["description"].astype(str).str.split().str.len()
         + 2) > 100
    ).sum()
)

normalised = train["comment"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
duplicates = normalised.duplicated(keep=False)
dup_by_label = train.loc[duplicates, "spam_label"].value_counts().sort_index().to_dict()

report = {
    "train_description_tokens": {
        "mean": round(float(description_tokens.mean()), 2),
        "median": int(description_tokens.median()),
        "p95": int(description_tokens.quantile(0.95)),
        "max": int(description_tokens.max()),
    },
    "train_comment_tokens": {
        "mean": round(float(comment_tokens.mean()), 2),
        "median": int(comment_tokens.median()),
    },
    "rows_where_comment_plus_description_exceeds_100_tokens": over_100,
    "share": round(over_100 / len(train), 4),
    "duplicate_rows_in_train": int(duplicates.sum()),
    "duplicate_rows_by_label": {int(k): int(v) for k, v in dup_by_label.items()},
    "test_rows_with_duplicate_text_in_train": int(
        test["comment"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.lower().isin(set(normalised)).sum()
    ),
}

print(json.dumps(report, indent=2))
with open("experiments/extra_stats.json", "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, ensure_ascii=False)
