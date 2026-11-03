"""What kinds of rows does the deduplication control remove?"""

from __future__ import annotations

import json
import os
import re
import zipfile

import pandas as pd

ZIP = r"C:\Users\ACER\Downloads\vispamdetectionv2_official.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "results", "t2_single_dedup_s42", "final_test_metrics.json")


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


with zipfile.ZipFile(ZIP) as handle:
    train = pd.read_csv(handle.open("train.csv"))
    test = pd.read_csv(handle.open("test.csv"))

train["clean"] = train["comment"].map(normalise)
test_clean = set(test["comment"].map(normalise))
mask = train["clean"].isin(test_clean)
removed = train[mask]

report = {
    "removed_rows": int(len(removed)),
    "removed_by_label": {int(k): int(v) for k, v in removed["spam_label"].value_counts().sort_index().items()},
    "removed_distinct_texts": int(removed["clean"].nunique()),
    "labels_of_distinct_texts": {},
    "train_label_counts": {int(k): int(v) for k, v in train["spam_label"].value_counts().sort_index().items()},
}
distinct = removed.drop_duplicates("clean")
report["labels_of_distinct_texts"] = {
    int(k): int(v) for k, v in distinct["spam_label"].value_counts().sort_index().items()
}

if os.path.exists(RESULT):
    with open(RESULT, encoding="utf-8") as handle:
        metrics = json.load(handle)
    report["removed_train_rows_reported"] = metrics.get("removed_train_rows")
    report["dedup_test_macro_f1"] = metrics["test_macro_f1"]

print(json.dumps(report, indent=2, ensure_ascii=False))
