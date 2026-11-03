"""One consistent definition of a duplicated review, applied to every split pair."""

from __future__ import annotations

import json
import re
import zipfile

import pandas as pd

ZIP = r"C:\Users\ACER\Downloads\vispamdetectionv2_official.zip"
DISJOINT = r"C:\Users\ACER\Downloads\vispam_outputs\product_disjoint_split_seed42"


def normalise(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()


with zipfile.ZipFile(ZIP) as handle:
    official = {name: pd.read_csv(handle.open(f"{name}.csv")) for name in ("train", "dev", "test")}

disjoint = {name: pd.read_csv(rf"{DISJOINT}\{name}.csv") for name in ("train", "dev", "test")}

report = {}
for name, parts in (("official", official), ("disjoint", disjoint)):
    train_text = normalise(parts["train"]["comment"])
    test_text = normalise(parts["test"]["comment"])
    all_text = pd.concat([normalise(part["comment"]) for part in parts.values()], ignore_index=True)
    report[name] = {
        "test_rows_matching_train": int(test_text.isin(set(train_text)).sum()),
        "distinct_test_texts_matching_train": int(len(set(test_text) & set(train_text))),
        "rows_in_duplicate_groups": int(all_text.duplicated(keep=False).sum()),
        "rows": int(len(all_text)),
        "unique_texts": int(all_text.nunique()),
    }

print(json.dumps(report, indent=2))
with open("experiments/duplicate_report.json", "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)
