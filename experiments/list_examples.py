"""List candidate examples per label so the paper can quote representative ones."""

from __future__ import annotations

import io
import re
import zipfile

import pandas as pd

ZIP = config.dataset_zip()

with zipfile.ZipFile(ZIP) as handle:
    train = pd.read_csv(handle.open("train.csv"))

names = {0: "NO-SPAM", 1: "SPAM-1", 2: "SPAM-2", 3: "SPAM-3"}
out = io.open("experiments/example_candidates.txt", "w", encoding="utf-8")
for label in range(4):
    subset = train[train["spam_label"] == label].drop_duplicates("comment").copy()
    subset["clean"] = subset["comment"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    subset = subset[(subset["clean"].str.len() > 55) & (subset["clean"].str.len() < 150)]
    out.write(f"\n##### {names[label]} (n={len(subset)})\n")
    for text in subset["clean"].head(25):
        out.write(f" - {text}\n")
out.close()
print("ok")
import config
