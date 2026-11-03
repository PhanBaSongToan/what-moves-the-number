"""Release the product-disjoint split as a compact row-index file.

The split itself is a partition of the public release, so storing three 40 MB
CSV copies is redundant. This script recovers the assignment and writes
``experiments/data/product_disjoint_ids.csv`` with one row per review:

    split,row_id

``row_id`` is the row position in ``concat([train, dev, test])`` of the public
release, in that order.
"""

from __future__ import annotations

import os
import zipfile

import pandas as pd
import config

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = config.dataset_zip()
DISJOINT = config.disjoint_dir()
OUT = os.path.join(HERE, "data", "product_disjoint_ids.csv")

with zipfile.ZipFile(ZIP) as handle:
    frames = [pd.read_csv(handle.open(f"{name}.csv")) for name in ("train", "dev", "test")]

base = pd.concat(frames, ignore_index=True)
base["row_id"] = range(len(base))
key_columns = [column for column in base.columns if column != "row_id"]
base["occurrence"] = base.groupby(key_columns, dropna=False).cumcount()

assignments = []
for split in ("train", "dev", "test"):
    part = pd.read_csv(os.path.join(DISJOINT, f"{split}.csv"))
    part["occurrence"] = part.groupby(key_columns, dropna=False).cumcount()
    merged = part.merge(base, on=key_columns + ["occurrence"], how="left", validate="one_to_one")
    missing = int(merged["row_id"].isna().sum())
    print(split, len(part), "unmatched:", missing)
    if missing:
        raise SystemExit("could not map every row; refusing to write a partial index")
    merged["split"] = split
    assignments.append(merged[["row_id", "split"]])

index = pd.concat(assignments, ignore_index=True).sort_values("row_id")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
index.to_csv(OUT, index=False)

counts = index["split"].value_counts().to_dict()
print("written:", OUT, len(index), counts)

# re-materialise the split from the index and check it matches the source files
check = base.merge(index, on="row_id", validate="one_to_one")
for split in ("train", "dev", "test"):
    expected = pd.read_csv(os.path.join(DISJOINT, f"{split}.csv")).sort_values(
        list(key_columns)
    ).reset_index(drop=True)
    got = check[check["split"] == split].sort_values(list(key_columns)).reset_index(drop=True)
    same_rows = len(expected) == len(got)
    same_labels = bool((expected["spam_label"].to_numpy() == got["spam_label"].to_numpy()).all())
    print(f"{split}: rows match={same_rows} labels match={same_labels}")
    assert same_rows and same_labels, f"split {split} not reproduced exactly"
print("index verified")
