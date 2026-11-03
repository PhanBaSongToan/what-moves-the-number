"""Build a group-disjoint split of the ViSpamReviews release.

Two grouping keys are supported:

* ``product``  -- no product appears in two splits (the split used in the paper);
* ``category`` -- no product *category* appears in two splits, which is a harder
  transfer test than holding out products.

Usage::

    python build_product_disjoint_split.py                     # product, seed 42
    python build_product_disjoint_split.py --seed 123          # another partition
    python build_product_disjoint_split.py --by category --seed 42
"""

from __future__ import annotations

import argparse
import os
import re
import unicodedata
import zipfile

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config


def norm(value) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    return re.sub(r"\s+", " ", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--by", choices=["product", "category"], default="product")
    parser.add_argument("--seed", type=int, default=42, help="seed of the train/test split")
    parser.add_argument("--dev-seed", type=int, default=None, help="seed of the train/dev split")
    parser.add_argument("--out", default=None, help="output directory")
    args = parser.parse_args()

    dev_seed = args.dev_seed if args.dev_seed is not None else args.seed + 1
    if args.out:
        out_dir = args.out
    elif args.by == "product" and args.seed == 42:
        out_dir = config.disjoint_dir()
    else:
        prefix = "product" if args.by == "product" else "category"
        out_dir = os.path.join(config.DATA_DIR, f"{prefix}_disjoint_split_seed{args.seed}")

    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(config.dataset_zip()) as handle:
        frames = [pd.read_csv(handle.open(f"{name}.csv")) for name in ("train", "dev", "test")]
    frame = pd.concat(frames, ignore_index=True)

    if args.by == "product":
        groups = (
            frame["product_name"].map(norm)
            + " || "
            + frame["category"].map(norm)
            + " || "
            + frame["description"].map(norm)
        ).to_numpy()
    else:
        groups = frame["category"].map(norm).to_numpy()

    first = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed)
    train_dev_idx, test_idx = next(first.split(frame, groups=groups))
    train_dev = frame.iloc[train_dev_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)

    second = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=dev_seed)
    train_idx, dev_idx = next(second.split(train_dev, groups=groups[train_dev_idx]))
    train = train_dev.iloc[train_idx].reset_index(drop=True)
    dev = train_dev.iloc[dev_idx].reset_index(drop=True)

    for name, part in (("train", train), ("dev", dev), ("test", test)):
        part.to_csv(os.path.join(out_dir, name + ".csv"), index=False, encoding="utf-8-sig")
        print(name, len(part), part["spam_label"].value_counts().sort_index().to_dict())

    train_groups = set(train["category"].map(norm)) if args.by == "category" else None
    if train_groups is not None:
        overlap = train_groups & set(test["category"].map(norm))
        print("category overlap train/test:", len(overlap))
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
