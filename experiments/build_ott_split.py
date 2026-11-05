"""Build the second corpus: Ott et al. (2011) deceptive opinion spam.

The archive ships 1,600 reviews as text files under
``<polarity>/<source>/<fold>/<file>.txt``. Each file is one review; the label is
the source (``deceptive`` = fake, ``truthful`` = genuine) and the hotel name is
encoded in the file name, which gives a grouping key analogous to the product
key in ViSpamReviews.

Two splits are produced:

``ott_random``  stratified random split, the usual evaluation;
``ott_hotel``   hotel-disjoint split, the analogue of product-disjoint.

Usage:
    python experiments/build_ott_split.py
"""

from __future__ import annotations

import glob
import os
import re

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import config

RAW = os.path.join(config.DATA_DIR, "ott2011", "raw")
OUT = os.path.join(config.DATA_DIR, "ott2011")


def parse() -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"op_spam_v1\.4/(?P<polarity>[^/]+)/(?P<source>[^/]+)/(?P<fold>fold\d)/(?P<file>[^/]+)\.txt$")
    for path in glob.glob(os.path.join(RAW, "**", "*.txt"), recursive=True):
        match = pattern.search(path.replace("\\", "/"))
        if not match:
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read().strip()
        if not text:
            continue
        hotel = match.group("file").split("_", 1)[1].rsplit("_", 1)[0]
        rows.append(
            {
                "comment": text,
                "label": 1 if match.group("source").startswith("deceptive") else 0,
                "hotel": hotel.lower(),
                "polarity": match.group("polarity"),
                "fold": match.group("fold"),
                "file": match.group("file"),
            }
        )
    frame = pd.DataFrame(rows)
    frame["spam_label"] = frame["label"]
    frame["category"] = frame["polarity"]
    frame["product_name"] = frame["hotel"]
    frame["description"] = ""
    return frame


def write(frame: pd.DataFrame, name: str, train, dev, test) -> None:
    directory = os.path.join(OUT, name)
    os.makedirs(directory, exist_ok=True)
    for part, index in (("train", train), ("dev", dev), ("test", test)):
        frame.iloc[index].to_csv(os.path.join(directory, f"{part}.csv"), index=False,
                                 encoding="utf-8-sig")
        labels = frame.iloc[index]["label"].value_counts().sort_index().to_dict()
        print(f"  {part:5s} {len(index):5d}  labels={labels}")


def main() -> None:
    frame = parse()
    print("rows:", len(frame), "deceptive:", int(frame["label"].sum()))
    print("hotels:", frame["hotel"].nunique())

    index = list(range(len(frame)))

    # random stratified split
    train_dev, test = train_test_split(index, test_size=0.2, random_state=42,
                                       stratify=frame["label"])
    train, dev = train_test_split(train_dev, test_size=0.125, random_state=43,
                                  stratify=frame.iloc[train_dev]["label"])
    print("\nott_random")
    write(frame, "ott_random", train, dev, test)

    # hotel-disjoint split
    groups = frame["hotel"].to_numpy()
    first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_dev_idx, test_idx = next(first.split(frame, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=43)
    train_idx, dev_idx = next(second.split(frame.iloc[train_dev_idx], groups=groups[train_dev_idx]))
    print("\nott_hotel")
    write(frame, "ott_hotel", train_dev_idx[train_idx], train_dev_idx[dev_idx], test_idx)

    overlap = set(frame.iloc[train_dev_idx[train_idx]]["hotel"]) & set(frame.iloc[test_idx]["hotel"])
    print("hotel overlap train/test:", len(overlap))


if __name__ == "__main__":
    main()
