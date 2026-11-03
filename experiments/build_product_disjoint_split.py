import os
import re
import unicodedata
import zipfile

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ZIP_PATH = r"C:\Users\ACER\Downloads\vispamdetectionv2_official.zip"
OUT_DIR = r"C:\Users\ACER\Downloads\vispam_outputs\product_disjoint_split_seed42"


def norm(x):
    x = unicodedata.normalize("NFKC", str(x)).lower().strip()
    return re.sub(r"\s+", " ", x)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        frames = [pd.read_csv(z.open(name + ".csv")) for name in ["train", "dev", "test"]]
    df = pd.concat(frames, ignore_index=True)
    groups = (df["product_name"].map(norm) + " || " + df["category"].map(norm) + " || " + df["description"].map(norm)).to_numpy()

    first = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_dev_idx, test_idx = next(first.split(df, groups=groups))
    train_dev = df.iloc[train_dev_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    train_dev_groups = groups[train_dev_idx]

    second = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=43)
    train_idx, dev_idx = next(second.split(train_dev, groups=train_dev_groups))
    train = train_dev.iloc[train_idx].reset_index(drop=True)
    dev = train_dev.iloc[dev_idx].reset_index(drop=True)

    for name, part in [("train", train), ("dev", dev), ("test", test)]:
        part.to_csv(os.path.join(OUT_DIR, name + ".csv"), index=False, encoding="utf-8-sig")
        print(name, len(part), part["spam_label"].value_counts().sort_index().to_dict())
    print("saved:", OUT_DIR)


if __name__ == "__main__": main()
