import json
import os
import re
import unicodedata
import zipfile

import pandas as pd
import config

ZIP_PATH = config.dataset_zip()
OUT = config.audit_dir()


def norm(x):
    x = unicodedata.normalize("NFKC", str(x)).lower().strip()
    return re.sub(r"\s+", " ", x)


def load():
    with zipfile.ZipFile(ZIP_PATH) as z:
        return {name: pd.read_csv(z.open(name + ".csv")) for name in ["train", "dev", "test"]}


def main():
    os.makedirs(OUT, exist_ok=True)
    data = load()
    groups = {}
    for split, df in data.items():
        key = (df["product_name"].map(norm) + " || " + df["category"].map(norm) + " || " + df["description"].map(norm))
        groups[split] = set(key)
    report = {"rows": {s: len(df) for s, df in data.items()}, "label_counts": {s: df["spam_label"].value_counts().sort_index().to_dict() for s, df in data.items()}}
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        report[f"{a}_{b}_overlap_groups"] = len(groups[a] & groups[b])
    report["unique_groups"] = {s: len(g) for s, g in groups.items()}
    report["interpretation"] = "Overlap means the same normalized product metadata occurs in both splits; random-split scores may be optimistic."
    with open(os.path.join(OUT, "product_leakage_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
