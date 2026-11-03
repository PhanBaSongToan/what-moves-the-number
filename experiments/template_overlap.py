"""Quantify template overlap in the false positives (label-noise check).

The qualitative inspection showed that many non-spam reviews flagged as spam are
near-identical to spam templates. This script measures that overlap with a fixed
phrase inventory taken from SPAM-1 training rows, so the claim in the paper is a
number rather than an impression.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
import zipfile

import pandas as pd
import config

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = config.dataset_zip()

PHRASES = [
    "chất lượng sản phẩm tuyệt vời",
    "đóng gói sản phẩm",
    "giao hàng nhanh",
    "đáng tiền",
    "nên mua",
]


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", " ", text).strip()


with zipfile.ZipFile(ZIP) as handle:
    train = pd.read_csv(handle.open("train.csv"))
    test = pd.read_csv(handle.open("test.csv"))

train["clean"] = train["comment"].map(normalise)
test["clean"] = test["comment"].map(normalise)

shared = set(test["clean"]) & set(train["clean"])


def share(frame: pd.DataFrame, phrase: str) -> float:
    return float(frame["clean"].str.contains(re.escape(normalise(phrase)), regex=True).mean())


report = {}
for phrase in PHRASES:
    report[phrase] = {
        "spam1_train": round(share(train[train.spam_label == 1], phrase), 4),
        "spam2_train": round(share(train[train.spam_label == 2], phrase), 4),
        "nonspam_train": round(share(train[train.spam_label == 0], phrase), 4),
    }

# false positives of the seed-42 single-task model
pred_path = os.path.join(HERE, "results", "t2_single_s42", "test_predictions.csv")
if os.path.exists(pred_path):
    preds = pd.read_csv(pred_path)
    preds["clean"] = preds["comment"].map(normalise)
    fp = preds[(preds.y_true == 0) & (preds.y_pred != 0)]
    report["false_positives"] = {
        "n": int(len(fp)),
        "share_with_template_phrase": round(
            float(fp["clean"].str.contains(re.escape(normalise(PHRASES[0]))).mean()), 4
        ),
        "share_verbatim_in_train": round(float(fp["clean"].isin(shared).mean()), 4),
    }
    # same measurement for true non-spam rows that were classified correctly
    tn = preds[(preds.y_true == 0) & (preds.y_pred == 0)]
    report["true_negatives"] = {
        "n": int(len(tn)),
        "share_with_template_phrase": round(
            float(tn["clean"].str.contains(re.escape(normalise(PHRASES[0]))).mean()), 4
        ),
        "share_verbatim_in_train": round(float(tn["clean"].isin(shared).mean()), 4),
    }

print(json.dumps(report, indent=2, ensure_ascii=False))
with open(os.path.join(HERE, "template_overlap.json"), "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, ensure_ascii=False)
