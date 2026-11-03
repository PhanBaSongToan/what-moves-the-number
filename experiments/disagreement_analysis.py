"""Composition of the rows where the two models disagree (seed 42).

The two prediction files come from the same test split and the evaluation loader
is not shuffled, so rows are aligned by position; we assert that the ground
truth matches before comparing.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def load(name: str) -> pd.DataFrame:
    frame = pd.read_csv(os.path.join(RESULTS, name, "test_predictions.csv"))
    frame["comment"] = frame["comment"].astype(str)
    frame["wrong"] = frame["y_true"] != frame["y_pred"]
    return frame


single = load("t2_single_s42")
multi = load("t2_multitask_s42")
if not np.array_equal(single["y_true"].to_numpy(), multi["y_true"].to_numpy()):
    raise SystemExit("prediction files are not aligned")

fixed_mask = single["wrong"].to_numpy() & ~multi["wrong"].to_numpy()
broken_mask = ~single["wrong"].to_numpy() & multi["wrong"].to_numpy()

print("fixed", int(fixed_mask.sum()), single.loc[fixed_mask, "y_true"].value_counts().sort_index().to_dict())
print("  single predicted:", single.loc[fixed_mask, "y_pred"].value_counts().sort_index().to_dict())
print("  multi predicted:", multi.loc[fixed_mask, "y_pred"].value_counts().sort_index().to_dict())
print("broken", int(broken_mask.sum()), single.loc[broken_mask, "y_true"].value_counts().sort_index().to_dict())
print("  multi predicted:", multi.loc[broken_mask, "y_pred"].value_counts().sort_index().to_dict())
print("total disagreements:", int(fixed_mask.sum() + broken_mask.sum()))
print("net correct gain:", int(fixed_mask.sum() - broken_mask.sum()))
