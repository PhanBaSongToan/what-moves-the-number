"""Sanity checks for the statistical helpers on synthetic predictions."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_results as A  # noqa: E402
from experiment_stats import gate_decomposition  # noqa: E402

rng = np.random.default_rng(0)
n = 2000
truth = rng.choice([0, 1, 2, 3], size=n, p=[0.73, 0.015, 0.08, 0.175])

# model A: 80% correct; model B: 85% correct
pred_a = np.where(rng.random(n) < 0.8, truth, rng.choice([0, 1, 2, 3], size=n))
pred_b = np.where(rng.random(n) < 0.85, truth, rng.choice([0, 1, 2, 3], size=n))

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp", "stats_check")
os.makedirs(root, exist_ok=True)
for name, pred in (("a", pred_a), ("b", pred_b)):
    os.makedirs(os.path.join(root, name))
    pd.DataFrame(
        {
            "y_true": truth,
            "y_pred": pred,
            "comment": [f"row {i}" for i in range(n)],
            "p0": 0.25,
            "p1": 0.25,
            "p2": 0.25,
            "p3": 0.25,
        }
    ).to_csv(os.path.join(root, name, "test_predictions.csv"), index=False)

A.RESULTS = root

boot = A.paired_bootstrap("a", "b")
manual = A.macro_f1_from_confusion(A.confusion_from_pairs(truth, pred_b)) - A.macro_f1_from_confusion(
    A.confusion_from_pairs(truth, pred_a)
)
print("bootstrap:", boot)
print("manual delta:", manual)
assert abs(boot["point_delta"] - manual) < 1e-9, "point estimate mismatch"
assert boot["ci_low"] <= boot["point_delta"] <= boot["ci_high"], "point outside CI"

mcn = A.mcnemar("a", "b")
print("mcnemar:", mcn)
assert mcn["only_a"] + mcn["only_b"] > 0

gate = gate_decomposition(pd.read_csv(os.path.join(root, "a", "test_predictions.csv")))
total = gate["correct"] + gate["non_spam_to_spam"] + gate["spam_to_non_spam"] + gate["spam_type_confusion"]
print("gate:", gate["n"], total)
assert total == gate["n"], "error decomposition does not add up"

# agreement check: identical models must produce a zero delta and a degenerate interval
os.makedirs(os.path.join(root, "c"))
pd.read_csv(os.path.join(root, "a", "test_predictions.csv")).to_csv(
    os.path.join(root, "c", "test_predictions.csv"), index=False
)
same = A.paired_bootstrap("a", "c")
print("identical models:", same)
assert abs(same["point_delta"]) < 1e-12 and abs(same["ci_high"] - same["ci_low"]) < 1e-12

print("all checks passed")
