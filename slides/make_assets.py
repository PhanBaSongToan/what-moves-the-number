"""Assets for the slide deck: confusion matrices rendered for a wide slide."""

from __future__ import annotations

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "experiments", "results")
OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)

NAMES = ["Non-spam", "Spam-1", "Spam-2", "Spam-3"]


def load_confusion(name: str) -> np.ndarray:
    with open(os.path.join(RESULTS, name, "final_test_metrics.json"), encoding="utf-8") as handle:
        data = json.load(handle)
    return np.asarray(data["confusion_matrix"], dtype=float)


single = load_confusion("t2_single_s42")
multi = load_confusion("t2_multitask_s42")

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
for axis, matrix, title in zip(axes, (single, multi), ("Mô hình phẳng", "Mô hình đa tác vụ")):
    normalised = matrix / matrix.sum(axis=1, keepdims=True)
    axis.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    for row in range(4):
        for col in range(4):
            value = normalised[row, col]
            axis.text(
                col,
                row,
                f"{100 * value:.0f}",
                ha="center",
                va="center",
                fontsize=13,
                color="white" if value > 0.55 else "#1b1b1b",
                fontweight="bold" if row == col else "normal",
            )
    axis.set_xticks(range(4))
    axis.set_yticks(range(4))
    axis.set_xticklabels(NAMES, fontsize=11)
    axis.set_yticklabels(NAMES, fontsize=11)
    axis.set_xlabel("Dự đoán", fontsize=12)
    axis.set_ylabel("Nhãn thật", fontsize=12)
    axis.set_title(title, fontsize=14, pad=10)
    for spine in axis.spines.values():
        spine.set_visible(False)

fig.savefig(os.path.join(OUT, "confusion.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("written", os.path.join(OUT, "confusion.png"))
