"""Collect representative errors for the qualitative table in the paper.

Reads the per-row predictions of the seed-42 runs and writes
``experiments/error_candidates.txt`` with short, quotable examples for each
error family used in Section VI.
"""

from __future__ import annotations

import io
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
NAMES = {0: "NO-SPAM", 1: "SPAM-1", 2: "SPAM-2", 3: "SPAM-3"}


def load(name: str) -> pd.DataFrame:
    frame = pd.read_csv(os.path.join(RESULTS, name, "test_predictions.csv"))
    frame["comment"] = frame["comment"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    frame["wrong"] = frame["y_true"] != frame["y_pred"]
    frame["confidence"] = frame[["p0", "p1", "p2", "p3"]].max(axis=1)
    return frame


def main() -> None:
    single = load("t2_single_s42")
    multi = load("t2_multitask_s42")

    out = io.open(os.path.join(HERE, "error_candidates.txt"), "w", encoding="utf-8")

    def block(title: str, frame: pd.DataFrame, limit: int = 6) -> None:
        pred_column = "y_pred_m" if "y_pred_m" in frame.columns else "y_pred"
        out.write(f"\n##### {title} (n={len(frame)})\n")
        for _, row in frame.head(limit).iterrows():
            out.write(
                f" - true={NAMES[int(row['y_true'])]} pred={NAMES[int(row[pred_column])]} "
                f"p={row['confidence']:.2f} len={len(row['comment'])}\n"
                f"   {row['comment'][:230]}\n"
            )

    both = single.merge(
        multi[["comment", "y_true", "y_pred", "wrong"]], on="comment", suffixes=("_s", "_m"), how="inner"
    )
    both["y_true"] = both["y_true_s"]
    both["y_pred"] = both["y_pred_s"]
    fixed = both[both["wrong_s"] & ~both["wrong_m"]]
    broken = both[~both["wrong_s"] & both["wrong_m"]]

    block("type confusion, single-task: SPAM-1 -> SPAM-3",
          single[(single.y_true == 1) & (single.y_pred == 3)])
    block("type confusion, single-task: SPAM-2 -> SPAM-3",
          single[(single.y_true == 2) & (single.y_pred == 3)])
    block("false positive: NO-SPAM -> SPAM-1",
          single[(single.y_true == 0) & (single.y_pred == 1)])
    block("missed spam: SPAM-2 -> NO-SPAM",
          single[(single.y_true == 2) & (single.y_pred == 0)])
    block("missed spam: SPAM-1 -> NO-SPAM",
          single[(single.y_true == 1) & (single.y_pred == 0)])
    block("fixed by multi-task", fixed, limit=8)
    block("broken by multi-task", broken, limit=8)

    out.close()

    stats = {
        "single_errors": int(single["wrong"].sum()),
        "multi_errors": int(multi["wrong"].sum()),
        "fixed": int(len(fixed)),
        "broken": int(len(broken)),
    }
    print(stats)


if __name__ == "__main__":
    main()
