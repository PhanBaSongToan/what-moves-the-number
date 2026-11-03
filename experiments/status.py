"""Print a progress table of finished runs."""

from __future__ import annotations

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for path in sorted(glob.glob(os.path.join(HERE, "results", "*", "final_test_metrics.json"))):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    name = os.path.basename(os.path.dirname(path))
    rows.append(
        (
            name,
            data["test_accuracy"],
            data["test_macro_f1"],
            data["best_epoch"],
            data["elapsed_seconds"] / 60.0,
            data.get("test_binary"),
        )
    )

header = f"{'run':30s} {'acc':>7s} {'macroF1':>8s} {'ep':>3s} {'min':>5s} {'binF1':>6s}"
print(header)
print("-" * len(header))
for name, acc, f1, epoch, minutes, binary in rows:
    bin_f1 = f"{binary['macro_f1']:.4f}" if binary else "   -  "
    print(f"{name:30s} {acc:7.4f} {f1:8.4f} {epoch:3d} {minutes:5.1f} {bin_f1:>6s}")
print(f"\n{len(rows)} run(s) finished")
