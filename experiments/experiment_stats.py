"""Small statistics helpers shared by the analysis script."""

from __future__ import annotations

import numpy as np


def aggregate(runs: dict, prefix: str, field: str | None = None) -> dict:
    """Mean and sample standard deviation of a metric over every run matching ``prefix``.

    ``field=None`` aggregates test accuracy and test macro-F1. Passing
    ``field="test_binary"`` aggregates the binary-head sub-dictionary instead.
    """
    names = [name for name in runs if name.startswith(prefix)]
    if not names:
        raise KeyError(f"no finished run matches prefix {prefix!r}")

    def collect(getter):
        values = [getter(runs[name]) for name in names]
        return {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "values": values,
            "n": len(values),
            "runs": names,
        }

    if field is None:
        return {
            "acc": collect(lambda run: run["test_accuracy"]),
            "f1": collect(lambda run: run["test_macro_f1"]),
        }
    return {
        "acc": collect(lambda run: run[field]["accuracy"]),
        "f1": collect(lambda run: run[field]["macro_f1"]),
    }


def gate_decomposition(frame) -> dict:
    """Split test errors into detection errors and spam-type errors."""
    true = frame["y_true"].to_numpy()
    pred = frame["y_pred"].to_numpy()
    return {
        "n": int(len(frame)),
        "correct": int(np.sum(true == pred)),
        "non_spam_to_spam": int(np.sum((true == 0) & (pred != 0))),
        "spam_to_non_spam": int(np.sum((true != 0) & (pred == 0))),
        "spam_type_confusion": int(np.sum((true != 0) & (pred != 0) & (true != pred))),
        "type_confusion_by_true": {
            str(cls): int(np.sum((true == cls) & (pred != 0) & (pred != cls)))
            for cls in (1, 2, 3)
        },
    }
