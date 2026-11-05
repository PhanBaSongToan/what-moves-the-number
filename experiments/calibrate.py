"""Fit a temperature on the development split and re-measure ECE on test.

The probability columns stored with every run are softmax outputs, so their logs
are logits up to an additive constant. That is enough to fit a single
temperature: softmax(log(p)/T) only needs the log-probabilities.

Usage:
    python experiments/calibrate.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import analyze_results as A

RESULTS = A.RESULTS


def logits(frame: pd.DataFrame) -> np.ndarray:
    probabilities = frame[["p0", "p1", "p2", "p3"]].to_numpy(dtype=float)
    return np.log(np.clip(probabilities, 1e-12, 1.0))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    exponent = np.exp(z)
    return exponent / exponent.sum(axis=1, keepdims=True)


def nll(frame: pd.DataFrame, temperature: float) -> float:
    scaled = softmax(logits(frame) / temperature)
    rows = np.arange(len(frame))
    return float(-np.log(np.clip(scaled[rows, frame["y_true"].to_numpy()], 1e-12, 1.0)).mean())


def with_temperature(frame: pd.DataFrame, temperature: float) -> pd.DataFrame:
    scaled = softmax(logits(frame) / temperature)
    out = frame.copy()
    for index in range(4):
        out[f"p{index}"] = scaled[:, index]
    return out


def fit_temperature(frame: pd.DataFrame) -> float:
    grid = np.exp(np.linspace(np.log(0.05), np.log(10.0), 400))
    losses = [nll(frame, temperature) for temperature in grid]
    return float(grid[int(np.argmin(losses))])


def main() -> None:
    report = {}
    for model, dev_name, test_name in (
        ("flat", "t2_single_cal_s42", "t2_single_s42"),
        ("cascade", "t2_multitask_cal_s42", "t2_multitask_s42"),
    ):
        dev = pd.read_csv(os.path.join(RESULTS, dev_name, "dev_predictions.csv"))
        test = pd.read_csv(os.path.join(RESULTS, test_name, "test_predictions.csv"))
        temperature = fit_temperature(dev)
        before = A.expected_calibration_error(test)
        after = A.expected_calibration_error(with_temperature(test, temperature))
        entry = {
            "temperature": round(temperature, 3),
            "ece_before": round(100 * before["ece"], 2),
            "ece_after": round(100 * after["ece"], 2),
            "dev_nll_before": round(nll(dev, 1.0), 4),
            "dev_nll_after": round(nll(dev, temperature), 4),
        }
        report[model] = entry
        print(model, entry)

    with open(os.path.join(RESULTS, "calibration.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
