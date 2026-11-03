"""Run every configuration of the paper, then regenerate the tables and figures.

    python experiments/run_all.py

Finished runs are skipped, so this is also the resume command after an
interruption. Needs a GPU for the training part; the analysis at the end does not.
"""

from __future__ import annotations

import sys

import run_experiment as R
import run_experiment  # noqa: F401  (explicit for clarity)
from analyze_results import main as analyze
from run_extra import JOBS as EXTRA_JOBS
from run_final import JOBS as FINAL_JOBS
from run_matrix import JOBS as MATRIX_JOBS

JOBS = MATRIX_JOBS + EXTRA_JOBS + FINAL_JOBS


def main() -> None:
    print(f"{len(JOBS)} configurations", flush=True)
    for index, job in enumerate(JOBS, 1):
        config = R.Config(**job)
        print(f"[{index}/{len(JOBS)}] {config.name}", flush=True)
        summary = R.run(config)
        print(
            f"[{index}/{len(JOBS)}] {config.name}: "
            f"macro_f1={summary['test_macro_f1']:.4f}",
            flush=True,
        )
    print("\nregenerating tables, figures and numbers", flush=True)
    analyze()


if __name__ == "__main__":
    sys.exit(main())
