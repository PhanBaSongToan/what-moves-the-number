"""Additional runs that close the remaining gaps of the paper.

1. Two more seeds for the product-disjoint split (both models), so that the
   disjoint comparison carries the same three-seed protocol as the main one.
2. A text-deduplicated control: training rows whose review text also occurs in
   the test split are removed, then both models are re-run once.
3. An alternative backbone (XLM-R base, already cached locally) to show that the
   conclusion is not specific to PhoBERT.
4. One sequence-length variation (128 tokens) for the single-task model, as a
   minimal sensitivity check on the "one hyperparameter setting" limitation.

Run order is by value: disjoint seeds first, then the controls.
"""

from __future__ import annotations

import argparse
import sys
import time

import run_experiment as R

XLM_R = "xlm-roberta-base"

JOBS = [
    # --- product-disjoint, seeds 43 and 44 -----------------------------------
    dict(name="t2_single_disjoint_s43", task=2, method="single", split="disjoint", seed=43),
    dict(name="t2_multitask_disjoint_s43", task=2, method="multitask", split="disjoint", seed=43),
    dict(name="t2_single_disjoint_s44", task=2, method="single", split="disjoint", seed=44),
    dict(name="t2_multitask_disjoint_s44", task=2, method="multitask", split="disjoint", seed=44),
    # --- text-deduplicated control (official split) --------------------------
    dict(name="t2_single_dedup_s42", task=2, method="single", split="official", seed=42, dedupe=True),
    dict(name="t2_multitask_dedup_s42", task=2, method="multitask", split="official", seed=42, dedupe=True),
    # --- alternative backbone ------------------------------------------------
    dict(name="t2_single_xlmr_s42", task=2, method="single", split="official", seed=42, backbone=XLM_R),
    # --- sequence-length sensitivity ----------------------------------------
    dict(name="t2_single_len128_s42", task=2, method="single", split="official", seed=42, max_len=128),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    jobs = [job for job in JOBS if args.only is None or job["name"] in args.only]
    print(f"running {len(jobs)} extra job(s)", flush=True)
    started = time.time()
    for index, job in enumerate(jobs, 1):
        config = R.Config(**job)
        print(f"\n[{index}/{len(jobs)}] {config.name}", flush=True)
        summary = R.run(config)
        print(
            f"[{index}/{len(jobs)}] {config.name}: acc={summary['test_accuracy']:.4f} "
            f"macro_f1={summary['test_macro_f1']:.4f}",
            flush=True,
        )
    print(f"\nall done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
