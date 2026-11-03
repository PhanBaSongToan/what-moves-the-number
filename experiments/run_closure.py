"""Close the nine remaining limitations of the paper.

1. split-to-split variance: two more product-disjoint partitions
2. separate description encoder (frozen), three seeds
3. near-duplicate control (character 3-gram Jaccard >= 0.8)
4. teencode normalisation control
5. category-disjoint transfer (no product category seen in training)
6. two more seeds for each learning-rate cell
7. two calibration runs, identical to the seed-42 pair, whose only job is to
   write the development predictions needed to measure ECE off the test set
"""

from __future__ import annotations

import argparse
import sys
import time

import run_experiment as R

JOBS = [
    # --- separate, frozen description encoder --------------------------------
    dict(name="t2_fusion2_s42", task=2, method="fusion2", text_mode="fusion", seed=42),
    dict(name="t2_fusion2_s43", task=2, method="fusion2", text_mode="fusion", seed=43),
    dict(name="t2_fusion2_s44", task=2, method="fusion2", text_mode="fusion", seed=44),
    # --- split-to-split variance ---------------------------------------------
    dict(name="t2_single_disjoint123_s42", task=2, method="single", split="disjoint_123", seed=42),
    dict(name="t2_multitask_disjoint123_s42", task=2, method="multitask", split="disjoint_123", seed=42),
    dict(name="t2_single_disjoint456_s42", task=2, method="single", split="disjoint_456", seed=42),
    dict(name="t2_multitask_disjoint456_s42", task=2, method="multitask", split="disjoint_456", seed=42),
    # --- text normalisation ---------------------------------------------------
    dict(name="t2_single_norm_s42", task=2, method="single", split="official", seed=42, normalize=True),
    dict(name="t2_multitask_norm_s42", task=2, method="multitask", split="official", seed=42, normalize=True),
    # --- near-duplicate control ----------------------------------------------
    dict(name="t2_single_neardup_s42", task=2, method="single", split="official", seed=42, dedupe="near"),
    dict(name="t2_multitask_neardup_s42", task=2, method="multitask", split="official", seed=42, dedupe="near"),
    # --- category-disjoint transfer ------------------------------------------
    dict(name="t2_single_category_s42", task=2, method="single", split="category_42", seed=42),
    dict(name="t2_multitask_category_s42", task=2, method="multitask", split="category_42", seed=42),
    # --- remaining seeds for the learning-rate control -----------------------
    dict(name="t2_single_lr2_s43", task=2, method="single", split="official", seed=43,
         encoder_lr=2e-5, head_lr=2e-5),
    dict(name="t2_single_lr2_s44", task=2, method="single", split="official", seed=44,
         encoder_lr=2e-5, head_lr=2e-5),
    dict(name="t2_multitask_lr2_s43", task=2, method="multitask", split="official", seed=43,
         encoder_lr=2e-5, head_lr=2e-5),
    dict(name="t2_multitask_lr2_s44", task=2, method="multitask", split="official", seed=44,
         encoder_lr=2e-5, head_lr=2e-5),
    # --- calibration runs (development predictions) --------------------------
    dict(name="t2_single_cal_s42", task=2, method="single", split="official", seed=42),
    dict(name="t2_multitask_cal_s42", task=2, method="multitask", split="official", seed=42),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    jobs = [job for job in JOBS if args.only is None or job["name"] in args.only]
    print(f"running {len(jobs)} closure job(s)", flush=True)
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
