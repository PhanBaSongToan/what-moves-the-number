"""Run the locked-protocol experiment matrix sequentially on one GPU.

Ordering is by importance: the Task-2 headline pair first, then seeds, then
Task 1, then the product-disjoint evaluation, then the two locked ablations.
Every run writes to experiments/results/<name>/.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import run_experiment as R

JOBS = [
    # --- Task 2, official split: the headline comparison, three seeds --------
    dict(name="t2_single_s42", task=2, method="single", split="official", seed=42),
    dict(name="t2_multitask_s42", task=2, method="multitask", split="official", seed=42),
    dict(name="t2_single_s43", task=2, method="single", split="official", seed=43),
    dict(name="t2_multitask_s43", task=2, method="multitask", split="official", seed=43),
    dict(name="t2_single_s44", task=2, method="single", split="official", seed=44),
    dict(name="t2_multitask_s44", task=2, method="multitask", split="official", seed=44),
    # --- Product-disjoint split ---------------------------------------------
    dict(name="t2_single_disjoint_s42", task=2, method="single", split="disjoint", seed=42),
    dict(name="t2_multitask_disjoint_s42", task=2, method="multitask", split="disjoint", seed=42),
    # --- Locked ablations ----------------------------------------------------
    dict(name="t2_multitask_weighted_s42", task=2, method="multitask", split="official",
         seed=42, type_loss="weighted"),
    dict(name="t2_single_metadata_s42", task=2, method="single", split="official",
         seed=42, text_mode="metadata"),
    # --- Task 1, official split: separate single-task models -----------------
    dict(name="t1_single_s42", task=1, method="single", split="official", seed=42),
    dict(name="t1_single_s43", task=1, method="single", split="official", seed=43),
    dict(name="t1_single_s44", task=1, method="single", split="official", seed=44),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="subset of job names")
    parser.add_argument("--smoke", action="store_true", help="256 rows, 1 epoch")
    args = parser.parse_args()

    if args.smoke:
        original = R.encode_split

        def small(split: str, text_mode: str, backbone: str = R.CHECKPOINT, max_len: int = R.MAX_LEN):
            data = original(split, text_mode, backbone, max_len)
            return {
                name: {
                    key: (value[:256] if isinstance(value, list) else value)
                    for key, value in part.items()
                }
                for name, part in data.items()
            }

        R.encode_split = small

    jobs = [job for job in JOBS if args.only is None or job["name"] in args.only]
    print(f"running {len(jobs)} job(s); results -> {R.RESULT_ROOT}", flush=True)
    started = time.time()
    for index, job in enumerate(jobs, 1):
        config = R.Config(**job)
        if args.smoke:
            config.name = config.name + "_smoke"
            config.epochs = 1
            config.patience = 1
        print(f"\n[{index}/{len(jobs)}] {config.name}", flush=True)
        try:
            summary = R.run(config)
            print(
                f"[{index}/{len(jobs)}] {config.name}: "
                f"acc={summary['test_accuracy']:.4f} macro_f1={summary['test_macro_f1']:.4f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{index}/{len(jobs)}] {config.name} FAILED: {exc!r}", flush=True)
            raise
    print(f"\nall done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
