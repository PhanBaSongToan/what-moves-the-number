"""Final round: close every remaining gap in the paper.

1. Fusion baseline: the product description is encoded separately with a shared
   encoder and concatenated with the review representation, so the comparison
   with the reference paper's metadata integration exists in our protocol.
2. Two more seeds for the text-deduplicated control (both models).
3. Two more seeds for the XLM-R backbone and for the 128-token window.
4. A single-learning-rate variant (2e-5 everywhere) for both models, as the
   minimal hyperparameter-sensitivity check.
"""

from __future__ import annotations

import argparse
import sys
import time

import run_experiment as R

XLM_R = "xlm-roberta-base"

JOBS = [
    # --- metadata fusion baseline (description encoded separately) -----------
    dict(name="t2_fusion_s42", task=2, method="fusion", text_mode="fusion", seed=42),
    dict(name="t2_fusion_s43", task=2, method="fusion", text_mode="fusion", seed=43),
    dict(name="t2_fusion_s44", task=2, method="fusion", text_mode="fusion", seed=44),
    # --- text-deduplicated control, seeds 43 and 44 --------------------------
    dict(name="t2_single_dedup_s43", task=2, method="single", split="official", seed=43, dedupe=True),
    dict(name="t2_single_dedup_s44", task=2, method="single", split="official", seed=44, dedupe=True),
    dict(name="t2_multitask_dedup_s43", task=2, method="multitask", split="official", seed=43, dedupe=True),
    dict(name="t2_multitask_dedup_s44", task=2, method="multitask", split="official", seed=44, dedupe=True),
    # --- backbone and window controls, seeds 43 and 44 ----------------------
    dict(name="t2_single_xlmr_s43", task=2, method="single", split="official", seed=43, backbone=XLM_R),
    dict(name="t2_single_xlmr_s44", task=2, method="single", split="official", seed=44, backbone=XLM_R),
    dict(name="t2_single_len128_s43", task=2, method="single", split="official", seed=43, max_len=128),
    dict(name="t2_single_len128_s44", task=2, method="single", split="official", seed=44, max_len=128),
    # --- single learning rate for both models -------------------------------
    dict(name="t2_single_lr2_s42", task=2, method="single", split="official", seed=42,
         encoder_lr=2e-5, head_lr=2e-5),
    dict(name="t2_multitask_lr2_s42", task=2, method="multitask", split="official", seed=42,
         encoder_lr=2e-5, head_lr=2e-5),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    jobs = [job for job in JOBS if args.only is None or job["name"] in args.only]
    print(f"running {len(jobs)} final job(s)", flush=True)
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
