"""Every configuration of the study, in one place.

A "job" is one training run: a dict of ``run_experiment.Config`` fields plus a
unique ``name``. ``run_all.py`` walks this registry and skips runs that already
finished, so the same command both trains and resumes.

Groups are by purpose, not by the order in which they were added:

    main         the headline comparison (Task 2, three seeds)
    task1        the binary task
    split        product-disjoint, extra partitions, category-disjoint
    control      leakage, backbone, window, schedule, metadata fusion
    ablation     class weighting and naive metadata concatenation
    calibration  duplicate of the seed-42 pair, used for dev ECE
"""

from __future__ import annotations

XLM_R = "xlm-roberta-base"

MAIN = [
    dict(name="t2_single_s42", task=2, method="single", split="official", seed=42),
    dict(name="t2_multitask_s42", task=2, method="multitask", split="official", seed=42),
    dict(name="t2_single_s43", task=2, method="single", split="official", seed=43),
    dict(name="t2_multitask_s43", task=2, method="multitask", split="official", seed=43),
    dict(name="t2_single_s44", task=2, method="single", split="official", seed=44),
    dict(name="t2_multitask_s44", task=2, method="multitask", split="official", seed=44),
]

TASK1 = [
    dict(name="t1_single_s42", task=1, method="single", split="official", seed=42),
    dict(name="t1_single_s43", task=1, method="single", split="official", seed=43),
    dict(name="t1_single_s44", task=1, method="single", split="official", seed=44),
]

SPLIT = [
    # product-disjoint, three seeds
    dict(name="t2_single_disjoint_s42", task=2, method="single", split="disjoint", seed=42),
    dict(name="t2_multitask_disjoint_s42", task=2, method="multitask", split="disjoint", seed=42),
    dict(name="t2_single_disjoint_s43", task=2, method="single", split="disjoint", seed=43),
    dict(name="t2_multitask_disjoint_s43", task=2, method="multitask", split="disjoint", seed=43),
    dict(name="t2_single_disjoint_s44", task=2, method="single", split="disjoint", seed=44),
    dict(name="t2_multitask_disjoint_s44", task=2, method="multitask", split="disjoint", seed=44),
    # two more partitions, to measure split-to-split variance
    dict(name="t2_single_disjoint123_s42", task=2, method="single", split="disjoint_123", seed=42),
    dict(name="t2_multitask_disjoint123_s42", task=2, method="multitask", split="disjoint_123", seed=42),
    dict(name="t2_single_disjoint456_s42", task=2, method="single", split="disjoint_456", seed=42),
    dict(name="t2_multitask_disjoint456_s42", task=2, method="multitask", split="disjoint_456", seed=42),
    # no product category shared with training
    dict(name="t2_single_category_s42", task=2, method="single", split="category_42", seed=42),
    dict(name="t2_multitask_category_s42", task=2, method="multitask", split="category_42", seed=42),
]

CONTROL = [
    # leakage: exact duplicates removed, three seeds
    dict(name="t2_single_dedup_s42", task=2, method="single", split="official", seed=42, dedupe=True),
    dict(name="t2_multitask_dedup_s42", task=2, method="multitask", split="official", seed=42, dedupe=True),
    dict(name="t2_single_dedup_s43", task=2, method="single", split="official", seed=43, dedupe=True),
    dict(name="t2_multitask_dedup_s43", task=2, method="multitask", split="official", seed=43, dedupe=True),
    dict(name="t2_single_dedup_s44", task=2, method="single", split="official", seed=44, dedupe=True),
    dict(name="t2_multitask_dedup_s44", task=2, method="multitask", split="official", seed=44, dedupe=True),
    # leakage: near-duplicates removed
    dict(name="t2_single_neardup_s42", task=2, method="single", split="official", seed=42, dedupe="near"),
    dict(name="t2_multitask_neardup_s42", task=2, method="multitask", split="official", seed=42, dedupe="near"),
    # text normalisation
    dict(name="t2_single_norm_s42", task=2, method="single", split="official", seed=42, normalize=True),
    dict(name="t2_multitask_norm_s42", task=2, method="multitask", split="official", seed=42, normalize=True),
    # backbone
    dict(name="t2_single_xlmr_s42", task=2, method="single", split="official", seed=42, backbone=XLM_R),
    dict(name="t2_single_xlmr_s43", task=2, method="single", split="official", seed=43, backbone=XLM_R),
    dict(name="t2_single_xlmr_s44", task=2, method="single", split="official", seed=44, backbone=XLM_R),
    # sequence length
    dict(name="t2_single_len128_s42", task=2, method="single", split="official", seed=42, max_len=128),
    dict(name="t2_single_len128_s43", task=2, method="single", split="official", seed=43, max_len=128),
    dict(name="t2_single_len128_s44", task=2, method="single", split="official", seed=44, max_len=128),
    # learning-rate schedule
    dict(name="t2_single_lr2_s42", task=2, method="single", split="official", seed=42,
         encoder_lr=2e-5, head_lr=2e-5),
    dict(name="t2_single_lr2_s43", task=2, method="single", split="official", seed=43,
         encoder_lr=2e-5, head_lr=2e-5),
    dict(name="t2_single_lr2_s44", task=2, method="single", split="official", seed=44,
         encoder_lr=2e-5, head_lr=2e-5),
    dict(name="t2_multitask_lr2_s42", task=2, method="multitask", split="official", seed=42,
         encoder_lr=2e-5, head_lr=2e-5),
    dict(name="t2_multitask_lr2_s43", task=2, method="multitask", split="official", seed=43,
         encoder_lr=2e-5, head_lr=2e-5),
    dict(name="t2_multitask_lr2_s44", task=2, method="multitask", split="official", seed=44,
         encoder_lr=2e-5, head_lr=2e-5),
    # metadata fusion, shared encoder
    dict(name="t2_fusion_s42", task=2, method="fusion", text_mode="fusion", seed=42),
    dict(name="t2_fusion_s43", task=2, method="fusion", text_mode="fusion", seed=43),
    dict(name="t2_fusion_s44", task=2, method="fusion", text_mode="fusion", seed=44),
    # metadata fusion, frozen separate encoder
    dict(name="t2_fusion2_s42", task=2, method="fusion2", text_mode="fusion", seed=42),
    dict(name="t2_fusion2_s43", task=2, method="fusion2", text_mode="fusion", seed=43),
    dict(name="t2_fusion2_s44", task=2, method="fusion2", text_mode="fusion", seed=44),
]

ABLATION = [
    dict(name="t2_multitask_weighted_s42", task=2, method="multitask", split="official",
         seed=42, type_loss="weighted"),
    dict(name="t2_single_metadata_s42", task=2, method="single", split="official",
         seed=42, text_mode="metadata"),
]

CALIBRATION = [
    dict(name="t2_single_cal_s42", task=2, method="single", split="official", seed=42),
    dict(name="t2_multitask_cal_s42", task=2, method="multitask", split="official", seed=42),
]

GROUPS = {
    "main": MAIN,
    "task1": TASK1,
    "split": SPLIT,
    "control": CONTROL,
    "ablation": ABLATION,
    "calibration": CALIBRATION,
}

ALL = MAIN + TASK1 + SPLIT + CONTROL + ABLATION + CALIBRATION
