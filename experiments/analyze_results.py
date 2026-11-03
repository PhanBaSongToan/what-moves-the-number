"""Turn the locked-protocol runs into every table, figure and inline number of the paper.

Run after ``run_matrix.py`` finishes::

    python experiments/analyze_results.py

Artifacts
---------
experiments/summary.json                  aggregates behind the prose
experiments/numbers.tex                   \\newcommand macros consumed by paper.tex
experiments/tables/*.tex                  booktabs tables (table environments)
experiments/figures/confusion.pdf         confusion matrices, single vs multi
experiments/figures/learning_curves.pdf   dev macro-F1 per epoch, mean +- sd
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from itertools import combinations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PAPER = os.path.join(os.path.dirname(HERE), "paper")
TABLES = os.path.join(PAPER, "tables")
FIGURES = os.path.join(PAPER, "figures")
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

RNG = np.random.default_rng(20261103)
ROUNDS = 10000
CLASSES = (0, 1, 2, 3)
EMPTY_GATE = {
    "n": 0,
    "correct": 0,
    "non_spam_to_spam": 0,
    "spam_to_non_spam": 0,
    "spam_type_confusion": 0,
    "type_confusion_by_true": {"1": 0, "2": 0, "3": 0},
}


# --------------------------------------------------------------------------- io
def load_runs() -> dict[str, dict]:
    runs = {}
    for name in sorted(os.listdir(RESULTS)):
        path = os.path.join(RESULTS, name, "final_test_metrics.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                runs[name] = json.load(handle)
    return runs


def predictions(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS, name, "test_predictions.csv"))


# ----------------------------------------------------------------------- metrics
def confusion_from_pairs(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.bincount(y_true * 4 + y_pred, minlength=16).reshape(4, 4)


def macro_f1_from_confusion(matrix: np.ndarray) -> float:
    tp = np.diag(matrix).astype(float)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros(4), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(4), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(4), where=(precision + recall) > 0)
    return float(f1.mean())


def paired_bootstrap(name_a: str, name_b: str) -> dict:
    frame_a, frame_b = predictions(name_a), predictions(name_b)
    y_true = frame_a["y_true"].to_numpy()
    if len(frame_a) != len(frame_b) or not np.array_equal(y_true, frame_b["y_true"].to_numpy()):
        return {}
    pred_a = frame_a["y_pred"].to_numpy()
    pred_b = frame_b["y_pred"].to_numpy()

    point = macro_f1_from_confusion(confusion_from_pairs(y_true, pred_b)) - macro_f1_from_confusion(
        confusion_from_pairs(y_true, pred_a)
    )
    n = len(y_true)
    deltas = np.empty(ROUNDS)
    for i in range(ROUNDS):
        idx = RNG.integers(0, n, n)
        deltas[i] = macro_f1_from_confusion(
            confusion_from_pairs(y_true[idx], pred_b[idx])
        ) - macro_f1_from_confusion(confusion_from_pairs(y_true[idx], pred_a[idx]))
    low, high = np.percentile(deltas, [2.5, 97.5])
    p_value = 2 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0)))
    return {
        "point_delta": float(point),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": min(1.0, p_value),
        "rounds": ROUNDS,
    }


def mcnemar(name_a: str, name_b: str) -> dict:
    frame_a, frame_b = predictions(name_a), predictions(name_b)
    correct_a = frame_a["y_true"].to_numpy() == frame_a["y_pred"].to_numpy()
    correct_b = frame_b["y_true"].to_numpy() == frame_b["y_pred"].to_numpy()
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    total = only_a + only_b
    if total == 0:
        return {"only_a": 0, "only_b": 0, "p_value": 1.0}
    from math import comb

    tail = sum(comb(total, k) for k in range(0, min(only_a, only_b) + 1)) / 2 ** total
    return {"only_a": only_a, "only_b": only_b, "p_value": min(1.0, 2 * tail)}


import sys  # noqa: E402  (kept next to mcnemar for readability)
sys.path.insert(0, HERE)
from experiment_stats import aggregate, gate_decomposition  # noqa: E402


# ------------------------------------------------------------------ formatting
def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}"


def plus_minus(mean: float, sd: float, digits: int = 2) -> str:
    if sd <= 0:
        return pct(mean, digits)
    return f"{pct(mean, digits)} $\\pm$ {pct(sd, digits)}"


def signed(value: float, digits: int = 2) -> str:
    return f"{100 * value:+.{digits}f}"


# ----------------------------------------------------------------------- tables
def table_environment(label: str, caption: str, colspec: str, body: str, star: bool = False) -> str:
    env = "table*" if star else "table"
    return (
        f"\\begin{{{env}}}[t]\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\centering\n\\small\n"
        f"\\begin{{tabular}}{{{colspec}}}\n"
        f"\\toprule\n{body}\n\\bottomrule\n"
        f"\\end{{tabular}}\n"
        f"\\end{{{env}}}\n"
    )


def write_reproduction_table(runs: dict, target: str) -> None:
    def value(prefix: str, field: str) -> str:
        names = [n for n in runs if n.startswith(prefix)]
        if not names:
            return "--"
        values = [runs[n][field] for n in names]
        return plus_minus(float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)

    lexical_path = os.path.join(HERE, "lexical_baseline.json")
    lexical_rows = ""
    if os.path.exists(lexical_path):
        with open(lexical_path, encoding="utf-8") as handle:
            lexical = json.load(handle)
        lexical_rows = (
            "\\midrule\n"
            "\\multicolumn{3}{@{}l}{\\emph{Lexical reference, same release, own protocol}} \\\\\n"
            f"TF-IDF + linear SVM, Task 1 & {pct(lexical['task1']['accuracy'])} & "
            f"{pct(lexical['task1']['macro_f1'])} \\\\\n"
            f"TF-IDF + linear SVM, Task 2 & {pct(lexical['task2']['accuracy'])} & "
            f"{pct(lexical['task2']['macro_f1'])} \\\\\n"
        )

    body = (
        "Setting & Accuracy & Macro-F1 \\\\\n\\midrule\n"
        "Published \\phobert{}, Task 1 \\cite{vispam2024} & 89.23 & 85.58 \\\\\n"
        "Published \\phobert{}, Task 2 \\cite{vispam2024} & 87.79 & 71.56 \\\\\n"
        "\\midrule\n"
        f"This work, Task 1 & {value('t1_single_s', 'test_accuracy')} & {value('t1_single_s', 'test_macro_f1')} \\\\\n"
        f"This work, Task 2 & {value('t2_single_s', 'test_accuracy')} & {value('t2_single_s', 'test_macro_f1')} \\\\\n"
        + lexical_rows
    )
    caption = (
        "Published reference, our locked-protocol reproduction, and a lexical reference point. "
        "Our \\phobert{} rows are mean $\\pm$ standard deviation over three seeds. The public release is "
        "not the published snapshot, so the first block and the rest are not a controlled comparison; "
        "the TF-IDF rows belong to a different model family and are shown only to locate the contextual model."
    )
    with open(os.path.join(TABLES, "reproduction.tex"), "w", encoding="utf-8") as handle:
        handle.write(table_environment("tab:reproduction", caption, "@{}lcc@{}", body))


def write_main_table(state: dict) -> None:
    single, multi = state["single"], state["multi"]
    delta = multi["f1"] - single["f1"]
    boot = state["bootstrap"]
    ci = f"[{signed(boot['ci_low'])}, {signed(boot['ci_high'])}]" if boot else "--"
    p_value = f"{boot['p_value']:.3f}" if boot else "--"
    body = (
        "Model & Seeds & Accuracy & Macro-F1 \\\\\n\\midrule\n"
        f"Single-task \\phobert{{}} & {single['n']} & {plus_minus(single['acc'], single['acc_sd'])} & "
        f"{plus_minus(single['f1'], single['f1_sd'])} \\\\\n"
        f"Multi-task \\phobert{{}} & {multi['n']} & {plus_minus(multi['acc'], multi['acc_sd'])} & "
        f"{plus_minus(multi['f1'], multi['f1_sd'])} \\\\\n"
        "\\midrule\n"
        f"Difference & & {signed(multi['acc'] - single['acc'])} & {signed(delta)} \\\\\n"
        f"95\\% bootstrap CI & & -- & {ci} \\\\\n"
        f"McNemar / bootstrap $p$ & & -- & {p_value} \\\\\n"
    )
    caption = (
        "Main comparison on the official released split (Task 2). Both models follow the protocol in "
        "Table~\\ref{tab:protocol} and differ only in the supervision objective. The last two rows are "
        "computed on seed 42 predictions, which is the seed used for the paired tests."
    )
    with open(os.path.join(TABLES, "main.tex"), "w", encoding="utf-8") as handle:
        handle.write(table_environment("tab:main", caption, "@{}lccc@{}", body))


def write_task1_table(state: dict) -> None:
    dedicated, head = state["task1_single"], state["task1_head"]
    body = (
        "Model & Seeds & Accuracy & Macro-F1 \\\\\n\\midrule\n"
        f"Dedicated binary model & 3 & {plus_minus(dedicated['acc'], dedicated['acc_sd'])} & "
        f"{plus_minus(dedicated['f1'], dedicated['f1_sd'])} \\\\\n"
        f"Multi-task binary head & 3 & {plus_minus(head['acc'], head['acc_sd'])} & "
        f"{plus_minus(head['f1'], head['f1_sd'])} \\\\\n"
    )
    caption = (
        "Task 1. The binary head is read off the multi-task model trained for Task 2, so it costs no "
        "additional training. Values are mean $\\pm$ standard deviation over three seeds."
    )
    with open(os.path.join(TABLES, "task1.tex"), "w", encoding="utf-8") as handle:
        handle.write(table_environment("tab:task1", caption, "@{}lccc@{}", body))


def write_disjoint_table(state: dict) -> None:
    single, multi = state["disjoint_single"], state["disjoint_multi"]
    ci = ""
    if state.get("disjoint_bootstrap"):
        boot = state["disjoint_bootstrap"]
        ci = f"[{signed(boot['ci_low'])}, {signed(boot['ci_high'])}]"
    body = (
        "Model & Seeds & Accuracy & Macro-F1 \\\\\n\\midrule\n"
        f"Single-task \\phobert{{}} & {single['n']} & {plus_minus(single['acc'], single['acc_sd'])} & "
        f"{plus_minus(single['f1'], single['f1_sd'])} \\\\\n"
        f"Multi-task \\phobert{{}} & {multi['n']} & {plus_minus(multi['acc'], multi['acc_sd'])} & "
        f"{plus_minus(multi['f1'], multi['f1_sd'])} \\\\\n"
        f"Difference & & {signed(multi['acc'] - single['acc'])} & {signed(multi['f1'] - single['f1'])} \\\\\n"
        + (f"95\\% bootstrap CI & & -- & {ci} \\\\\n" if ci else "")
    )
    caption = (
        "Product-disjoint evaluation (Task 2). The test set contains no product key seen in training. "
        "Values are mean $\\pm$ standard deviation over three seeds and the interval is a paired "
        "bootstrap on the seed-42 predictions."
    )
    with open(os.path.join(TABLES, "disjoint.tex"), "w", encoding="utf-8") as handle:
        handle.write(table_environment("tab:disjoint", caption, "@{}lccc@{}", body))


def write_ablation_table(state: dict) -> None:
    body = (
        "Variant & Accuracy & Macro-F1 \\\\\n\\midrule\n"
        f"Multi-task (reference) & {pct(state['multi_s42']['acc'])} & {pct(state['multi_s42']['f1'])} \\\\\n"
        f"\\quad + inverse-frequency type loss & {pct(state['weighted']['acc'])} & {pct(state['weighted']['f1'])} \\\\\n"
        f"Single-task (reference) & {pct(state['single_s42']['acc'])} & {pct(state['single_s42']['f1'])} \\\\\n"
        f"\\quad + category/description text & {pct(state['metadata']['acc'])} & {pct(state['metadata']['f1'])} \\\\\n"
    )
    caption = (
        "Interventions re-run under the locked protocol (seed 42). Reference rows repeat the seed-42 "
        "value of the corresponding main configuration so that each ablation is read against its own "
        "baseline."
    )
    with open(os.path.join(TABLES, "ablation.tex"), "w", encoding="utf-8") as handle:
        handle.write(table_environment("tab:ablation", caption, "@{}lcc@{}", body))


def write_perclass_table(state: dict) -> None:
    rows = ["Class & Support & Precision & Recall & F1 \\\\", "\\midrule"]
    names = {0: "Non-spam", 1: "Spam-1", 2: "Spam-2", 3: "Spam-3"}
    rows.append("\\multicolumn{5}{@{}l}{\\emph{Single-task}} \\\\")
    for cls in CLASSES:
        stats = state["perclass_single"][cls]
        rows.append(
            f"{names[cls]} & {int(stats['support'])} & {stats['precision']:.3f} & "
            f"{stats['recall']:.3f} & {stats['f1']:.3f} \\\\"
        )
    rows.append("\\midrule")
    rows.append("\\multicolumn{5}{@{}l}{\\emph{Multi-task}} \\\\")
    for cls in CLASSES:
        stats = state["perclass_multi"][cls]
        rows.append(
            f"{names[cls]} & {int(stats['support'])} & {stats['precision']:.3f} & "
            f"{stats['recall']:.3f} & {stats['f1']:.3f} \\\\"
        )
    caption = (
        "Per-class results on the official test split, averaged over three seeds. Support is the number "
        "of test rows of that class."
    )
    body = "\n".join(rows)
    content = (
        "\\begin{table}[t]\n"
        f"\\caption{{{caption}}}\n"
        "\\label{tab:perclass}\n"
        "\\centering\n\\small\n"
        "\\begin{tabular}{@{}lcccc@{}}\n"
        "\\toprule\n" + body + "\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table}\n"
    )
    with open(os.path.join(TABLES, "perclass.tex"), "w", encoding="utf-8") as handle:
        handle.write(content)


def write_gate_table(state: dict) -> None:
    single, multi = state["gate_single"], state["gate_multi"]
    total = single["n"]
    body = (
        "Error type & Single-task & Multi-task \\\\\n\\midrule\n"
        f"Non-spam $\\rightarrow$ spam & {single['non_spam_to_spam']} & {multi['non_spam_to_spam']} \\\\\n"
        f"Spam $\\rightarrow$ non-spam & {single['spam_to_non_spam']} & {multi['spam_to_non_spam']} \\\\\n"
        f"Spam type confusion & {single['spam_type_confusion']} & {multi['spam_type_confusion']} \\\\\n"
        f"\\quad of which Spam-1 & {single['type_confusion_by_true']['1']} & "
        f"{multi['type_confusion_by_true']['1']} \\\\\n"
        f"\\quad of which Spam-2 & {single['type_confusion_by_true']['2']} & "
        f"{multi['type_confusion_by_true']['2']} \\\\\n"
        f"\\quad of which Spam-3 & {single['type_confusion_by_true']['3']} & "
        f"{multi['type_confusion_by_true']['3']} \\\\\n"
        "\\midrule\n"
        f"Total errors (of {total}) & {total - single['correct']} & {total - multi['correct']} \\\\\n"
    )
    caption = (
        "Decomposition of test errors into detection errors and type errors (official split, seed 42). "
        "A type-confusion error is a spam review labelled with the wrong spam type, i.e.\\ the binary "
        "decision was correct."
    )
    with open(os.path.join(TABLES, "gate.tex"), "w", encoding="utf-8") as handle:
        handle.write(table_environment("tab:gate", caption, "@{}lcc@{}", body))



def write_controls_table(state: dict, runs: dict) -> None:
    """Controls that address the remaining validity concerns."""

    single_ref = state["single"]["f1"]
    multi_ref = state["multi"]["f1"]

    def row(label: str, prefix: str, reference: float) -> str | None:
        try:
            agg = aggregate(runs, prefix)
        except KeyError:
            return None
        mean = agg["f1"]["mean"]
        delta = 100 * (mean - reference)
        return (
            f"{label} & {agg['acc']['n']} & {plus_minus(agg['acc']['mean'], agg['acc']['sd'])} & "
            f"{plus_minus(mean, agg['f1']['sd'])} & {delta:+.2f} \\\\"
        )

    rows = [
        "Variant & Seeds & Accuracy & Macro-F1 & $\\Delta$ \\\\",
        "\\midrule",
        row("Single-task (reference)", "t2_single_s", single_ref),
        row("\\quad deduplicated", "t2_single_dedup_s", single_ref),
        row("\\quad XLM-R base", "t2_single_xlmr_s", single_ref),
        row("\\quad 128 tokens", "t2_single_len128_s", single_ref),
        row("\\quad single LR", "t2_single_lr2_s", single_ref),
        "\\midrule",
        row("Multi-task (reference)", "t2_multitask_s", multi_ref),
        row("\\quad deduplicated", "t2_multitask_dedup_s", multi_ref),
        row("\\quad single LR", "t2_multitask_lr2_s", multi_ref),
        "\\midrule",
        row("Description fusion", "t2_fusion_s", single_ref),
    ]
    body = "\n".join(item for item in rows if item)
    caption = (
        "Every control re-run against its own reference under the locked protocol. $\\Delta$ is the "
        "macro-F1 difference against the matching reference mean, in percentage points. The deduplicated "
        "rows drop every training review whose text also occurs in the test split; the fusion row encodes "
        "the product description in a second forward pass and concatenates it with the review vector, so "
        "the description is no longer truncated."
    )
    content = (
        "\\begin{table}[t]\n"
        f"\\caption{{{caption}}}\n"
        "\\label{tab:controls}\n"
        "\\centering\n\\small\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{@{}lcccc@{}}\n"
        "\\toprule\n" + body + "\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table}\n"
    )
    with open(os.path.join(TABLES, "controls.tex"), "w", encoding="utf-8") as handle:
        handle.write(content)


def aggregate_mean(runs: dict, prefix: str) -> dict:
    """Mean accuracy and macro-F1 over the runs matching ``prefix``."""
    try:
        agg = aggregate(runs, prefix)
    except KeyError:
        return {"acc": 0.0, "f1": 0.0, "n": 0}
    return {"acc": agg["acc"]["mean"], "f1": agg["f1"]["mean"], "n": agg["acc"]["n"]}


def write_effects_table(state: dict) -> None:
    """One row per design choice, so the paper can say what actually moves the metric."""
    single, multi = state["single"], state["multi"]
    rows = [
        "Factor & Change & $\\Delta$ Macro-F1 \\\\",
        "\\midrule",
        f"Supervision & flat $\\rightarrow$ cascade, official & {signed(multi['f1'] - single['f1'])} \\\\",
        f" & same, text-deduplicated & "
        f"{signed(state['dedupe_multi']['f1'] - state['dedupe_single']['f1'])} \\\\",
        f" & same, product-disjoint & "
        f"{signed(state['disjoint_multi']['f1'] - state['disjoint_single']['f1'])} \\\\",
        f"Metadata & $+$ description fusion & {signed(state['fusion']['f1'] - single['f1'])} \\\\",
        f"Backbone & \\phobert{{}} $\\rightarrow$ XLM-R & {signed(state['xlmr']['f1'] - single['f1'])} \\\\",
        f"Window & 100 $\\rightarrow$ 128 tokens & {signed(state['len128']['f1'] - single['f1'])} \\\\",
        f"Schedule & two-rate $\\rightarrow$ single $2\\times10^{{-5}}$ & "
        f"{signed(state['lr_single']['f1'] - single['f1'])} / {signed(state['lr_multi']['f1'] - multi['f1'])} \\\\",
        f"Evaluation & full $\\rightarrow$ deduplicated, flat & "
        f"{signed(state['dedupe_single']['f1'] - single['f1'])} \\\\",
        f" & full $\\rightarrow$ deduplicated, cascade & "
        f"{signed(state['dedupe_multi']['f1'] - multi['f1'])} \\\\",
        "\\midrule",
        f"Seed noise & standard deviation, flat / cascade & "
        f"$\\pm${pct(single['f1_sd'])} / $\\pm${pct(multi['f1_sd'])} \\\\",
    ]
    caption = (
        "Which choices move macro-F1. Every $\\Delta$ is measured against the matching reference in "
        "Tables~\\ref{tab:main}--\\ref{tab:controls}; the last row is the seed-to-seed standard "
        "deviation of the two models, which sets the scale for judging the others."
    )
    body = "\n".join(rows)
    content = (
        "\\begin{table}[t]\n"
        f"\\caption{{{caption}}}\n"
        "\\label{tab:effects}\n"
        "\\centering\n\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}llc@{}}\n"
        "\\toprule\n" + body + "\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table}\n"
    )
    with open(os.path.join(TABLES, "effects.tex"), "w", encoding="utf-8") as handle:
        handle.write(content)


def write_numbers(state: dict) -> None:
    single, multi = state["single"], state["multi"]
    boot = state.get("bootstrap") or {}
    lines = [
        "% Generated by experiments/analyze_results.py -- do not edit by hand.",
        f"\\newcommand{{\\SingleAcc}}{{{pct(single['acc'])}}}",
        f"\\newcommand{{\\SingleFone}}{{{pct(single['f1'])}}}",
        f"\\newcommand{{\\MultiAcc}}{{{pct(multi['acc'])}}}",
        f"\\newcommand{{\\MultiFone}}{{{pct(multi['f1'])}}}",
        f"\\newcommand{{\\SingleFoneSD}}{{{pct(single['f1_sd'])}}}",
        f"\\newcommand{{\\MultiFoneSD}}{{{pct(multi['f1_sd'])}}}",
        f"\\newcommand{{\\DeltaFone}}{{{signed(multi['f1'] - single['f1'])}}}",
        f"\\newcommand{{\\BootLow}}{{{signed(boot.get('ci_low', 0.0))}}}",
        f"\\newcommand{{\\BootHigh}}{{{signed(boot.get('ci_high', 0.0))}}}",
        f"\\newcommand{{\\BootP}}{{{boot.get('p_value', 1.0):.3f}}}",
        f"\\newcommand{{\\McNemarP}}{{{state.get('mcnemar', {}).get('p_value', 1.0):.3f}}}",
        f"\\newcommand{{\\SingleTOneAcc}}{{{pct(state['task1_single']['acc'])}}}",
        f"\\newcommand{{\\SingleTOneFone}}{{{pct(state['task1_single']['f1'])}}}",
        f"\\newcommand{{\\MultiTOneAcc}}{{{pct(state['task1_head']['acc'])}}}",
        f"\\newcommand{{\\MultiTOneFone}}{{{pct(state['task1_head']['f1'])}}}",
        f"\\newcommand{{\\DiscSingleAcc}}{{{pct(state['disjoint_single']['acc'])}}}",
        f"\\newcommand{{\\DiscSingleFone}}{{{pct(state['disjoint_single']['f1'])}}}",
        f"\\newcommand{{\\DiscMultiAcc}}{{{pct(state['disjoint_multi']['acc'])}}}",
        f"\\newcommand{{\\DiscMultiFone}}{{{pct(state['disjoint_multi']['f1'])}}}",
        f"\\newcommand{{\\DiscDelta}}{{{signed(state['disjoint_multi']['f1'] - state['disjoint_single']['f1'])}}}",
        f"\\newcommand{{\\WeightedFone}}{{{pct(state['weighted']['f1'])}}}",
        f"\\newcommand{{\\MetaFone}}{{{pct(state['metadata']['f1'])}}}",
        f"\\newcommand{{\\GateFP}}{{{state['gate_single']['non_spam_to_spam']}}}",
        f"\\newcommand{{\\GateFN}}{{{state['gate_single']['spam_to_non_spam']}}}",
        f"\\newcommand{{\\GateType}}{{{state['gate_single']['spam_type_confusion']}}}",
        f"\\newcommand{{\\GateFPm}}{{{state['gate_multi']['non_spam_to_spam']}}}",
        f"\\newcommand{{\\GateFNm}}{{{state['gate_multi']['spam_to_non_spam']}}}",
        f"\\newcommand{{\\GateTypem}}{{{state['gate_multi']['spam_type_confusion']}}}",
        f"\\newcommand{{\\RunMinutes}}{{{state.get('run_minutes', 8):.0f}}}",
        f"\\newcommand{{\\DedupSingleFone}}{{{pct(state['dedupe_single']['f1'])}}}",
        f"\\newcommand{{\\DedupMultiFone}}{{{pct(state['dedupe_multi']['f1'])}}}",
        f"\\newcommand{{\\DedupRemoved}}{{{state['dedupe_removed']}}}",
        f"\\newcommand{{\\XlmrFone}}{{{pct(state['xlmr']['f1'])}}}",
        f"\\newcommand{{\\LenFone}}{{{pct(state['len128']['f1'])}}}",
        f"\\newcommand{{\\FusionFone}}{{{pct(state['fusion']['f1'])}}}",
        f"\\newcommand{{\\FusionAcc}}{{{pct(state['fusion']['acc'])}}}",
        f"\\newcommand{{\\LrSingleFone}}{{{pct(state['lr_single']['f1'])}}}",
        f"\\newcommand{{\\LrMultiFone}}{{{pct(state['lr_multi']['f1'])}}}",
        f"\\newcommand{{\\EceSingle}}{{{100 * state.get('calibration', {}).get('single', {}).get('ece', 0.0):.1f}}}",
        f"\\newcommand{{\\EceMulti}}{{{100 * state.get('calibration', {}).get('multi', {}).get('ece', 0.0):.1f}}}",
    ]
    lexical_path = os.path.join(HERE, "lexical_baseline.json")
    if os.path.exists(lexical_path):
        with open(lexical_path, encoding="utf-8") as handle:
            lexical = json.load(handle)
        lines += [
            f"\\newcommand{{\\LexAcc}}{{{pct(lexical['task2']['accuracy'])}}}",
            f"\\newcommand{{\\LexFone}}{{{pct(lexical['task2']['macro_f1'])}}}",
            f"\\newcommand{{\\LexTOneAcc}}{{{pct(lexical['task1']['accuracy'])}}}",
            f"\\newcommand{{\\LexTOneFone}}{{{pct(lexical['task1']['macro_f1'])}}}",
        ]
    with open(os.path.join(PAPER, "numbers.tex"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------- figures
def plot_confusion(runs: dict) -> None:
    panels = [("t2_single_s42", "Single-task"), ("t2_multitask_s42", "Multi-task")]
    panels = [item for item in panels if item[0] in runs]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.45, 1.95), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for axis, (name, title) in zip(axes, panels):
        matrix = np.asarray(runs[name]["confusion_matrix"], dtype=float)
        normalised = matrix / matrix.sum(axis=1, keepdims=True)
        axis.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
        for row in range(4):
            for col in range(4):
                value = normalised[row, col]
                axis.text(
                    col, row, f"{100 * value:.0f}", ha="center", va="center", fontsize=5.5,
                    color="white" if value > 0.55 else "black",
                )
        axis.set_xticks(range(4))
        axis.set_yticks(range(4))
        axis.set_xticklabels(["NS", "S1", "S2", "S3"], fontsize=5.5)
        axis.set_yticklabels(["NS", "S1", "S2", "S3"], fontsize=5.5)
        axis.set_xlabel("Predicted", fontsize=6)
        axis.set_ylabel("True", fontsize=6)
        axis.set_title(title, fontsize=6.5)
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.savefig(os.path.join(FIGURES, "confusion.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_curves(runs: dict) -> None:
    series = {}
    for prefix, label in (("t2_single_s", "Single-task"), ("t2_multitask_s", "Multi-task")):
        curves = [
            [row["dev_macro_f1"] for row in run["history"]]
            for name, run in runs.items()
            if name.startswith(prefix) and "disjoint" not in name and "weighted" not in name
        ]
        if curves:
            series[label] = curves
    if not series:
        return
    fig, axis = plt.subplots(figsize=(3.4, 2.2))
    for label, curves in series.items():
        width = min(len(curve) for curve in curves)
        grid = np.array([curve[:width] for curve in curves])
        epochs = np.arange(1, width + 1)
        axis.plot(epochs, grid.mean(axis=0), marker="o", markersize=3, linewidth=1.3, label=label)
        if len(curves) > 1:
            axis.fill_between(
                epochs, grid.mean(axis=0) - grid.std(axis=0), grid.mean(axis=0) + grid.std(axis=0),
                alpha=0.16, linewidth=0,
            )
    axis.set_xlabel("Epoch", fontsize=7)
    axis.set_ylabel("Development macro-F1", fontsize=7)
    axis.tick_params(labelsize=7)
    axis.legend(fontsize=7, frameon=False)
    axis.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "learning_curves.pdf"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------- main
def main() -> None:
    runs = load_runs()
    if not runs:
        raise SystemExit("no finished runs in experiments/results")

    state: dict = {"run_minutes": float(np.median([
        run.get("elapsed_seconds", 0) for run in runs.values()
    ])) / 60.0}

    single = safe_aggregate(runs, "t2_single_s")
    multi = safe_aggregate(runs, "t2_multitask_s")
    state["single"] = {"acc": single["acc"]["mean"], "acc_sd": single["acc"]["sd"],
                       "f1": single["f1"]["mean"], "f1_sd": single["f1"]["sd"],
                       "n": single["acc"]["n"]}
    state["multi"] = {"acc": multi["acc"]["mean"], "acc_sd": multi["acc"]["sd"],
                      "f1": multi["f1"]["mean"], "f1_sd": multi["f1"]["sd"],
                      "n": multi["acc"]["n"]}

    task1 = safe_aggregate(runs, "t1_single_s")
    state["task1_single"] = {"acc": task1["acc"]["mean"], "acc_sd": task1["acc"]["sd"],
                             "f1": task1["f1"]["mean"], "f1_sd": task1["f1"]["sd"]}
    head = safe_aggregate(runs, "t2_multitask_s", field="test_binary")
    state["task1_head"] = {"acc": head["acc"]["mean"], "acc_sd": head["acc"]["sd"],
                           "f1": head["f1"]["mean"], "f1_sd": head["f1"]["sd"]}

    disjoint_single = safe_aggregate(runs, "t2_single_disjoint")
    disjoint_multi = safe_aggregate(runs, "t2_multitask_disjoint")
    state["disjoint_single"] = {
        "acc": disjoint_single["acc"]["mean"], "acc_sd": disjoint_single["acc"]["sd"],
        "f1": disjoint_single["f1"]["mean"], "f1_sd": disjoint_single["f1"]["sd"],
        "n": disjoint_single["acc"]["n"],
    }
    state["disjoint_multi"] = {
        "acc": disjoint_multi["acc"]["mean"], "acc_sd": disjoint_multi["acc"]["sd"],
        "f1": disjoint_multi["f1"]["mean"], "f1_sd": disjoint_multi["f1"]["sd"],
        "n": disjoint_multi["acc"]["n"],
    }

    state["multi_s42"] = single_run_metrics(runs, "t2_multitask_s42")
    state["single_s42"] = single_run_metrics(runs, "t2_single_s42")
    state["dedupe_single"] = aggregate_mean(runs, "t2_single_dedup_s")
    state["dedupe_multi"] = aggregate_mean(runs, "t2_multitask_dedup_s")
    state["xlmr"] = aggregate_mean(runs, "t2_single_xlmr_s")
    state["len128"] = aggregate_mean(runs, "t2_single_len128_s")
    state["fusion"] = aggregate_mean(runs, "t2_fusion_s")
    state["lr_single"] = aggregate_mean(runs, "t2_single_lr2_s")
    state["lr_multi"] = aggregate_mean(runs, "t2_multitask_lr2_s")
    state["dedupe_removed"] = runs.get("t2_single_dedup_s42", {}).get("removed_train_rows", 0)
    state["weighted"] = single_run_metrics(runs, "t2_multitask_weighted_s42")
    state["metadata"] = single_run_metrics(runs, "t2_single_metadata_s42")

    state["bootstrap"] = (
        paired_bootstrap("t2_single_s42", "t2_multitask_s42")
        if {"t2_single_s42", "t2_multitask_s42"} <= set(runs)
        else {}
    )
    state["mcnemar"] = (
        mcnemar("t2_single_s42", "t2_multitask_s42")
        if {"t2_single_s42", "t2_multitask_s42"} <= set(runs)
        else {"p_value": 1.0}
    )
    if "t2_single_disjoint_s42" in runs and "t2_multitask_disjoint_s42" in runs:
        state["disjoint_bootstrap"] = paired_bootstrap(
            "t2_single_disjoint_s42", "t2_multitask_disjoint_s42"
        )

    state["perclass_single"] = per_class_average(runs, "t2_single_s")
    state["perclass_multi"] = per_class_average(runs, "t2_multitask_s")
    state["gate_single"] = (
        gate_decomposition(predictions("t2_single_s42")) if "t2_single_s42" in runs else EMPTY_GATE
    )
    if {"t2_single_s42", "t2_multitask_s42"} <= set(runs):
        state["calibration"] = {
            "single": expected_calibration_error(predictions("t2_single_s42")),
            "multi": expected_calibration_error(predictions("t2_multitask_s42")),
        }
    state["gate_multi"] = (
        gate_decomposition(predictions("t2_multitask_s42")) if "t2_multitask_s42" in runs else EMPTY_GATE
    )

    write_numbers(state)
    write_reproduction_table(runs, PAPER)
    write_main_table(state)
    write_task1_table(state)
    write_disjoint_table(state)
    write_ablation_table(state)
    write_perclass_table(state)
    write_gate_table(state)
    write_controls_table(state, runs)
    write_effects_table(state)
    plot_confusion(runs)
    plot_curves(runs)

    with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"runs": sorted(runs), "state": state}, handle, indent=2, ensure_ascii=False)
    print(json.dumps(state, indent=2, ensure_ascii=False))



def expected_calibration_error(frame, bins: int = 10) -> dict:
    """Top-label expected calibration error on the test predictions."""
    probabilities = frame[["p0", "p1", "p2", "p3"]].to_numpy(dtype=float)
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == frame["y_true"].to_numpy()).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return {"ece": float(error), "mean_confidence": float(confidence.mean()),
            "accuracy": float(correct.mean()), "bins": bins}


def per_class_average(runs: dict, prefix: str) -> dict:
    names = [n for n in runs if n.startswith(prefix)]
    out = {}
    for cls in CLASSES:
        rows = [runs[n]["test_per_class"][str(cls)] for n in names]
        if not rows:
            out[cls] = {"support": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue
        out[cls] = {
            "support": rows[0]["support"],
            "precision": float(np.mean([row["precision"] for row in rows])),
            "recall": float(np.mean([row["recall"] for row in rows])),
            "f1": float(np.mean([row["f1"] for row in rows])),
        }
    return out


def safe_aggregate(runs: dict, prefix: str, field: str | None = None) -> dict:
    """Like :func:`aggregate`, but returns zeros instead of raising when a group is missing."""
    try:
        return aggregate(runs, prefix, field)
    except KeyError:
        print(f"[warn] no finished run matches {prefix!r}; writing placeholder values")
        empty = {"mean": 0.0, "sd": 0.0, "values": [], "n": 0, "runs": []}
        return {"acc": dict(empty), "f1": dict(empty)}


def single_run_metrics(runs: dict, name: str) -> dict:
    if name in runs:
        return {"acc": runs[name]["test_accuracy"], "f1": runs[name]["test_macro_f1"]}
    print(f"[warn] missing run {name!r}; writing placeholder values")
    return {"acc": 0.0, "f1": 0.0}


if __name__ == "__main__":
    main()
