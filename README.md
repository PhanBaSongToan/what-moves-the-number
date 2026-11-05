# What Moves the Number?

**Evaluation and Encoder Choices Dominate Architecture in Vietnamese Spam-Review Classification**

A controlled study of Vietnamese spam-review type classification on ViSpamReviews v2.
Thirty-four runs under one locked protocol measure how much each design choice moves
macro-F1: the multi-task objective and metadata fusion move it by at most a tenth of a
point, changing the backbone costs 1.5, and changing the evaluation (deduplicating
training text, or holding products out) moves it by 1.4 to 2.3 points and reverses the
ranking of the two models.

The paper is `paper/paper.pdf` (source in `paper/paper.tex`, 8 pages, IEEEtran + XeLaTeX).

## Layout

| Path | Contents |
|---|---|
| `paper/` | **The paper.** `paper.tex`, generated tables and figures, `numbers.tex`, `build.ps1`. |
| `experiments/` | Code and results. See `experiments/README.md`. |
| `review_20261103/` | Audit of the previous draft: findings, severity, evidence, verification scripts. |
| `archive/` | Superseded drafts, including the 2026-11-02 LNCS draft. |
| `.github/workflows/` | CI: checks that the committed tables match the results, and builds the PDF. |

## Get the data

The dataset is public but gated on Hugging Face:
<https://huggingface.co/datasets/sonlam1102/vispamdetection2>

Download the release and put the archive at `data/vispamdetectionv2_official.zip`
(it contains `train.csv`, `dev.csv`, `test.csv`). Alternatively point
`VISPAM_ZIP` at the file. Then build the product-disjoint split once:

```powershell
python experiments/build_product_disjoint_split.py
```

`experiments/config.py` resolves both paths and explains what is missing if they are not there.

## Build the paper

Requires MiKTeX or TeX Live with `ieeetran` and XeLaTeX; Vietnamese review text needs a
Unicode engine, so it is XeLaTeX and not pdfLaTeX. The font is Times New Roman on
Windows/macOS and falls back to TeX Gyre Termes elsewhere.

```powershell
powershell -ExecutionPolicy Bypass -File paper/build.ps1
```

The output is `paper/paper.pdf`.

## Reproduce the experiments

```powershell
pip install -r experiments/requirements.txt
python experiments/run_all.py
```

`run_all.py` runs every configuration in order (finished runs are skipped, so it also
resumes), then regenerates `paper/tables/`, `paper/figures/` and `paper/numbers.tex`.
It needs a GPU; the analysis at the end does not.

## What changed on 2026-11-03

The first draft was audited and found not submittable: the baseline and the multi-task
model had been trained with different protocols, two headline numbers had no stored
metrics, the PDF was a stale IEEEtran build of a one-author version while the source was
an LNCS draft with three authors, and the draft cited the wrong first author for the
reference paper. The current version addresses all of that:

* one locked protocol for every model (`experiments/run_experiment.py`), 53 runs;
* three seeds for the headline comparison, paired bootstrap and McNemar test;
* both models evaluated on the product-disjoint split over three seeds;
* controls that answer the obvious objections: text-deduplicated training data, an XLM-R
  backbone, a 128-token window and a single-rate schedule;
* a metadata-fusion baseline that encodes the description separately, so the "metadata
  was truncated" confound is removed;
* ablations re-run under the same protocol instead of a mixed-protocol table;
* a verbatim-duplicate audit on top of the product-group audit;
* label definitions with real examples, dataset statistics, error decomposition,
  calibration, per-class tables and confusion matrices;
* an effect-size table that puts every choice on one scale against seed noise;
* IEEEtran source that compiles on Windows and Linux, with the three authors.

### What the results say

| Setting | Macro-F1 |
|---|---|
| Single-task PhoBERT, official split (3 seeds) | 76.10 ± 0.71 |
| Multi-task PhoBERT, official split (3 seeds) | 76.02 ± 0.96 |
| Single-task, product-disjoint (3 seeds) | 75.94 |
| Multi-task, product-disjoint (3 seeds) | 73.98 |
| Single-task, text-deduplicated (3 seeds) | 74.72 ± 1.34 |
| Multi-task, text-deduplicated (3 seeds) | 77.02 ± 0.77 |
| Single-task, near-duplicates removed (3 seeds) | 74.40 ± 0.80 |
| Multi-task, near-duplicates removed (3 seeds) | 73.18 ± 2.51 |
| Single-task, teencode normalised (3 seeds) | 75.86 ± 2.20 |
| Multi-task, teencode normalised (3 seeds) | 74.65 ± 1.49 |
| Fusion, shared encoder (3 seeds) | 75.98 ± 1.15 |
| Fusion, frozen encoder (3 seeds) | 75.95 ± 0.75 |
| Single-task, XLM-R backbone (3 seeds) | 74.63 ± 1.55 |
| Single-task, 128-token window (3 seeds) | 76.49 ± 0.46 |
| Single-task, single learning rate (3 seeds) | 74.41 ± 0.37 |
| Multi-task, single learning rate (3 seeds) | 74.75 ± 1.65 |
| Category-disjoint, single-task | 72.25 |
| Category-disjoint, multi-task | 71.59 |
| Seed noise (standard deviation) | 0.71 / 0.96 |

The product-disjoint split is one draw: further partitions move the flat model
between 75.22 and 75.94 and the cascade between 73.98 and 77.37 — the partition
alone is worth almost three points on the cascade. A second corpus (Ott et al.,
deceptive opinion spam) gives 83.58 ± 1.64 on a random split and 80.78 ± 0.20
when hotels are held out, i.e. the same evaluation effect on a different language.

The hierarchy gives no gain on the official split, is behind on unseen products, and is
ahead once duplicated training text is removed. Metadata never helps, the backbone matters
more than the objective, and the only intervention that clearly helps is text normalisation
(+1.46 for the flat model). The conclusion is that the ranking depends on how the
evaluation is built.

The full audit is in `review_20261103/BAO_CAO_KIEM_TRA_BAI_BAO.md` (Vietnamese).

## Notes

* Model checkpoints (~540 MB each) are regenerated by the training script and are not
  committed; the metrics and per-row predictions in `experiments/results/` are.
* `experiments/legacy/` holds the pre-lock exploratory scripts. Their numbers are not
  comparable with the paper and must not be mixed into it.
* Other course deliverables are kept outside this repository (they are ignored by
  `.gitignore` if present locally).
* License: MIT (see `LICENSE`).
