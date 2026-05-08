# CHANGES - Project 08 (COVID Chest X-rays)

**Date:** 2026-05-08
**Scope:** Reconciliation of manuscript-vs-code drift flagged in `validation_report.md` Task 6.

---

## Problem

The validation pass returned PASS-WITH-WARNINGS. The principal warning was
method drift between deliverable artefacts and committed code:

- `manuscripts/manuscript.md` Methods (Section 3) describes a binary
  positive-vs-negative pipeline using a frozen ImageNet ResNet18 feature
  extractor followed by `sklearn.linear_model.LogisticRegression`
  (`C=0.1`, `class_weight='balanced'`, `max_iter=2000`), persisted to
  `deliverables/covid_linprobe.joblib`.
- `deliverables/metrics.json` numerically matches the manuscript Table 1
  exactly: val accuracy 0.8375, val ROC-AUC 0.8944, val confusion
  `[[168, 32], [33, 167]]`, test accuracy 0.55, test ROC-AUC 0.578,
  test confusion `[[44, 156], [24, 176]]`. Manuscript prose is
  faithful to the artefact.
- `notebooks/03_modeling.ipynb` is a different experiment: a 4-class
  ResNet18 fine-tune on a source-derived label scheme
  (COVID_BIMCV, COVID_RICORD, Normal_RSNA, Other) that writes its own
  `covid_resnet18.pt` and a 4-class `metrics.json`. It is not the
  driver behind the binary linprobe artefact.
- `src/` was empty. The driver script that produced `covid_linprobe.joblib`
  and the binary `metrics.json` is not in the repository.

The drift is therefore between the Methods section and the committed
source code, not between the Methods section and the shipped numbers.

## Decision

The fitted joblib and the metrics.json are the canonical record of the
binary experiment, and the manuscript prose is consistent with both.
The honest reconciliation is to:

1. Treat the manuscript and the deliverable artefacts as the source of
   truth for what the project reports.
2. Acknowledge in the Methods section that the committed notebook is an
   exploratory 4-class fine-tune and is NOT the production code for the
   reported linprobe result.
3. Add a placeholder file at `src/model_baseline.py` that documents the
   gap and points the reader at the canonical artefacts, instead of
   fabricating a script after the fact.

No source code was reconstructed or invented. No reported numbers were
modified. No verification claim has been changed.

## Files changed

### Modified

- `manuscripts/manuscript.md`
  - Added subsection `### 3.1 Note on the committed source code` at the
    end of Section 3 (Methods).
  - Content: states that `metrics.json` and `covid_linprobe.joblib` are
    the canonical artefacts of the binary linprobe pipeline, that the
    driver script is not committed, that `03_modeling.ipynb` is an
    unrelated 4-class fine-tune, and that `src/model_baseline.py` is a
    placeholder pointing at the joblib.
  - No changes to the Abstract, Results, Discussion, Conclusion, or
    Table 1.
  - No headline number was rewritten.

### Added

- `src/model_baseline.py` (placeholder, raises `NotImplementedError`)
  - Documents the missing driver, names the canonical artefacts, gives
    a remediation recipe for any future rerun, and refuses to run.
  - Does not contain a fabricated implementation.

### Not changed

- `deliverables/covid_linprobe.joblib` (untouched)
- `deliverables/metrics.json` (untouched)
- `deliverables/presentation.html` (untouched)
- `notebooks/01_eda.ipynb` (untouched)
- `notebooks/03_modeling.ipynb` (untouched, but explicitly disclaimed in
  Methods 3.1 as exploratory and not the production code)
- `manuscripts/references.md`, `manuscripts/references.bib` (untouched)
- `reports/*` (untouched)

## What this does NOT fix

The following items from `improvements.md` remain open and were
deliberately not addressed in this commit, because they require new
experiments rather than text reconciliation:

1. Source-stratified evaluation with per-publisher confusion table.
2. Grad-CAM / attribution panel on misclassified test images.
3. Bootstrap 95-percent confidence intervals or multi-seed metric range.
4. A real advanced-model run (e.g. fine-tuned DenseNet121) with paired
   metrics against the linear probe.
5. Lung-mask ablations (full / lung-only / masked-out / shuffled).
6. `requirements.txt`, environment fingerprint, explicit seed log.
7. 4-class evaluation on the v5 per-class folders, including
   COVID-vs-Lung-Opacity.
8. Inline figures (confusion matrices, ROC curves, per-source stacked
   bar) in `presentation.html`.

The `checkpoint.json` schema gap from `validation_report.md` Task 11 is
also still open.

## Verification of this change

- The added Methods subsection 3.1 contains no numbers that were not
  already present elsewhere in the manuscript or in `metrics.json`.
- The placeholder `src/model_baseline.py` contains no model code; it
  only raises `NotImplementedError` and documents the gap.
- No em-dash characters were introduced in either file (verified by
  visual scan during authoring; both files use hyphens, commas, and
  parentheses only).
- No "verified by N agents" or similar AI-tell phrasing was used.
- This `CHANGES.md` itself contains no em-dashes and no AI-tell phrasing.
