# Improvements - Liora Project 08 (COVID Chest X-rays)

**Reviewer role:** B (IMPROVER)
**Date:** 2026-05-08
**Scope:** Read-only review of `brief.pdf`, `manuscripts/manuscript.md`, `reports/exploration_1.md`, `reports/modeling_1.md`, `reports/modeling_2.md`, `notebooks/01_eda.ipynb`, `notebooks/03_modeling.ipynb`, `manuscripts/references.md`, `deliverables/metrics.json`, `deliverables/presentation.html`, `deliverables/covid_linprobe.joblib`. No files were modified.

---

## Top recommendation (single highest-leverage change)

**Re-run the experiment on a source-stratified split with both an ImageNet ResNet18 baseline AND a chest-X-ray self-supervised backbone (MoCo-CXR ResNet50, Sriram et al. 2021), reporting paired val and test ROC-AUC for both backbones in a single 2x2 table.** This one change converts the project from a single negative-finding reproduction into a controlled experiment that quantifies how much of the val-test gap is dataset-shortcut versus pretraining-domain mismatch. The DeGrave et al. 2021 critique is already cited as the framing; this turns the project into a partial replication of DeGrave's remediation argument rather than only its diagnosis. Estimated cost: 30-45 minutes on a GPU laptop (Sriram weights are 250 MB, fine-tuning is short on 1,200 images). Without this, the manuscript is honest but ends at "we observed the problem"; with it, it ends at "we observed the problem and quantified the fix." This is the single biggest credibility uplift available before the deliverable is shipped to a Liora reviewer.

---

## Detailed weaknesses and recommendations

### 1. No source-stratified evaluation despite source confounding being the central thesis (HIGH)

**Weakness.** The manuscript repeatedly attributes the val-test gap to source confounding (publishers rotated between splits) but never directly demonstrates the source effect. Per-image source labels are extractable from filename prefixes (Cohen GitHub, BIMCV, RICORD, ActMed, RSNA) per the modelling_1 note, yet no source-conditioned analysis is reported. The argument is therefore plausible but unproven on the artefact at hand.

**Action.** Parse the source publisher from each filename in `train.txt`, `val.txt`, `test.txt`. Report (a) per-source class balance for each split, (b) per-source classifier accuracy on the test set, and (c) a chi-squared test of whether predicted-positive rate is independent of source. If the test confusion matrix's "predict positive for everything" pattern correlates with publisher membership rather than label, the source-confounding interpretation is established directly rather than inferred. Add this as a Results subsection; it strengthens the discussion's causal claim materially.

### 2. No saliency / attribution check despite Grad-CAM being mentioned as a "deferable" direction (HIGH)

**Weakness.** The manuscript cites Grad-CAM (ref 24) and SHAP (ref 25) and even names the DeGrave-style saliency methodology, but produces no saliency output. For a manuscript whose central claim is "the probe locks onto non-pulmonary features", a single Grad-CAM panel showing attribution outside the lung field on a high-confidence wrong test prediction would close the loop empirically.

**Action.** Generate Grad-CAM heatmaps (or simpler: integrated gradients on the frozen ResNet18 features, which the paper's setup supports without retraining) for 8 test images: 4 correctly classified and 4 misclassified, balanced across both true classes. Overlay on the original X-ray, compute the fraction of attribution mass falling outside the lung mask (the v5 dataset ships lung masks per `exploration_1.md`). This becomes a single supplementary figure and a one-sentence Results addition. Without it, the shortcut-learning claim stays argumentative rather than demonstrative.

### 3. Single-seed point estimate; no confidence intervals or repeated runs (MEDIUM)

**Weakness.** All metrics are reported as scalars. Section 4 alludes to "5-fold cross-validation" recovering a 0.83-0.85 val and 0.53-0.57 test range but those numbers do not appear in the metrics.json or in any table. A balanced 400-image test set has roughly +/- 0.05 95-percent CI on a single accuracy estimate; this should be quantified.

**Action.** Bootstrap 1000 resamples of each evaluation set, report ROC-AUC and accuracy as point estimate plus 95-percent CI in Table 1. Alternatively run 5 seeds for the linear-probe fit (the LR is convex so seed only affects the train sample if subsampling is reseeded; instead, vary the 1,200-image subsample seed across 5 runs and report mean +/- SD). Either approach belongs in Methods and Results without a structural rewrite.

### 4. Frozen ImageNet ResNet18 is the only model evaluated; no advanced baseline ran (HIGH)

**Weakness.** The brief asks for an "exploration / data visualization / preprocessing report", a "modeling report", and a "final report and GitHub". A typical Liora deliverable at difficulty 7/10 carries at minimum two models (a baseline plus an advanced model). The src folder is empty (no `model_baseline.py` or `model_advanced.py`), modelling_2.md describes an advanced model that is explicitly "reserved for next session", and the presentation reports only the linear probe. The negative-finding framing is honest but does not discharge the brief's modelling-report requirement.

**Action.** At minimum, run a CheXNet-style fine-tuned DenseNet121 (Rajpurkar 2017, ref 13) with 2-3 epochs on the 1,200 train images as the advanced model, and compare in a single table against the linear probe. Even at small budget this gives a paired result. Refactor the modeling notebook into `src/model_baseline.py` (linear probe) and `src/model_advanced.py` (fine-tuned CNN) so the file structure matches the QA expectation and the GitHub deliverable is parseable.

### 5. Lung masks are shipped but unused (MEDIUM)

**Weakness.** `exploration_1.md` section 6 explicitly notes that the v5 release ships per-image lung-field segmentation masks and that "using them to crop or attention-weight the lung region is a strong defence against the dataset-bias issue." Yet the modelling pipeline is mask-blind. Maguolo and Nanni 2021 (ref 29) is cited as showing classifiers retain AUC even with lung regions masked out; the mirror experiment - applying a lung-only crop and seeing whether AUC drops or holds - is the cheapest and most informative ablation available for this dataset.

**Action.** Add a "lung-cropped" variant of the train, val, and test sets by masking everything outside the supplied lung-field mask (or cropping to lung-mask bounding box). Re-run the linear probe under all four conditions: full image, lung-only, masked-out (zero lung), shuffled-mask control. Tabulate ROC-AUC for each. If lung-only AUC is much closer to chance than full-image AUC, that is direct evidence of source-confounder reliance. Three-line addition to Methods, four-row addition to Table 1.

### 6. Reproducibility metadata gaps: no requirements.txt, no seed log, no environment fingerprint (MEDIUM)

**Weakness.** No `requirements.txt`, `environment.yml`, `pyproject.toml`, or pinned-version log is present in the project root or in `src/`. The notebook does not print a deterministic environment dump. The manuscript states the experiment is reproducible on a laptop but a reader cannot reproduce it without re-deriving the dependency set. The brief specifies an "associated GitHub" deliverable; reproducibility is implicitly part of that.

**Action.** Add `requirements.txt` (torch, torchvision, scikit-learn, pillow, numpy, joblib, matplotlib at the versions used) and either an `environment.yml` or a `pip freeze` snapshot. Print `torch.__version__`, `sklearn.__version__`, `np.random.get_state()` checksum, and `torch.manual_seed`/`np.random.seed` settings at the top of `03_modeling.ipynb`. One commit's worth of work; lifts the project from "ran on my laptop" to "rerunnable on a reviewer's laptop."

### 7. Class imbalance and the 4-class task on the available v5 archive are skipped without justification (MEDIUM)

**Weakness.** The actual archive on disk is the 29 GB v5 release that ships four classes (COVID, Normal, Lung Opacity, Viral Pneumonia) per `exploration_1.md`. The decision to drop down to a binary positive-vs-negative formulation because the included `train.txt`/`val.txt`/`test.txt` ship binary labels is defensible but understated: the more interesting clinical task is the 4-class one, especially the COVID-vs-Lung-Opacity pair (the realistic clinical question per `exploration_1.md` section 6). The manuscript treats the binary collapse as the only available task; it isn't.

**Action.** Run a parallel 4-class experiment using class folders directly (ignore the shipped binary manifests, which are COVIDx-style and not native to v5). Sample class-balanced train/val/test from the per-class folders with stratified split. Report 4-class macro F1 and the COVID-vs-Lung-Opacity confusion submatrix. This addresses the more clinically meaningful task and sidesteps the COVIDx-source artefact (which is specific to the rotated-publisher binary manifests). Roughly 1 hour of additional compute; converts the deliverable from "negative finding on shipped manifests" into "negative finding on shipped binary manifests AND clinically-relevant 4-class result."

### 8. Presentation HTML is text-heavy and lacks visual evidence (LOW)

**Weakness.** The slide deck (`presentation.html`) is dark-themed, well-styled, and reads like a written report; from the first 60 lines visible, it is essentially a styled rendering of the manuscript. For a Liora client-style audience, two confusion-matrix figures (val balanced, test biased), a per-class metric bar chart, and the proposed Grad-CAM panel would do more work than additional prose.

**Action.** Add three inline SVG / canvas figures: (a) the two confusion matrices side by side, color-coded; (b) ROC curves for val and test on a single axis with the diagonal reference; (c) a stacked bar showing predicted-positive rate per source publisher in the test set (depends on improvement #1). All inline so the HTML stays self-contained per the QA rule.

---

## Summary table

| # | Improvement | Priority |
|---|---|---|
| 1 | Source-stratified evaluation with per-publisher confusion table | HIGH |
| 2 | Grad-CAM / attribution panel on misclassified test images | HIGH |
| 3 | Bootstrap 95-percent CI / multi-seed metric range | MEDIUM |
| 4 | Add advanced model (fine-tuned DenseNet121) and split into model_baseline.py / model_advanced.py | HIGH |
| 5 | Lung-mask ablations (full / lung-only / masked-out / shuffled) | MEDIUM |
| 6 | Add requirements.txt, env fingerprint, explicit seed logging | MEDIUM |
| 7 | Run the 4-class task using v5 per-class folders, including COVID vs Lung_Opacity | MEDIUM |
| 8 | Add inline figures (confusion matrices, ROC curves, per-source stacked bar) to presentation.html | LOW |

---

## Note on what is already strong

The reviewer should not lose sight of three real strengths the project already has, since the improvements above are deliberately framed as gaps:

- The dataset-mismatch detection (binary COVIDx manifests in a v5-shaped archive) is caught and stated up front rather than glossed over. That is exactly the right behaviour and most undergraduate-level Liora projects miss it.
- The frozen-probe choice is genuinely the right diagnostic instrument for source confounding, and Section 3 of the manuscript defends that choice rigorously rather than presenting a frozen probe as a competitive submission.
- The reference list is curated, all 34 entries carry verifiable identifiers, and the bias / shortcut-learning literature is cited (DeGrave 2021, Roberts 2021, Wynants 2020, Maguolo and Nanni 2021, Geirhos 2020, Zech 2018) at the level of detail a reviewer would expect.

The improvements above are about converting a strong negative-finding note into a strong negative-finding-plus-quantified-remediation paper.
