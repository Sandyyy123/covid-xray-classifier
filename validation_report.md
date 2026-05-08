# Validation Report - Project #08 COVID Chest X-rays

## Compact Summary

**Overall: PASS-WITH-WARNINGS**

The deliverable bundle (manuscript + metrics + saved model + presentation) is internally consistent for a binary linear-probe study on COVIDx-style splits. Notebooks parse, references resolve live on CrossRef, IMRaD is complete, the manuscript is in target word range, the HTML is fully self-contained, and zero em-dashes or AI-tell phrases appear. Two warnings: (1) `src/` is empty, so the binary linear-probe pipeline described in Methods has no committed script - the only modeling notebook (`03_modeling.ipynb`) implements a different 4-class fine-tune experiment, which is method drift between deliverable artefacts and the manuscript narrative; (2) `checkpoint.json` and `brief.md` are absent (only `brief.pdf` and `deliverables/metrics.json` exist). No FAILs. Saved model artefact `covid_linprobe.joblib` is present and metrics match the manuscript headline numbers.

---

## Task Findings

### 1. Notebook validity
- [PASS] `notebooks/01_eda.ipynb` parses as JSON, 17 cells.
- [PASS] `notebooks/03_modeling.ipynb` parses as JSON, 18 cells.

### 2. Python script syntax
- [WARN] `src/` directory is empty. Neither `model_baseline.py` nor `model_advanced.py` exists. Modeling code lives only in notebooks.

### 3. Manuscript word count
- [PASS] `manuscripts/manuscript.md` = 4,718 words. Within 4,000-5,000 target.

### 4. Self-contained HTML
- [PASS] `deliverables/presentation.html` has 0 hits for `href="http` or `src="http`. Fully inline.

### 5. IMRaD completeness
- [PASS] All required sections present: Title, Abstract, Introduction (1.), Data (2.), Methods (3.), Results (4.), Discussion (5.), Conclusion (6.), References. "Data" used in place of a separate Materials section, which is acceptable. Note: standard IMRaD does not require a separate "Title" header line; title is the H1, which is present.

### 6. Method drift
- [WARN] Manuscript Methods names: ResNet18, IMAGENET1K_V1 weights, frozen backbone, `nn.Identity` head replacement, ImageNet normalisation `[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]`, `sklearn.linear_model.LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000)`, accuracy / ROC-AUC / F1 / macro-F1 / classification_report / confusion_matrix, joblib persistence to `covid_linprobe.joblib`.
- The only committed modeling notebook (`03_modeling.ipynb`) implements a **4-class** ResNet18 fine-tune (CLASS_NAMES = COVID_BIMCV, COVID_RICORD, Normal_RSNA, Other) with augmentation, class weights, and `torch.save(... covid_resnet18.pt)`. There is **no LogisticRegression / linear-probe / joblib code** in the notebook. The binary linear-probe pipeline that produced `deliverables/covid_linprobe.joblib` and `deliverables/metrics.json` is not represented anywhere in the repo `src/` or `notebooks/`.
- The deliverable `metrics.json` numerically matches the manuscript headline (val acc 0.8375, val AUC 0.8944, test acc 0.55, test AUC 0.578, val CM `[[168,32],[33,167]]`), so the manuscript is faithful to the artefact - but the script that produced the artefact is missing. Drift is between Methods and committed code, not between Methods and metrics.

### 7. Citation drift
- [PASS] Inline citations use bracketed numeric form `[N]`. Used set: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32. References list contains entries 1-34. Every inline citation maps to an entry. No orphans.
- [PASS] Refs 13 (CheXNet), 16 (Inception), 23 (Focal Loss), 33 (Esteva), 34 (Litjens) are listed but never cited inline. Unused references are not orphans by the rule (only inline-without-entry would fail), flagging here as housekeeping note.

### 8. Re-verify 5 random references (CrossRef live)
All 5 returned HTTP 200 with title-match against the reference text:
- [PASS] Ref 1 - DOI 10.1109/ACCESS.2020.3010287 - "Can AI Help in Screening Viral and COVID-19 Pneumonia?"
- [PASS] Ref 2 - DOI 10.1016/j.compbiomed.2021.104319 - "Exploring the effect of image enhancement techniques on COVID-19 detection..."
- [PASS] Ref 14 - DOI 10.1109/CVPR.2016.90 - "Deep Residual Learning for Image Recognition"
- [PASS] Ref 26 - DOI 10.1038/s42256-021-00338-7 - "AI for radiographic COVID-19 detection selects shortcuts over signal"
- [PASS] Ref 30 - DOI 10.1038/s42256-020-00257-z - "Shortcut learning in deep neural networks"

### 9. Em-dash scan
- [PASS] Total em-dash characters across `notebooks/01_eda.ipynb`, `notebooks/03_modeling.ipynb`, `reports/exploration_1.md`, `reports/modeling_1.md`, `reports/modeling_2.md`, `manuscripts/manuscript.md`, `manuscripts/references.md`, `deliverables/presentation.html` = **0**. (`brief.pdf` skipped - binary file, not applicable to text scan.)

### 10. AI-tell scan
- [PASS] Recursive grep for `verified by [0-9]+ agents`, `AI-verified`, `cross-checked by Claude` returned 0 hits.

### 11. Checkpoint schema
- [WARN] `checkpoint.json` is absent at the project root. Spec requires it with keys `project_number`, `title`, `methodology`, `status`. Cannot be validated. (`deliverables/metrics.json` exists but is a metrics dump, not a checkpoint.)

### Extra (project #1-#8 saved-model check)
- [PASS] `deliverables/covid_linprobe.joblib` present (4,959 bytes).
- [PASS] `deliverables/metrics.json` present (2,266 bytes) with full classification reports and confusion matrices for both val and test splits.
- [WARN] `notebooks/03_modeling.ipynb` references `deliverables/covid_resnet18.pt` (4-class fine-tune output); this `.pt` file is not on disk - it appears the 4-class notebook was authored but not run end-to-end, or its output was not retained. Not a FAIL since the binary linear-probe artefact (which is what the manuscript reports on) is present.

---

## Blockers
None.

## Status
Role A (VALIDATOR) complete. Output written to `/root/AI/liora_projects/08_covid_xrays/validation_report.md`.
