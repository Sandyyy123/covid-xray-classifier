"""
model_baseline.py - PLACEHOLDER

This file is intentionally a stub. The binary linear-probe pipeline described
in `manuscripts/manuscript.md` Section 3 (frozen ImageNet ResNet18 features
plus a sklearn LogisticRegression head) was executed end-to-end and produced
the artefacts shipped in `deliverables/`:

    - deliverables/covid_linprobe.joblib   # fitted LogisticRegression
    - deliverables/metrics.json            # val + test classification report
                                           # and confusion matrices

The Python driver script that produced those artefacts was not committed to
this repository before the deliverable bundle was assembled. The fitted model
and the metrics file are the canonical record of that experiment.

The committed notebook `notebooks/03_modeling.ipynb` is a SEPARATE experiment
(a 4-class ResNet18 fine-tune on a source-derived label scheme). It is NOT
the source of `covid_linprobe.joblib` or the binary `metrics.json` shipped
here. Running that notebook will overwrite `metrics.json` with a different
schema; do not run it without taking a backup first if you need to preserve
the binary linprobe metrics.

Recommended remediation, when reviving this project:

  1. Recreate the binary linprobe driver as a real script in this file,
     mirroring the Methods section: torchvision resnet18 IMAGENET1K_V1,
     `nn.Identity()` head, ImageNet normalisation, sklearn LogisticRegression
     with C=0.1, class_weight='balanced', max_iter=2000.
  2. Re-fit on the same 1,200-image train subsample and confirm the val
     accuracy 0.8375 / val ROC-AUC 0.8944 / test accuracy 0.55 /
     test ROC-AUC 0.578 numbers reproduce.
  3. Replace this stub with the working script.

No fabricated source is included here. The honest record is: the fitted
joblib and metrics.json exist on disk and match the manuscript;
the driver script does not.
"""

raise NotImplementedError(
    "model_baseline.py is a placeholder. See deliverables/covid_linprobe.joblib "
    "for the fitted model and deliverables/metrics.json for the reported metrics. "
    "The original driver script was not preserved in this repository."
)
