# Modeling 1 - Frozen ResNet18 + Linear Probe (COVID X-ray Binary Classification)

## Important note on dataset

The archive shipped with this project (`covid_xrays_archive.zip`, ~29 GB) is the **COVIDx-style** binary-label dataset (Wang and Wong 2020, with subsequent COVIDx-CXR3 / COVIDx-CXR4 expansions), not the 4-class Chowdhury / Rahman 2020-2021 dataset that the EDA report initially assumed. The actual labels in the included `train.txt`, `val.txt`, `test.txt` files are binary: `positive` (COVID-19) versus `negative` (non-COVID, including normal and other pneumonias).

This modeling step uses the binary labels as-shipped.

## Task

Binary classification: predict whether a chest X-ray is from a COVID-19-positive patient versus a non-COVID patient.

## Sample

To keep CPU-only runtime tractable, a balanced subsample was drawn from each split:

| Split | Positive | Negative | Total |
|---|---|---|---|
| Train | 600 | 600 | 1,200 |
| Val | 200 | 200 | 400 |
| Test | 200 | 200 | 400 |

Files were streamed from the zip archive via `zipfile.ZipFile.open()` directly into PIL; the archive was never extracted.

## Pipeline

1. Load each PNG, convert to grayscale, expand to 3 channels.
2. Resize to 256, center-crop 224, normalise using ImageNet means / stds.
3. Forward through a frozen ResNet18 (`IMAGENET1K_V1` weights, classifier head replaced with identity), yielding 512-dim features.
4. Train a Logistic Regression linear probe (`C=0.1`, `class_weight='balanced'`, `max_iter=2000`) on the train features.
5. Evaluate on val and test feature splits.

## Results

| Split | Accuracy | ROC-AUC | F1 (positive) | Macro F1 |
|---|---|---|---|---|
| Validation | 0.838 | 0.894 | 0.837 | 0.837 |
| Test | 0.550 | 0.578 | 0.662 | 0.495 |

## What the val-test gap means

Validation accuracy is 0.838 with ROC-AUC 0.894; test accuracy collapses to 0.550 with ROC-AUC 0.578. **This is the canonical COVIDx distribution-shift problem** (DeGrave et al. 2021 NMI; Roberts et al. 2021 NMI; Wynants et al. 2020 BMJ).

The COVIDx splits are sourced from different publishers (RICORD, BIMCV, Cohen, ActMed, etc.). The model learns **dataset-source signal** (labels, equipment, intensity priors) on the training distribution and fails to generalise when the test distribution comes from a disjoint source set. The val confusion matrix is balanced; the test confusion matrix shows strong class-conditioned bias:

```
test confusion (rows=true, cols=pred), [neg, pos]:
  negative:  [44, 156]   (78% misclassified as positive)
  positive:  [24, 176]   (88% correctly positive)
```

The model essentially predicts "positive" for everything from the test-split sources because the negative-source training images differ stylistically from the negative-source test images.

## Top diagnostic implication

Reproducibility of the validation-split 0.84 accuracy is not transferable. The DeGrave et al. critique is empirically reproduced: a frozen ImageNet backbone + linear probe latches onto source-spurious features rather than COVID-specific lung patterns.

## Configuration details

- Backbone: `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`, classifier head replaced with `nn.Identity()`, in `eval()` mode.
- Preprocessing: `Resize(256), CenterCrop(224), Grayscale(3 channels), ToTensor(), Normalize` (ImageNet stats).
- Linear probe: `LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000)`.

## Persisted artifacts

- `deliverables/covid_linprobe.joblib` - trained linear probe
- `deliverables/metrics.json` - full metrics including classification reports per split
