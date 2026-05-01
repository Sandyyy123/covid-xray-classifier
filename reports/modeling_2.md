# Modeling 2 - Source-Bias-Aware Modeling (COVID X-rays, planned)

## Why the modeling_1 result is the wrong baseline to improve on

`modeling_1.md` reports val accuracy 0.838 collapsing to test accuracy 0.550. This is **not** a hyperparameter problem and tuning the linear probe will not fix it. The problem is **source-label confounding** in the COVIDx splits: positive-class images come from RICORD / BIMCV / ACTMED / Cohen sources whose distribution differs systematically from the negative-class sources, and the train and test splits draw from different source populations.

DeGrave et al. (2021, *Nature Machine Intelligence*) showed that ImageNet-pretrained CNNs trained on COVIDx-style splits learn the source watermark, not pathology. The test-split distribution shift exposes that.

## Recommended Modeling 2 (compute reserved for next session)

Three changes are required, in order:

1. **Source stratification at train time**: draw train and test from a disjoint source set per class but with matched marginal priors per source. The COVIDx-CXR4 documentation provides such curated splits; use those rather than the bulk `train.txt` provided in the archive.
2. **Source-confounding penalty**: train with a domain-adversarial loss (Ganin and Lempitsky 2015 DANN-style) that penalises a source-classification head while training the COVID head, forcing the backbone to throw away source-correlated features.
3. **Foundation-model evaluation**: replace ResNet18-ImageNet with a chest-X-ray-pretrained backbone such as MoCo-CXR (Sriram et al. 2021, ResNet50 self-supervised on 250k unlabelled CXRs) or MAE-CXR. These backbones have shown 5-10 ROC-AUC point gains on transfer-shifted benchmarks because the pretraining domain matches the downstream task.

## Configuration sketch

- Backbone: MoCo-CXR ResNet50 (Sriram et al. 2021 weights), unfrozen final block.
- Head: 2-layer MLP with dropout, binary cross-entropy.
- Optimiser: AdamW, lr 1e-4 head / 1e-5 backbone, cosine schedule, 15 epochs.
- Domain-adversarial branch: gradient-reversal layer plus 4-class source classifier, loss weight 0.1.
- Data augmentation: RandomResizedCrop(224, scale=0.8-1.0), RandomHorizontalFlip(0.5), RandomRotation(10).
- Class balance: equal positive / negative sampling per minibatch.

## Expected outcome

Based on the DeGrave + Sriram papers, this configuration is expected to yield **test ROC-AUC in the 0.75-0.85 range** with a smaller val-test gap (within 5 points) versus the 0.578 collapse seen in `modeling_1`. That would be a defensible model rather than a source-spurious one.

## Why it is not run in this session

The MoCo-CXR weights are 250 MB; download + 15-epoch fine-tuning on 1,200 train images is approximately 30-40 minutes on CPU. This run is reserved for the next session.

## Persisted artifacts (modeling_1 baseline)

- `deliverables/covid_linprobe.joblib` - the source-spurious linear probe baseline (kept as a reference, not a deployment model)
- `deliverables/metrics.json` - per-split metrics
