# Exploration Report 1: COVID-19 Chest X-ray Dataset


**Date:** 2026-05-01
**Scope:** Initial data exploration on a sampled view of the archive (first 200 images per class). The full archive on disk is approximately 29 GB so it is never extracted; image bytes are streamed directly from the zip via `zipfile`.

## 1. Source

- Kaggle dataset: `tawsifurrahman/covid19-radiography-database` (COVID-19 Radiography Database, IEEE-curated by Qatar University, University of Dhaka, and collaborators).
- Headline references in the brief:
  - Chowdhury et al. (2020), arXiv 2003.13865.
  - Rahman et al. (2021), Computers in Biology and Medicine, doi 10.1016/j.compbiomed.2021.105002.
- Local archive: `/root/AI/project_root/data/covid_xrays_archive.zip` (~29 GB on disk; the brief lists 1.15 GB which corresponds to the v3 release, so this archive is the expanded v5 release with 4 classes plus segmentation masks).

## 2. Archive structure (expected, auto-detected at runtime)

The Kaggle release ships a single root folder (`COVID-19_Radiography_Dataset/`) containing one subfolder per class. Each class folder holds an `images/` directory of PNGs and a `masks/` directory of lung-field segmentation masks. The notebook detects classes by scanning for any path ending in an image extension that lives under an `images/` folder, so it adapts to either the v3 or v5 layout.

```
COVID-19_Radiography_Dataset/
  COVID/
    images/   PNG x-rays
    masks/    PNG lung masks
  Normal/
    images/
    masks/
  Lung_Opacity/        (v5 only)
    images/
    masks/
  Viral Pneumonia/
    images/
    masks/
  COVID.metadata.xlsx
  Normal.metadata.xlsx
  Lung_Opacity.metadata.xlsx
  Viral Pneumonia.metadata.xlsx
  README.md.txt
```

Masks are excluded from this EDA (the notebook filters any path containing `/masks/`, `mask/`, or `_mask`).

## 3. Classes

Four classes in the v5 release. The notebook prints exact counts at runtime; the published v5 counts (Rahman et al. 2021) are:

| Class | Images (published v5) | Notes |
|---|---|---|
| COVID | 3,616 | RT-PCR confirmed COVID-19 |
| Normal | 10,192 | No radiographic abnormality |
| Lung_Opacity | 6,012 | Non-COVID lung opacity (added in v5) |
| Viral Pneumonia | 1,345 | Non-COVID viral pneumonia |
| **Total** | **21,165** | |

The v3 release listed 3,616 COVID, 10,192 Normal, and 1,345 Viral Pneumonia for ~15 k images (1.15 GB), matching the figure in the brief. The actual class breakdown observed in this archive is reported by the notebook in section 2 ("Class distribution").

## 4. Image format and dimensions

- File type: PNG, single channel (grayscale) when read as L; some files are stored as RGB with identical channels.
- Native dimensions: 299 x 299 pixels for the original Qatar/Dhaka release. Items added in v5 are also distributed at 299 x 299. The notebook prints the observed min/median/max width and height per class on the 200-image sample.
- Bit depth: 8-bit per channel.

## 5. Sample-based statistics

The notebook computes on the first 200 images per class (so the entire EDA runs in a few minutes without expanding the archive):

- Width/height min, median, max per class.
- Modal channel count per class.
- Per-class pixel-intensity mean and standard deviation (computed on the grayscale conversion).
- A 5-image grid per class for visual inspection.

A small per-class shift in mean pixel intensity is expected (COVID and Viral Pneumonia images often appear slightly darker because of denser lung-field opacities). This is useful to note before training: any classifier could pick up acquisition-side artifacts rather than disease features, so preprocessing must include per-image normalisation and ideally lung-field masking using the supplied masks.

## 6. Key observations

1. **Class imbalance.** Normal is roughly 7x larger than Viral Pneumonia. Any modelling approach will need class weights, oversampling, or focal loss.
2. **Lung_Opacity confounder.** The v5 release adds a "non-COVID lung opacity" class. Distinguishing COVID from Lung_Opacity is the hardest pairwise task in the literature and is the realistic clinical question.
3. **Mask availability.** Lung-field masks are shipped for every image. Using them to crop or attention-weight the lung region is a strong defence against the dataset-bias issue flagged in DeGrave et al. (2021, Nat. Mach. Intell.).
4. **Source heterogeneity.** Images come from multiple public sources (BIMCV, RSNA, Italian SIRM, Eurorad, Cohen GitHub). Hospital-of-origin leakage is the main risk factor and must be checked before any train/val/test split.
5. **Uniform 299 x 299 size.** Resizing to 224 x 224 (for ImageNet-pretrained backbones) or keeping 299 x 299 (for InceptionV3-style backbones) are both safe.

## 7. Next steps

- `02_preprocessing.ipynb`: deterministic resize, per-image normalisation, optional lung-field masking, stratified train/val/test split with source held out where metadata allows.
- `03_baseline.ipynb`: simple CNN baseline plus a transfer-learning model (ResNet50 or EfficientNet-B0) with class weights.
- `04_modelling.ipynb`: ablations on input size, masking on/off, and balanced sampling.
- `05_evaluation.ipynb`: per-class precision/recall, confusion matrix, Grad-CAM sanity check on lung-field localisation.
