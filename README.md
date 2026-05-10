![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Medical CV](https://img.shields.io/badge/Medical-Imaging-red) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

# COVID-19 Chest X-ray Classification

Binary and multi-class classification of chest X-rays (COVID-19 / Pneumonia / Normal) using transfer learning on ResNet and EfficientNet.

---

## Task

**Medical Image Classification**

---

## Architecture

```
Chest X-ray → Preprocessing + Augmentation → EfficientNet-B4 → 3-class Head → GradCAM Heatmap
```

---

## Key Features

- 3-class classification: COVID-19 / Viral Pneumonia / Normal
- EfficientNet-B4 transfer learning (ImageNet pretrained)
- GradCAM saliency maps for radiologist-interpretable explanations
- Class imbalance handling with weighted sampler and focal loss
- AUC-ROC, sensitivity, specificity per class

---

## Dataset

[COVID-19 Radiography Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

---

## Project Structure

```
├── src/
│   ├── model_baseline.py      # Baseline model
│   └── model_advanced.py      # Advanced model
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory analysis
├── manuscripts/
│   └── manuscript.md          # IMRaD writeup
├── reports/
│   └── references.md          # Verified references
├── deliverables/
│   └── presentation.html      # Self-contained HTML
├── data/
│   └── README.md              # Dataset download instructions
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Sandyyy123/covid-xray-classifier.git
cd covid-xray-classifier
pip install -r requirements.txt

# See data/README.md for dataset download
python src/model_baseline.py
jupyter notebook notebooks/03_modeling.ipynb  # advanced transfer learning
```

---

## Tech Stack

`PyTorch · torchvision · EfficientNet · GradCAM · scikit-learn`

---

## Author

**Dr. Sandeep Grover** — PhD Data Science, independent ML researcher, Mössingen, Germany.

---

## License

MIT
