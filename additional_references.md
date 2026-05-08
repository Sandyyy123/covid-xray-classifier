# Additional References — Project 08 (COVID-19 Chest X-rays)

**Role:** Literature Scout
**Author:** Sandeep Grover
**Date:** 2026-05-08
**Scope:** 30 NEW papers (2024-2026), independently sourced from CrossRef and Europe PMC live APIs. Each entry was DOI-verified at write time. The existing `reports/references.md` was peeked at only after the search, solely to compile the SOTA-gap callout below; no entries were copied or modified.

---

## State-of-the-art callout: gaps in `reports/references.md` worth closing

The current reference list ends in 2022 and contains the foundational shortcut-learning critique (DeGrave 2021, Roberts 2021, Wynants 2020, Maguolo 2021, Geirhos 2020, Zech 2018) plus generic transfer-learning backbones. Five 2024-2026 strands are absent and should be cited if the manuscript is taken to a journal:

1. **A 2024 mechanistic confirmation that shortcut learning still degrades clinical-AI generalisation** — Ong Ly et al. 2024, *npj Digital Medicine*, with a method to estimate model generalisation without external data. This is the natural successor to DeGrave 2021 [ref 26 in `references.md`] and supplies a quantitative tool the current manuscript does not cite.
2. **A 2024 multi-centre re-benchmarking of COVID-19 CXR detection across centres** — Harkness et al. 2024, *Frontiers in Radiology*. This is the most direct external-validity reproduction available in 2024 of the COVID-CXR claim and should anchor section 5 of the manuscript.
3. **A 2024 NPJ-DM scoping review of the Clever-Hans effect across medical imaging** — Vasquez-Venegas et al. 2024, *Journal of Imaging Informatics in Medicine*. Subsumes Maguolo 2021 [ref 29] and updates Roberts 2021 [ref 27].
4. **2025 chest-X-ray foundation-model literature**, which has overtaken MoCo-CXR [ref 22] as SOTA: Ma et al. 2025 *Nat Biomed Eng* (vision-language pretrained transformer for respiratory disease), Paschali et al. 2025 *Radiology* (foundation models in radiology — what / how / why / why not), de Almeida et al. 2025 *Insights into Imaging* (AI4HI position paper), Yuan et al. 2025 *Health Care Science* (head-to-head supervised vs SSL pretraining for CXR). These four references should replace or augment refs 19-22 when discussing remediation 3.
5. **2024-2026 fairness and bias-mitigation literature for chest X-ray AI**, missing entirely from the current list: Mottez et al. 2026 *Pac Symp Biocomput* (bias detection-to-mitigation pipeline for chest X-ray), Rehman et al. 2026 *PLOS Digit Health* (diffusion-synthesised chest X-rays improving fairness), Gao et al. 2026 *Med Image Anal* (FairREAD), and Bassi et al. 2024 *Nat Commun* (LRP-based background-bias suppression). The manuscript currently treats source confounding only as an external-validity issue; these references would let it engage with the fairness-AI literature.

---

## Architectures and self-supervised pretraining for chest X-rays (2024-2026)

Anwar SM, Parida A, Atito S, Awais M, Nino G, Kittler J, et al. SS-CXR: Self-Supervised Pretraining Using Chest X-Rays Towards A Domain Specific Foundation Model. 2024 IEEE International Conference on Image Processing (ICIP). 2024. DOI:10.1109/icip51287.2024.10647378

Burshtein E, Cahan N, Ayzenberg L, Greenspan H. CXR-DINO: paving the way for a medical vision foundation model through self-supervised learning in chest x-ray analysis. Medical Imaging 2025: Image Processing. 2025. DOI:10.1117/12.3045886

Sheng H, Ma L, Samson J, Liu D. BarlowTwins-CXR: enhancing chest X-ray abnormality localization in heterogeneous data with cross-domain self-supervised learning. BMC Medical Informatics and Decision Making. 2024. DOI:10.1186/s12911-024-02529-9

Yuan H, Zhu M, Yang R, Liu H, Li I, Hong C. Rethinking Domain-Specific Pretraining by Supervised or Self-Supervised Learning for Chest Radiograph Classification: A Comparative Study Against ImageNet Counterparts in Cold-Start Active Learning. Health Care Science. 2025. DOI:10.1002/hcs2.70009

Huang Y, Sharma P, Palepu A, Greenbaum N, Beam A, Beam K. NeoCLIP: a self-supervised foundation model for the interpretation of neonatal radiographs. npj Digital Medicine. 2025. DOI:10.1038/s41746-025-01922-6

Ma L, Liang H, He Y, Wang W, Yan Z, Li W, et al. A vision-language pretrained transformer for versatile clinical respiratory disease applications. Nature Biomedical Engineering. 2025. DOI:10.1038/s41551-025-01544-z

Paschali M, Chen Z, Blankemeier L, Varma M, Youssef A, Bluethgen C, et al. Foundation Models in Radiology: What, How, Why, and Why Not. Radiology. 2025. DOI:10.1148/radiol.240597

de Almeida JG, Alberich LC, Tsakou G, Marias K, Tsiknakis M, Lekadir K, et al. Foundation models for radiology — the position of the AI for Health Imaging (AI4HI) network. Insights into Imaging. 2025. DOI:10.1186/s13244-025-02056-9

Ahmad IS, Suleiman RB, Yu T, Lin R, Zhao C, Liao J, et al. Foundation models for X-ray interpretation: a narrative review of current techniques and future perspectives in diagnostic imaging. Quantitative Imaging in Medicine and Surgery. 2026. DOI:10.21037/qims-2025-1-2782

---

## Shortcut learning, Clever-Hans effect, and generalisation failure (2024-2026)

Ong Ly C, Unnikrishnan B, Tadic T, Patel T, Duhamel J, Kandel S, et al. Shortcut learning in medical AI hinders generalization: method for estimating AI model generalization without external data. npj Digital Medicine. 2024. DOI:10.1038/s41746-024-01118-4

Vasquez-Venegas C, Wu C, Sundar S, Proa R, Beloy FJ, Medina JR, et al. Detecting and Mitigating the Clever Hans Effect in Medical Imaging: A Scoping Review. Journal of Imaging Informatics in Medicine. 2024. DOI:10.1007/s10278-024-01335-z

Pathak AK, Gupta M, Jain G. Unmasking the Clever Hans effect in AI models: shortcut learning, spurious correlations, and the path toward robust intelligence. Frontiers in Artificial Intelligence. 2026. DOI:10.3389/frai.2025.1692454

Sourget T, Hestbek-Moller M, Jimenez-Sanchez A, Junchi Xu J, Cheplygina V. Mask of Truth: Model Sensitivity to Unexpected Regions of Medical Images. Journal of Imaging Informatics in Medicine. 2025. DOI:10.1007/s10278-025-01531-5

Bassi PRAS, Dertkigil SSJ, Cavalli A. Improving deep neural network generalization and robustness to background bias via layer-wise relevance propagation optimization. Nature Communications. 2024. DOI:10.1038/s41467-023-44371-z

---

## Domain shift, domain adaptation, and external validation in chest X-ray AI (2024-2026)

Harkness R, Frangi AF, Zucker K, Ravikumar N. Multi-centre benchmarking of deep learning models for COVID-19 detection in chest x-rays. Frontiers in Radiology. 2024. DOI:10.3389/fradi.2024.1386906

Musa A, Prasad R, Hernandez M. Addressing cross-population domain shift in chest X-ray classification through supervised adversarial domain adaptation. Scientific Reports. 2025. DOI:10.1038/s41598-025-95390-3

Singthongchai J, Wangkhamhan T. Adaptive Normalization Enhances the Generalization of Deep Learning Model in Chest X-Ray Classification. Journal of Imaging. 2025. DOI:10.3390/jimaging12010014

Suleman MU, Mursaleen M, Khalil U, Saboor A, Bilal M, Khan SA, et al. Assessing the generalizability of artificial intelligence in radiology: a systematic review of performance across different clinical settings. Annals of Medicine and Surgery. 2025. DOI:10.1097/ms9.0000000000004166

Khan FK, Tahir WB, Lee MS, Kim JY, Byon SS, Pi S, et al. Leveraging Large-Scale Public Data for Artificial Intelligence-Driven Chest X-Ray Analysis and Diagnosis. Diagnostics. 2026. DOI:10.3390/diagnostics16010146

---

## Fairness, bias mitigation, and synthetic-data debiasing (2024-2026)

Mottez C, Fay L, Varma M, Ostmeier S, Langlotz C. From Detection to Mitigation: Addressing Bias in Deep Learning Models for Chest X-Ray Diagnosis. Pacific Symposium on Biocomputing. 2026. DOI:10.1142/9789819824755_0039

Rehman D, Chen H, Lee C, Vilor-Tejedor N, Upadhyay S, Kuo P. Diffusion-synthesized Chest X-rays improve fairness and diagnostic performance. PLOS Digital Health. 2026. DOI:10.1371/journal.pdig.0001277

Gao Y, Hao J, Zhou B. FairREAD: Re-fusing demographic attributes after disentanglement for fair medical image classification. Medical Image Analysis. 2026. DOI:10.1016/j.media.2025.103858

Ranard BL, Park S, Jia Y, Zhang Y, Alwan F, Celi LA, et al. Minimizing bias when using artificial intelligence in critical care medicine. Journal of Critical Care. 2024. DOI:10.1016/j.jcrc.2024.154796

Jiang J, Domingues L, Mendes JM. Synthetic data in medical imaging within the EHDS: a path forward for ethics, regulation, and standards. Frontiers in Digital Health. 2025. DOI:10.3389/fdgth.2025.1620270

---

## Explainable AI and interpretable architectures for chest X-ray (2024-2026)

Sannasi Chakravarthy SR, Bharanidharan N, Vinothini C, Vinoth Kumar V, Mahesh TR, Guluwadi S. Adaptive Mish activation and ranger optimizer-based SEA-ResNet50 model with explainable AI for multiclass classification of COVID-19 chest X-ray images. BMC Medical Imaging. 2024. DOI:10.1186/s12880-024-01394-2

Wienholt P, Kuhl C, Kather JN, Nebelung S, Truhn D. MedicalPatchNet: a patch-based self-explainable AI architecture for chest X-ray classification. Scientific Reports. 2026. DOI:10.1038/s41598-026-40358-0

Chiang L. An Interpretable Chest X-ray Classification Framework Using Prototype Memory and Counterfactual Consistency. Cureus. 2026. DOI:10.7759/cureus.103134

---

## Surveys, scoping reviews, and field-status assessments (2024-2026)

Maheswari S, Suresh S, Ahamed Ali S. A systematic literature review on machine learning and deep learning-based covid-19 detection frameworks using X-ray Images. Applied Soft Computing. 2024. DOI:10.1016/j.asoc.2024.112137

Hasanah U, Leu J, Avian C, Azmi I, Prakosa SW. A systematic review of multilabel chest X-ray classification using deep learning. Multimedia Tools and Applications. 2024. DOI:10.1007/s11042-024-20172-4

Vasilev Y, Bazhin A, Reshetnikov R, Nanova O, Vladzymyrskyy A, Arzamasov K, et al. Autonomous chest x-ray image classification, capabilities and prospects: rapid evidence assessment. Frontiers in Digital Health. 2026. DOI:10.3389/fdgth.2025.1685771

Terespolsky S, Yassi A, Ehrlich R, Bruton J, Lockhart K, Wang H, et al. The Case for Local AI Development: Lessons From Computer-Aided Detection of Tuberculosis and Silicosis in Southern Africa's Ex-Miners. Annals of Global Health. 2026. DOI:10.5334/aogh.5064

---

## Verification note

All 30 entries above were resolved against `https://api.crossref.org/works/{doi}` at write time on 2026-05-08 with HTTP 200 responses, and metadata (authors, title, journal, year) were extracted from the same CrossRef payload. No entry was retained without a successful resolve. Volume, issue, and page numbers are intentionally omitted per project convention.
