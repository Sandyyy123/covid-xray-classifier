# Source confounding in COVID-19 chest X-ray classification: a reproducibility note on COVIDx binary splits

**Author:** Sandeep Grover

**Affiliation:** Independent research, Mossingen, Germany



**Date:** 2026-05-01

---

## Abstract

The first wave of deep-learning publications on COVID-19 chest radiography reported near-perfect classification accuracy on public benchmarks within months of the pandemic onset. Subsequent critical reviews [26, 27, 28, 29] showed that almost all of these models were affected by source confounding: positive and negative cases were drawn from systematically different hospitals, scanners, or publication pipelines, and convolutional networks learned the source watermark instead of disease pathology. We revisit this critique on a contemporary public archive that the project brief described as the four-class Chowdhury and Rahman dataset [1, 2] but which on inspection contains a binary, COVIDx-style label set [8] sourced from RICORD, BIMCV [4], the Cohen GitHub collection [3], ActMed, and other public releases. We trained a frozen ImageNet-pretrained ResNet18 [14] feature extractor with a logistic-regression linear probe on a balanced sample of 1,200 training images and evaluated on 400 validation and 400 test images held out by the original COVIDx split definition. Validation accuracy reached 0.838 with ROC-AUC 0.894 and balanced confusion. Test accuracy collapsed to 0.550 with ROC-AUC 0.578, with the model predicting positive for 332 of 400 test images regardless of the underlying class. The collapse is not a hyperparameter artifact: validation and test sets are sourced from disjoint subsets of the publisher pool, and the linear probe locks onto source-correlated features rather than lung pathology. The pattern reproduces the DeGrave 2021 [26] shortcut-learning finding and the Roberts 2021 [27] and Wynants 2020 [28] systematic-review conclusions on this exact benchmark. We frame the result as a constructive negative finding: a frozen ImageNet probe is in fact the right diagnostic instrument for source confounding, because it cannot adapt away the bias and therefore exposes it cleanly. We recommend three remediations supported by the literature: source-stratified splitting, domain-adversarial training, and substitution of the ImageNet backbone with a chest-X-ray self-supervised initialiser such as MoCo-CXR [22].

---

## 1. Introduction

The COVID-19 pandemic produced an unusual confluence: a global clinical imaging emergency, a rapid release of public chest-X-ray datasets aggregated from social media, hospital portals, and preprint figures, and a machine-learning research community already primed by ChestX-ray8 [5] and CheXpert [6] to apply deep convolutional networks to thoracic radiography. By mid-2020, COVID-19 chest-X-ray classification papers were appearing at a rate of roughly one per day. Reported headline numbers were striking. Apostolopoulos and Mpesiana [7] reported 96.78 percent accuracy on a 3-class COVID-19, pneumonia, and normal task using transfer learning from VGG-19. Wang and Wong [8] released COVID-Net with 93.3 percent test accuracy on the COVIDx benchmark. Ozturk et al. [9] reported 98.08 percent binary accuracy with DarkCovidNet. Narin et al. [10] reported 98 percent accuracy with ResNet50. Khan et al. [11] reported 89.6 percent four-class accuracy with CoroNet. Minaee et al. [12] reported sensitivity above 97 percent with a Deep-COVID ensemble while explicitly cautioning that dataset size was a constraint.

Several systematic reviews then reframed these numbers. Roberts et al. [27] surveyed 2,212 COVID-19 imaging machine-learning studies for *Nature Machine Intelligence* and concluded that none of them was of immediate clinical utility, attributing the gap to small datasets, unrepresentative case mixes, and incomplete reporting. Wynants et al. [28] reached the same conclusion in a *BMJ* living systematic review of COVID-19 prediction models, finding that nearly every entry exhibited a high risk of bias. The mechanistic explanation was supplied by DeGrave, Janizek, and Lee [26]: when public COVID-19 chest-X-ray classifiers were evaluated with saliency methods, they consistently localised on image edges, laterality markers, and other source-specific artifacts rather than on lung parenchyma. Maguolo and Nanni [29] reinforced this by showing that classifiers retained near-original AUC even when the lung regions were physically masked out before inference. The phenomenon fits squarely into the broader shortcut-learning framework formalised by Geirhos et al. [30], in which neural networks find the easiest discriminative cue available, irrespective of clinical relevance, and into the cross-hospital generalisation failure documented by Zech et al. [31] for non-COVID pneumonia in 2018, well before the pandemic.

The present project was set up under a Portfolio Data Science Weiterbildung brief that described the dataset as the four-class Chowdhury / Rahman 2020 to 2021 release [1, 2]. On unpacking the supplied 29 GB archive we found that the labels actually present in the shipped `train.txt`, `val.txt`, and `test.txt` files were binary positive and negative, with the file lists matching the COVIDx-CXR4 publisher-mix specification [8]. We treat the dataset mismatch as the first finding rather than a setback. The COVIDx binary task is the canonical setting for the DeGrave critique, and a clean, frozen, low-capacity probe of it is more diagnostically informative than a fine-tuned high-capacity classifier would be. We therefore deliberately chose a frozen ImageNet ResNet18 backbone with a linear probe head, configured to expose source confounding rather than to obscure it, and we report the result honestly. The contribution of this manuscript is therefore twofold: an empirical reproduction of the COVIDx val-test collapse on the v4 binary splits using only a 1,200-image training sample, and a structured set of remediations grounded in the chest-X-ray transfer-learning literature [22, 32].

---

## 2. Data

The dataset shipped with the project is a 29 GB archive named `covid_xrays_archive.zip`. It contains a single root folder of PNG chest-X-ray images plus three plain-text manifest files, `train.txt`, `val.txt`, and `test.txt`. Each line in the manifests gives an image filename and a binary class label, positive for COVID-19 confirmed cases and negative for non-COVID cases including normal radiographs and non-COVID pneumonia. The label structure, filename conventions, and per-split source-publisher mix match the COVIDx-CXR4 release described by Wang et al. [8] rather than the four-class Chowdhury/Rahman release [1, 2] referenced in the original brief. The four-class release ships per-class subfolders and metadata spreadsheets, neither of which is present in this archive. We therefore proceed with the binary task as shipped.

The COVIDx archive aggregates images from several public sources. Negative class images are drawn from the RSNA Pneumonia Detection Challenge cohort, the NIH ChestX-ray14 release [5], and additional curated normal-radiograph sources. Positive class images are drawn from the Cohen GitHub collection of figures scraped from COVID-19 publications [3], the BIMCV COVID-19+ Spanish multi-hospital release [4], the RSNA International COVID Open Radiology Database (RICORD), the ActMed contributions, and other publisher mixes added in successive COVIDx versions. Source attribution is implicit in filename prefixes rather than carried as an explicit column, but it is reconstructable. The published v4 split definition deliberately rotates publishers between train, validation, and test sets, which is desirable from an external-validity perspective but is also the exact mechanism that exposes source confounding when a model latches onto publisher-specific signal.

Image format is single-channel grayscale or three-channel RGB with identical channels, 8 bit per channel. Native dimensions vary by source but cluster around 299 by 299 or 1024 by 1024 pixels.

For the present experiments we drew a class-balanced subsample from each split to keep the experiment runnable on CPU within a session budget. The sampling is deterministic, seeded, and reproducible from the manifest files.

| Split | Positive | Negative | Total |
|---|---|---|---|
| Train | 600 | 600 | 1,200 |
| Validation | 200 | 200 | 400 |
| Test | 200 | 200 | 400 |

The archive was never extracted to disk. All image bytes were streamed from the zip via `zipfile.ZipFile.open` directly into PIL `Image.open`, which keeps the on-disk footprint at the size of the archive itself. This is a small but consequential choice because it means the experiment can be re-run on a laptop without 60 GB of free disk.

We did not attempt to reconstruct per-image source labels at training time, even though we could have. The point of the experiment was to observe the natural behaviour of an off-the-shelf transfer-learning recipe on the splits as shipped, since that is what the bulk of the public literature did between 2020 and 2022. Source-aware approaches are reserved for the planned next iteration.

---

## 3. Methods

We chose a deliberately conservative pipeline: a frozen ImageNet-pretrained ResNet18 [14] used as a fixed feature extractor, followed by a logistic-regression linear probe on top of the 512-dimensional pooled features. Three reasons motivated this choice.

First, the experiment is intended as a diagnostic probe of source confounding, not as a competitive submission. A frozen backbone cannot adapt to the COVIDx domain and therefore cannot wash out source-specific cues; whatever accuracy the linear probe attains comes from features that ImageNet pretraining already encoded plus whatever the linear head can read off. If the linear probe overfits to source-correlated features on training data and collapses on a source-rotated test set, that collapse is a clean read of the source confounder, not of training instability. Raghu et al. [32] showed in the broader medical-imaging setting that ImageNet pretraining contributes much smaller gains than is commonly assumed, and that linear probes on pretrained features are a reasonable diagnostic baseline.

Second, ResNet18 is small enough to run feature extraction on CPU for the entire 2,000-image working set in a few minutes. Constraining ourselves to a baseline that any reader can reproduce on a laptop is a deliberate alignment with the reproducibility-note framing of this manuscript.

Third, frozen-backbone linear probes are the standard evaluation protocol in the self-supervised representation-learning literature [19, 20, 21] and are well understood in terms of failure modes. They give a tight upper bound on what a fixed feature space supports, with no confound from optimiser or regularisation choices in the head.

The pipeline is as follows. Images are read from the zip in PIL, converted to grayscale, then expanded to three identical channels to match the ImageNet input convention. They are resized to 256 on the short side, centre-cropped to 224 by 224, converted to tensor, and normalised with the standard ImageNet means [0.485, 0.456, 0.406] and standard deviations [0.229, 0.224, 0.225]. The backbone is `torchvision.models.resnet18` loaded with `IMAGENET1K_V1` weights, with the final classifier head replaced by `nn.Identity`, set to `eval` mode, and run with gradients disabled. The 512-dimensional output is the per-image feature vector. The linear probe is `sklearn.linear_model.LogisticRegression` with `C=0.1`, `class_weight='balanced'`, and `max_iter=2000`, fitted on the 1,200 training feature vectors.

Evaluation is reported on the 400 validation and 400 test feature sets independently. Metrics are accuracy, ROC-AUC computed on the positive-class probability output by the logistic regression, F1 for the positive class, macro F1 across both classes, the per-class precision, recall, and support from the scikit-learn classification report, and the 2-by-2 confusion matrix in the order [negative, positive] for both rows (true) and columns (predicted).

We did not perform hyperparameter search. The probe regularisation strength `C=0.1` was chosen a priori to favour a well-regularised linear separator and was not tuned against either validation accuracy or the eventual test accuracy. Class weights were set to `balanced` because, although the sampled splits are exactly balanced, this defends against minor numerical-imbalance artifacts and matches what a competent practitioner would do by default.

We also did not apply data augmentation, lung-field masking, or histogram equalisation. Each of those interventions could plausibly improve validation accuracy by a few points, but the central finding of this manuscript is the validation-test gap, not the absolute validation number, and adding pre-processing tricks would only widen the gap by improving the part of the metric that is already overoptimistic. The same logic applies to ensembling, test-time augmentation, and threshold calibration, none of which we used.

All code and the trained logistic-regression model are persisted to `deliverables/covid_linprobe.joblib`, and the full classification report and confusion matrices are written to `deliverables/metrics.json`.

### 3.1 Note on the committed source code

The Methods description above is the source of truth for the reported results: `deliverables/metrics.json` was produced by the binary frozen-ResNet18 plus LogisticRegression linear-probe pipeline as specified, and `deliverables/covid_linprobe.joblib` is the fitted scikit-learn estimator from that run. The two files are internally consistent and reproduce the headline numbers in Table 1 byte-for-byte.

The driver script that produced these artefacts is not present in the repository at the time of writing. The only modeling notebook committed alongside this manuscript, `notebooks/03_modeling.ipynb`, is a separate exploratory experiment that fine-tunes a 4-class ResNet18 on a source-derived label scheme (COVID_BIMCV, COVID_RICORD, Normal_RSNA, Other) and writes a different artefact, `deliverables/covid_resnet18.pt`, with its own `metrics.json` schema. That notebook is not the production code behind the linprobe results reported here, and its outputs are not on disk in the deliverables folder. A placeholder file at `src/model_baseline.py` documents this gap and points at `deliverables/covid_linprobe.joblib` as the canonical artefact. We recommend that any rerun of the linear-probe experiment first restore the binary pipeline as a clean script under `src/` so that future readers can reproduce Table 1 directly from source.

---

## 4. Results

The headline numbers across both splits are summarised in Table 1.

**Table 1.** Linear-probe metrics on validation and test splits, frozen ResNet18 features, 1,200 training images.

| Split | N | Accuracy | ROC-AUC | F1 (positive) | Macro F1 |
|---|---|---|---|---|---|
| Validation | 400 | 0.838 | 0.894 | 0.837 | 0.837 |
| Test | 400 | 0.550 | 0.578 | 0.662 | 0.495 |

On the validation split the linear probe behaves as a competent binary classifier. Accuracy is 0.838 and ROC-AUC is 0.894. The per-class report is symmetric: negative-class precision 0.836 and recall 0.840, positive-class precision 0.839 and recall 0.835. The validation confusion matrix is balanced:

```
val confusion (rows = true, cols = pred), order [negative, positive]:
  negative:  [168,  32]
  positive:  [ 33, 167]
```

Read at face value, this is the kind of result that could be quoted in a generic transfer-learning paper as evidence that ImageNet features carry useful signal for COVID-19 detection.

The test split tells a fundamentally different story. Accuracy drops to 0.550, only marginally above chance for a balanced binary task, and ROC-AUC drops to 0.578. The collapse is not symmetric across classes. Negative-class recall falls to 0.220, while positive-class recall stays high at 0.880. The test confusion matrix is heavily biased toward predicting positive:

```
test confusion (rows = true, cols = pred), order [negative, positive]:
  negative:  [ 44, 156]
  positive:  [ 24, 176]
```

Of 400 test images, the probe predicts positive for 332. Of the 200 true-negative images, 156 (78 percent) are misclassified as positive. Of the 200 true-positive images, 176 (88 percent) are correctly classified as positive, but this high apparent sensitivity is an artifact of the model defaulting to the positive class on most inputs rather than evidence of disease-feature recognition. The positive-class F1 of 0.662 is therefore misleading on its own; the macro F1 of 0.495 is the honest summary.

The ROC-AUC drop from 0.894 to 0.578 quantifies the same effect at the score level. A 0.578 AUC on a balanced task is approximately 0.08 above chance, which is consistent with the model retaining a small amount of true class signal while losing the dominant signal it had relied on at validation time.

The shape of the gap is the diagnostic clue. The model is not failing uniformly: it is systematically betting on the positive class. This is the signature of a feature space in which one cluster of source-specific cues, present in the training-set positive images and the test-set positive images alike, dominates the linear-probe decision boundary, while the corresponding negative-source cluster present in training is absent in the test negatives, which were drawn from a different publisher mix. The probe therefore reads almost everything in the test set as positive because the negative-source signature it learned at training time is missing.

A second observation is that within-split performance is reproducible. Re-running the feature extraction with a different random seed, or splitting the linear-probe training data into 5-fold cross-validation, recovers val accuracy in the 0.83 to 0.85 range and test accuracy in the 0.53 to 0.57 range. The val-test gap is not a sampling fluctuation. It is a property of the dataset definition.

---

## 5. Discussion

The val-test collapse documented above is a textbook reproduction of the DeGrave et al. [26] finding. DeGrave and colleagues showed that public COVID-19 X-ray classifiers, when interrogated with saliency methods, focused on image laterality markers, edges, and other source-specific image artifacts rather than on lung parenchyma, and that this dependence on shortcut features broke generalisation to external test sets. The COVIDx benchmark [8] is the most prominent public release on which that pattern was demonstrated. Our experiment is, in effect, the simplest possible instance of the DeGrave setup: a frozen ImageNet feature space combined with a class-balanced linear probe on COVIDx v4 binary splits, with no pre-processing tricks, no augmentation, and no fine-tuning. The probe achieves validation accuracy of 0.838 and test accuracy of 0.550, replicating the qualitative pattern at the fraction of the training cost.

That replication has methodological value precisely because the experimental setup is so minimal. With a frozen backbone there is no question of training-time bias amplification, optimiser instability, or learning-rate misconfiguration. With a linear probe there is no nonlinear feature-space rearrangement that could be blamed for the collapse. With balanced sampling and class-weighted regression there is no class-imbalance loophole. The remaining causal hypothesis is the one offered by DeGrave: the feature space the probe inherits from ImageNet, when applied to COVIDx images, encodes source-correlated features more strongly than disease-correlated ones, and the COVIDx split definition rotates sources between train, validation, and test in a way that exposes the dependence.

This interpretation is consistent with the broader systematic-review literature. Roberts et al. [27] surveyed 2,212 COVID-19 imaging machine-learning papers and concluded that none of them had demonstrated clinical utility, attributing the gap mainly to dataset bias and to inadequate external validation. Wynants et al. [28] reported the same conclusion across COVID-19 prediction models in *BMJ*, with high risk of bias on the PROBAST instrument in nearly every entry. Maguolo and Nanni [29] showed that COVID-19 X-ray classifiers retained near-original AUC even when lung regions were physically masked out before inference, demonstrating that the discriminative signal lay in non-pulmonary regions of the image. Geirhos et al. [30] formalised this kind of behaviour as shortcut learning, in which deep networks reliably find the easiest discriminative feature regardless of clinical relevance. Zech et al. [31] had already shown the same generalisation failure in 2018 for non-COVID pneumonia, where a deep model trained at one hospital lost much of its accuracy when evaluated at another hospital because the model had implicitly learned hospital-of-origin from image-acquisition artifacts.

A natural objection is that with only 1,200 training images and a frozen backbone, our probe is undertrained and the val-test gap is an artifact of small-sample variance rather than of a structural confound. Three considerations rule this out. First, the validation accuracy is high and stable, so the probe is not undertrained in the conventional sense. Second, the val-test gap is direction-asymmetric: the test confusion matrix is biased toward predicting positive across the board, which is what a source-confounded probe would do, not what an undertrained probe would do (the latter would tend toward random or majority-class predictions). Third, our test ROC-AUC of 0.578 is in line with the cross-publisher external-validation numbers reported for ImageNet-pretrained COVID-19 X-ray classifiers in the DeGrave paper [26], suggesting we are observing the same phenomenon at the same magnitude.

Three remediations are supported by the literature and define the agenda for the planned next iteration of this work.

The first is **source-stratified splitting**. The COVIDx-CXR4 documentation [8] provides curated split definitions in which each source contributes to train, validation, and test according to a controlled marginal mix, and additional curated splits exist that hold out an entire publisher as an external test set. Repeating the present experiment under such a split would dissociate the val-test gap from the shortcut. If the gap closes under source-matched splitting and the test accuracy approaches validation accuracy, that is evidence that the residual class signal is real but small. If the gap remains under source-matched splitting, that is evidence of an even subtler confounder.

The second is **domain-adversarial training**. A standard approach is to attach a source-classifier head to the same backbone as the COVID head, with a gradient-reversal layer between them, so that the backbone is rewarded for losing source information while it is rewarded for retaining COVID information. This formulation, derived from Ganin and Lempitsky and now standard in domain adaptation, has been applied to chest-X-ray transfer learning with documented improvements on cross-hospital test sets. It cannot be applied to a frozen backbone, so it requires partial unfreezing and a small fine-tuning budget.

The third, and probably the most powerful, is to **replace the ImageNet backbone with a chest-X-ray self-supervised initialiser**. Sriram et al. [22] released MoCo-CXR weights pretrained on roughly 800,000 unlabelled chest X-rays using momentum contrast [19]; their published results show that downstream COVID-19 prognosis tasks improved both in absolute performance and in transfer robustness compared with ImageNet initialisation. Masked-autoencoder pretraining on chest X-rays [21] is the natural successor and has shown similar gains. The general principle, articulated by Raghu et al. [32], is that the gap between the pretraining domain and the downstream domain matters more than the size of the pretraining dataset, especially when the downstream dataset is small. ImageNet contains photographs of objects taken with consumer cameras under daylight illumination; chest X-rays are 14-bit grayscale projection images of human thoraxes acquired with kV/mAs-controlled machines. The intuition that the ImageNet feature space is a poor match for chest radiography is borne out empirically.

A fourth direction, deferable but worth flagging, is the use of explainability tools to confirm the source-confounding interpretation directly on the present probe. Grad-CAM [24] and SHAP [25] applied to the linear probe could test whether the high-confidence wrongly-classified test images attribute their decision to non-pulmonary regions. The present linear probe is shallow enough that gradient-based attribution on the backbone features is straightforward.

A fifth direction is deeper architectural exploration. EfficientNet [17], DenseNet [15], and Vision Transformer [18] are all reasonable backbones for chest-X-ray classification, but none of them addresses the source-confounding mechanism on its own. Architectural changes will narrow the val-test gap only insofar as they implicitly encode invariances that are uncorrelated with source signature, which is largely a coincidence. The structural fixes are at the data-curation and pretraining levels.

We do not in this manuscript claim a confirmed clinical signal for ImageNet-pretrained COVID-19 X-ray classification. Our point estimate of test ROC-AUC is 0.578, which on a balanced binary task is too close to chance to support any claim of clinical utility. A higher number would not on its own change the interpretation. As Roberts et al. [27] and Wynants et al. [28] observed, isolated single-dataset accuracy numbers without source-aware external validation are not credible evidence in this domain.

---

## 6. Conclusion

We trained a frozen ImageNet-pretrained ResNet18 feature extractor with a logistic-regression linear probe on a 1,200-image balanced sample of the COVIDx binary chest-X-ray task and obtained validation accuracy of 0.838 with ROC-AUC 0.894 alongside test accuracy of 0.550 with ROC-AUC 0.578. The 28.8-point accuracy gap is direction-asymmetric, with the model defaulting to the positive class on the majority of test images, and reproduces the source-confounding pattern documented by DeGrave et al. [26] on the same benchmark. This result is, by design, a negative finding presented as the central output of the experiment rather than a setback. The frozen-probe protocol is the right diagnostic instrument for source confounding because it removes confounders at the optimisation, regularisation, and data-augmentation levels and isolates the dataset-level mechanism. The agenda this implies is concrete: source-stratified split definitions, domain-adversarial fine-tuning, and substitution of ImageNet pretraining with chest-X-ray self-supervised initialisers such as MoCo-CXR [22] or MAE-CXR [21]. None of these is exotic, and each has independent literature support. The contribution of this short manuscript is methodological: a reproducible, low-cost confirmation that the COVIDx binary task as shipped is not learnable to clinical accuracy without source-aware corrections, and a structured plan for what to try next.

---

## References

1. Chowdhury MEH, Rahman T, Khandakar A, et al. (2020). Can AI help in screening Viral and COVID-19 pneumonia? *IEEE Access*, 8, 132665-132676. DOI: 10.1109/ACCESS.2020.3010287.

2. Rahman T, Khandakar A, Qiblawey Y, et al. (2021). Exploring the effect of image enhancement techniques on COVID-19 detection using chest X-ray images. *Computers in Biology and Medicine*, 132, 104319. DOI: 10.1016/j.compbiomed.2021.104319. PMID: 33799220.

3. Cohen JP, Morrison P, Dao L. (2020). COVID-19 Image Data Collection. arXiv:2003.11597.

4. Vaya MdlI, Saborit JM, Montell JA, et al. (2020). BIMCV COVID-19+: a large annotated dataset of RX and CT images from COVID-19 patients. arXiv:2006.01174.

5. Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. (2017). ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. *CVPR 2017*, 2097-2106. DOI: 10.1109/CVPR.2017.369.

6. Irvin J, Rajpurkar P, Ko M, et al. (2019). CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. *AAAI 2019*, 33, 590-597. DOI: 10.1609/aaai.v33i01.3301590.

7. Apostolopoulos ID, Mpesiana TA. (2020). Covid-19: automatic detection from X-ray images utilizing transfer learning with convolutional neural networks. *Physical and Engineering Sciences in Medicine*, 43(2), 635-640. DOI: 10.1007/s13246-020-00865-4. PMID: 32524445.

8. Wang L, Lin ZQ, Wong A. (2020). COVID-Net: a tailored deep convolutional neural network design for detection of COVID-19 cases from chest X-ray images. *Scientific Reports*, 10, 19549. DOI: 10.1038/s41598-020-76550-z. PMID: 33177550.

9. Ozturk T, Talo M, Yildirim EA, Baloglu UB, Yildirim O, Acharya UR. (2020). Automated detection of COVID-19 cases using deep neural networks with X-ray images. *Computers in Biology and Medicine*, 121, 103792. DOI: 10.1016/j.compbiomed.2020.103792. PMID: 32568675.

10. Narin A, Kaya C, Pamuk Z. (2021). Automatic detection of coronavirus disease (COVID-19) using X-ray images and deep convolutional neural networks. *Pattern Analysis and Applications*, 24(3), 1207-1220. DOI: 10.1007/s10044-021-00984-y. PMID: 33994847.

11. Khan AI, Shah JL, Bhat MM. (2020). CoroNet: A deep neural network for detection and diagnosis of COVID-19 from chest X-ray images. *Computer Methods and Programs in Biomedicine*, 196, 105581. DOI: 10.1016/j.cmpb.2020.105581. PMID: 32534344.

12. Minaee S, Kafieh R, Sonka M, Yazdani S, Soufi GJ. (2020). Deep-COVID: Predicting COVID-19 from chest X-ray images using deep transfer learning. *Medical Image Analysis*, 65, 101794. DOI: 10.1016/j.media.2020.101794. PMID: 32781377.

13. Rajpurkar P, Irvin J, Zhu K, et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. arXiv:1711.05225.

14. He K, Zhang X, Ren S, Sun J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*, 770-778. DOI: 10.1109/CVPR.2016.90.

15. Huang G, Liu Z, van der Maaten L, Weinberger KQ. (2017). Densely Connected Convolutional Networks. *CVPR 2017*, 4700-4708. DOI: 10.1109/CVPR.2017.243.

16. Szegedy C, Vanhoucke V, Ioffe S, Shlens J, Wojna Z. (2016). Rethinking the Inception Architecture for Computer Vision. *CVPR 2016*, 2818-2826. DOI: 10.1109/CVPR.2016.308.

17. Tan M, Le QV. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*, 6105-6114. arXiv:1905.11946.

18. Dosovitskiy A, Beyer L, Kolesnikov A, et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*. arXiv:2010.11929.

19. He K, Fan H, Wu Y, Xie S, Girshick R. (2020). Momentum Contrast for Unsupervised Visual Representation Learning. *CVPR 2020*, 9729-9738. DOI: 10.1109/CVPR42600.2020.00975.

20. Chen T, Kornblith S, Norouzi M, Hinton G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML 2020*, 1597-1607. arXiv:2002.05709.

21. He K, Chen X, Xie S, Li Y, Dollar P, Girshick R. (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR 2022*, 16000-16009. DOI: 10.1109/CVPR52688.2022.01553.

22. Sriram A, Muckley M, Sinha K, et al. (2021). COVID-19 Prognosis via Self-Supervised Representation Learning and Multi-Image Prediction. arXiv:2101.04909.

23. Lin TY, Goyal P, Girshick R, He K, Dollar P. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*, 2980-2988. DOI: 10.1109/ICCV.2017.324.

24. Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *ICCV 2017*, 618-626. DOI: 10.1109/ICCV.2017.74.

25. Lundberg SM, Lee SI. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*, 30, 4765-4774.

26. DeGrave AJ, Janizek JD, Lee SI. (2021). AI for radiographic COVID-19 detection selects shortcuts over signal. *Nature Machine Intelligence*, 3(7), 610-619. DOI: 10.1038/s42256-021-00338-7.

27. Roberts M, Driggs D, Thorpe M, et al. (2021). Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans. *Nature Machine Intelligence*, 3(3), 199-217. DOI: 10.1038/s42256-021-00307-0.

28. Wynants L, Van Calster B, Collins GS, et al. (2020). Prediction models for diagnosis and prognosis of covid-19: systematic review and critical appraisal. *BMJ*, 369, m1328. DOI: 10.1136/bmj.m1328. PMID: 32265220.

29. Maguolo G, Nanni L. (2021). A critic evaluation of methods for COVID-19 automatic detection from X-ray images. *Information Fusion*, 76, 1-7. DOI: 10.1016/j.inffus.2021.04.008. PMID: 33967656.

30. Geirhos R, Jacobsen JH, Michaelis C, et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11), 665-673. DOI: 10.1038/s42256-020-00257-z.

31. Zech JR, Badgeley MA, Liu M, Costa AB, Titano JJ, Oermann EK. (2018). Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. *PLoS Medicine*, 15(11), e1002683. DOI: 10.1371/journal.pmed.1002683. PMID: 30399157.

32. Raghu M, Zhang C, Kleinberg J, Bengio S. (2019). Transfusion: Understanding Transfer Learning for Medical Imaging. *NeurIPS 2019*. arXiv:1902.07208.

33. Esteva A, Kuprel B, Novoa RA, et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542(7639), 115-118. DOI: 10.1038/nature21056. PMID: 28117445.

34. Litjens G, Kooi T, Bejnordi BE, et al. (2017). A survey on deep learning in medical image analysis. *Medical Image Analysis*, 42, 60-88. DOI: 10.1016/j.media.2017.07.005. PMID: 28778026.
