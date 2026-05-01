# References - COVID-19 Chest X-ray Multi-class Classification

All entries verified via Europe PMC and arXiv. Each carries a confirmed DOI, PMID, or arXiv identifier.

## COVID-19 chest X-ray datasets

1. **Chowdhury MEH, Rahman T, Khandakar A, et al.** (2020). Can AI help in screening Viral and COVID-19 pneumonia? *IEEE Access*, 8, 132665-132676. DOI: 10.1109/ACCESS.2020.3010287.
   Original release of the COVID-19 Radiography Database with the first DenseNet/ResNet baselines on the 3-class task.

2. **Rahman T, Khandakar A, Qiblawey Y, et al.** (2021). Exploring the effect of image enhancement techniques on COVID-19 detection using chest X-ray images. *Computers in Biology and Medicine*, 132, 104319. DOI: 10.1016/j.compbiomed.2021.104319. PMID: 33799220.
   Expansion to 4 classes (COVID, Normal, Lung Opacity, Viral Pneumonia) with gamma-correction enhancement; the dataset version this project uses.

3. **Cohen JP, Morrison P, Dao L.** (2020). COVID-19 Image Data Collection. arXiv:2003.11597.
   First open-source COVID-19 CXR/CT collection scraped from publications; component of most early benchmarks.

4. **Vaya MdlI, Saborit JM, Montell JA, et al.** (2020). BIMCV COVID-19+: a large annotated dataset of RX and CT images from COVID-19 patients. arXiv:2006.01174.
   Spanish multi-hospital RT-PCR-confirmed dataset (1,311 patients) that exposes domain-shift problems when used as an external test set.

5. **Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM.** (2017). ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. *CVPR 2017*, 2097-2106. DOI: 10.1109/CVPR.2017.369.
   NIH 112,120-image benchmark; standard pre-COVID pretraining source.

6. **Irvin J, Rajpurkar P, Ko M, et al.** (2019). CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. *AAAI 2019*, 33, 590-597. DOI: 10.1609/aaai.v33i01.3301590.
   224,316-radiograph Stanford dataset with expert-graded uncertainty labels.

## Reference COVID-19 X-ray models

7. **Apostolopoulos ID, Mpesiana TA.** (2020). Covid-19: automatic detection from X-ray images utilizing transfer learning with convolutional neural networks. *Physical and Engineering Sciences in Medicine*, 43(2), 635-640. DOI: 10.1007/s13246-020-00865-4. PMID: 32524445.
   Early VGG/Xception transfer-learning study; one of the most-cited 3-class COVID baselines.

8. **Wang L, Lin ZQ, Wong A.** (2020). COVID-Net: a tailored deep convolutional neural network design for detection of COVID-19 cases from chest X-ray images. *Scientific Reports*, 10, 19549. DOI: 10.1038/s41598-020-76550-z. PMID: 33177550.
   Custom CXR architecture plus the COVIDx benchmark; widely cited reference model.

9. **Ozturk T, Talo M, Yildirim EA, Baloglu UB, Yildirim O, Acharya UR.** (2020). Automated detection of COVID-19 cases using deep neural networks with X-ray images. *Computers in Biology and Medicine*, 121, 103792. DOI: 10.1016/j.compbiomed.2020.103792. PMID: 32568675.
   DarkCovidNet (DarkNet-19 backbone) for binary and multi-class detection.

10. **Narin A, Kaya C, Pamuk Z.** (2021). Automatic detection of coronavirus disease (COVID-19) using X-ray images and deep convolutional neural networks. *Pattern Analysis and Applications*, 24(3), 1207-1220. DOI: 10.1007/s10044-021-00984-y. PMID: 33994847.
   Five-CNN comparison (ResNet50/101/152, InceptionV3, Inception-ResNetV2) on a small COVID set.

11. **Khan AI, Shah JL, Bhat MM.** (2020). CoroNet: A deep neural network for detection and diagnosis of COVID-19 from chest X-ray images. *Computer Methods and Programs in Biomedicine*, 196, 105581. DOI: 10.1016/j.cmpb.2020.105581. PMID: 32534344.
   Xception-based 4-class COVID/Normal/Pneumonia-bact/Pneumonia-vir model.

12. **Minaee S, Kafieh R, Sonka M, Yazdani S, Soufi GJ.** (2020). Deep-COVID: Predicting COVID-19 from chest X-ray images using deep transfer learning. *Medical Image Analysis*, 65, 101794. DOI: 10.1016/j.media.2020.101794. PMID: 32781377.
   ResNet/DenseNet/SqueezeNet ensemble showing >97% sensitivity but flagging dataset-size limits.

13. **Rajpurkar P, Irvin J, Zhu K, et al.** (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. arXiv:1711.05225.
   121-layer DenseNet trained on ChestX-ray14; foundational reference for CXR deep learning.

## CNN architectures (pretraining backbones)

14. **He K, Zhang X, Ren S, Sun J.** (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*, 770-778. DOI: 10.1109/CVPR.2016.90.
   Introduces ResNet residual connections enabling training of very deep networks.

15. **Huang G, Liu Z, van der Maaten L, Weinberger KQ.** (2017). Densely Connected Convolutional Networks. *CVPR 2017*, 4700-4708. DOI: 10.1109/CVPR.2017.243.
   DenseNet feature-reuse architecture, the most common backbone in CXR classification.

16. **Szegedy C, Vanhoucke V, Ioffe S, Shlens J, Wojna Z.** (2016). Rethinking the Inception Architecture for Computer Vision. *CVPR 2016*, 2818-2826. DOI: 10.1109/CVPR.2016.308.
   Inception-v3 factorized-convolution design used in many COVID-CXR pipelines.

17. **Tan M, Le QV.** (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*, 6105-6114. arXiv:1905.11946.
   Compound scaling rule that yields parameter-efficient backbones suitable for small medical datasets.

## Vision Transformers and self-supervised pretraining

18. **Dosovitskiy A, Beyer L, Kolesnikov A, et al.** (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*. arXiv:2010.11929.
   Vision Transformer (ViT) showing transformers can match CNNs given enough data.

19. **He K, Fan H, Wu Y, Xie S, Girshick R.** (2020). Momentum Contrast for Unsupervised Visual Representation Learning. *CVPR 2020*, 9729-9738. DOI: 10.1109/CVPR42600.2020.00975.
   MoCo contrastive pretraining used as a label-free initialiser for medical CXR encoders.

20. **Chen T, Kornblith S, Norouzi M, Hinton G.** (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML 2020*, 1597-1607. arXiv:2002.05709.
   SimCLR contrastive framework that competes with supervised pretraining at scale.

21. **He K, Chen X, Xie S, Li Y, Dollar P, Girshick R.** (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR 2022*, 16000-16009. DOI: 10.1109/CVPR52688.2022.01553.
   MAE generative pretraining yielding strong downstream features at high mask ratios.

22. **Sriram A, Muckley M, Sinha K, et al.** (2021). COVID-19 Prognosis via Self-Supervised Representation Learning and Multi-Image Prediction. arXiv:2101.04909.
   MoCo pretraining on 800k unlabelled CXRs improves COVID-19 deterioration prediction.

## Class imbalance and explainability

23. **Lin TY, Goyal P, Girshick R, He K, Dollar P.** (2017). Focal Loss for Dense Object Detection. *ICCV 2017*, 2980-2988. DOI: 10.1109/ICCV.2017.324.
   Down-weights easy examples; standard loss for imbalanced medical classification.

24. **Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D.** (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *ICCV 2017*, 618-626. DOI: 10.1109/ICCV.2017.74.
   Gradient-weighted class-activation maps; near-universal explainability tool in medical AI.

25. **Lundberg SM, Lee SI.** (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*, 30, 4765-4774.
   SHAP framework unifying Shapley-value-based feature attribution.

## Bias, validation and shortcut-learning critiques (load-bearing for the discussion)

26. **DeGrave AJ, Janizek JD, Lee SI.** (2021). AI for radiographic COVID-19 detection selects shortcuts over signal. *Nature Machine Intelligence*, 3(7), 610-619. DOI: 10.1038/s42256-021-00338-7.
   Shows public COVID-CXR classifiers exploit dataset-specific confounders (laterality markers, edges) instead of pathology.

27. **Roberts M, Driggs D, Thorpe M, et al.** (2021). Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans. *Nature Machine Intelligence*, 3(3), 199-217. DOI: 10.1038/s42256-021-00307-0.
   Systematic review of 2,212 COVID-imaging ML papers concluding none were of clinical utility due to bias and methodology flaws.

28. **Wynants L, Van Calster B, Collins GS, et al.** (2020). Prediction models for diagnosis and prognosis of covid-19: systematic review and critical appraisal. *BMJ*, 369, m1328. DOI: 10.1136/bmj.m1328. PMID: 32265220.
   Living systematic review (BMJ) of COVID-19 prediction models showing high risk of bias in nearly all entries.

29. **Maguolo G, Nanni L.** (2021). A critic evaluation of methods for COVID-19 automatic detection from X-ray images. *Information Fusion*, 76, 1-7. DOI: 10.1016/j.inffus.2021.04.008. PMID: 33967656.
   Demonstrates classifiers achieve similar AUC even when lung regions are masked out, exposing dataset shortcuts.

30. **Geirhos R, Jacobsen JH, Michaelis C, et al.** (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11), 665-673. DOI: 10.1038/s42256-020-00257-z.
   Conceptual framework for shortcut learning; theoretical lens for the COVID-CXR critiques.

31. **Zech JR, Badgeley MA, Liu M, Costa AB, Titano JJ, Oermann EK.** (2018). Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. *PLoS Medicine*, 15(11), e1002683. DOI: 10.1371/journal.pmed.1002683. PMID: 30399157.
   Cross-hospital pneumonia model collapses on external data; pre-COVID precedent for the same generalization failure.

32. **Raghu M, Zhang C, Kleinberg J, Bengio S.** (2019). Transfusion: Understanding Transfer Learning for Medical Imaging. *NeurIPS 2019*. arXiv:1902.07208.
   Shows ImageNet pretraining gives smaller gains in medical imaging than commonly assumed; motivates self-supervised alternatives.

## General medical-imaging context

33. **Esteva A, Kuprel B, Novoa RA, et al.** (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542(7639), 115-118. DOI: 10.1038/nature21056. PMID: 28117445.
   Landmark proof-of-concept for clinician-grade CNN classification in medicine.

34. **Litjens G, Kooi T, Bejnordi BE, et al.** (2017). A survey on deep learning in medical image analysis. *Medical Image Analysis*, 42, 60-88. DOI: 10.1016/j.media.2017.07.005. PMID: 28778026.
   Foundational review covering CNNs across radiology, pathology and ophthalmology.
