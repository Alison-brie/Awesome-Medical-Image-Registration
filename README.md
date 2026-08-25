
# Awesome Medical Image Registration [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/Alison-brie/Awesome-ImageRegistration/)

:wave: Hi! This repo is a collection of AWESOME things about :star2:**Medical Image Registration**:star2:, including useful materials, papers, code. Please feel free to start and fork.

For researchers who are new to registration, it is recommended to read papers in **chronological order** to study the **evolution of research trends**.


TODO:
- [ ] Add a list of organs in medical image registration


## Contributing
:running: **We will keep updating it.** :running:    

Please feel free to send me [pull requests](https://github.com/Alison-brie/Awesome-ImageRegistration/pulls) or email (alisonbrielee@gmail.com) to add links.

## Table of Contents

- [Related Awesome Lists](#related-awesome-lists)
- [Papers](#papers)
  - [1. Fundamentals & Transformation Models](#1-fundamentals--transformation-models)
  - [2. Learning Methods](#2-learning-methods)
  - [3. Generalizable & Foundation Registration](#3-generalizable--foundation-registration)
  - [4. Registration Settings](#4-registration-settings)
  - [5. Registration Quality Assessment](#5-registration-quality-assessment)
  - [6. Registration-Enabled Medical Image Analysis](#6-registration-enabled-medical-image-analysis)
- [7. Datasets & Challenges](#7-datasets--challenges)
- [8. Software](#8-software)

## Related Awesome Lists
 - [Awesome Optical Flow](https://github.com/hzwer/Awesome-Optical-Flow)
 - [Awesome Image-to-Image Translation](https://github.com/weihaox/awesome-image-translation)
 - [Awesome Medical Imaging](https://github.com/fepegar/awesome-medical-imaging)
 - [Awesome Machine Learning in Biomedical(Healthcare) Imaging](https://github.com/XindiWu/Awesome-Machine-Learning-in-Biomedical-Healthcare-Imaging)

## Papers

The papers are organized along complementary dimensions including transformation models, learning methodology, quality assessment, registration settings, and registration-enabled applications. Some representative papers are cross-listed when they naturally span multiple categories.

### Table of Contents
- [1. Fundamentals & Transformation Models](#1-fundamentals--transformation-models)
  - [Rigid / Affine Registration](#rigid--affine-registration)
  - [Deformable Registration](#deformable-registration)
  - [Diffeomorphic / Topology-preserving Registration](#diffeomorphic--topology-preserving-registration)
  - [Composite / Affine-to-Deformable Registration](#composite--affine-to-deformable-registration)
- [2. Learning Methods](#2-learning-methods)
  - [CNN-based Registration](#cnn-based-registration)
  - [Pyramid / Multi-scale Registration](#pyramid--multi-scale-registration)
  - [Transformer & MLP Registration](#transformer--mlp-registration)
  - [Iterative / Multi-stage Registration](#iterative--multi-stage-registration)
  - [Discrete & Instance Optimization](#discrete--instance-optimization)
  - [Hyperparameter / Adaptive Registration](#hyperparameter--adaptive-registration)
  - [Efficient Registration](#efficient-registration)
- [3. Generalizable & Foundation Registration](#3-generalizable--foundation-registration)
  - [Universal / Generalist Registration Models](#universal--generalist-registration-models)
  - [Foundation Features for Registration](#foundation-features-for-registration)
- [4. Registration Quality Assessment](#4-registration-quality-assessment)
- [5. Registration Settings](#5-registration-settings)
  - [Multi-modal Registration](#multi-modal-registration)
  - [2D–3D Registration](#2d3d-registration)
  - [Longitudinal Registration](#longitudinal-registration)
  - [Histological / Microscopy Registration](#histological--microscopy-registration)
  - [Pathology-aware / Missing-correspondence Registration](#pathology-aware--missing-correspondence-registration)
  - [Cortical Surface Registration](#cortical-surface-registration)
- [6. Registration-Enabled Medical Image Analysis](#6-registration-enabled-medical-image-analysis)
  - [Registration-guided Segmentation](#registration-guided-segmentation)
  - [Atlas & Template Construction](#atlas--template-construction)
  - [Motion Estimation & Tracking](#motion-estimation--tracking)
  - [Representation Learning](#representation-learning)
  - [Image-guided Intervention & Surgical Navigation](#image-guided-intervention--surgical-navigation)

### 1. Fundamentals & Transformation Models

#### Rigid / Affine Registration
* [CVPR 2022] Affine Medical Image Registration with Coarse-to-Fine Vision Transformer [[pdf]](https://arxiv.org/abs/2203.15216) [[code]](https://github.com/cwmok/C2FViT)
* [MICCAI 2023] X-ray to ct rigid registration using scene coordinate regression [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_74) [[code]](https://github.com/Pragyanstha/SCR-Registration)
* [MedIA 2025] PViT-AIR: Puzzling vision transformer-based affine image registration for multi histopathology and faxitron images of breast tissue [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S1361841524002810) [[code]](https://github.com/pimed/PViT-AIR)

#### Deformable Registration
> Most learning-based methods collected in this repository address deformable registration. They are organized primarily by methodology in [Learning Methods](#2-learning-methods), with transformation-specific methods listed below.

#### Diffeomorphic / Topology-preserving Registration
* [MICCAI 2018] Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration [[pdf]](https://arxiv.org/abs/1805.04605) [[code]](https://github.com/voxelmorph/voxelmorph)
* [TMI 2019] Learning a Probabilistic Model for Diffeomorphic Registration [[pdf]](https://arxiv.org/pdf/1812.07460)
* [MedIA 2019] Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces [[pdf]](https://arxiv.org/abs/1903.03545) [[code]](https://github.com/voxelmorph/voxelmorph) 
* [NeurIPS 2019] Region-specific Diffeomorphic Metric Mapping [[pdf]](https://drive.google.com/file/d/1kIuunw6FP2ek8ZsLw92zL6RJw02YU7Nx) [[code]](https://github.com/uncbiag/easyreg)
* [CVPR 2019] Metric Learning for Image Registration [[pdf]](https://drive.google.com/file/d/1kIuunw6FP2ek8ZsLw92zL6RJw02YU7Nx) [[code]](https://github.com/uncbiag/mermaid)
* [CVPR 2020] Fast Symmetric Diffeomorphic Image Registration with Convolutional Neural Networks [[pdf]](https://arxiv.org/abs/2003.09514) [[code]](https://github.com/cwmok/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks)
* [ICCV 2023] ICON: Learning Regular Maps Through Inverse Consistency [[pdf]](https://arxiv.org/pdf/2105.04459) [[code]](https://github.com/uncbiag/ICON)
* [CVPR 2023] GradICON: Approximate Diffeomorphisms via Gradient Inverse Consistency [[pdf]](https://drive.google.com/file/d/1j8u5n50knQUxhnHp1OMGEwsl8CX-lODX) [[code]](https://github.com/uncbiag/ICON)
* [MedIA 2019] R2Net: Efficient and flexible diffeomorphic image registration using Lipschitz continuous residual networks [[pdf]](https://www.sciencedirect.com/science/article/pii/S1361841523001779) [[code]](https://github.com/ankitajoshi15/R2Net)
* [ECCV 2024] NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration [[pdf]](https://arxiv.org/abs/2309.07322) [[code]](https://github.com/uncbiag/NePhi)
* [MELBA 2024] SITReg: Multi-resolution architecture for symmetric, inverse consistent, and topology preserving image registration [[pdf]](https://arxiv.org/abs/2303.10211) [[code]](https://github.com/honkamj/SITReg?tab=readme-ov-file)
* [CVPR 2025] CARL: A Framework for Equivariant Image Registration [[pdf]](https://arxiv.org/pdf/2405.16738)
* [CVPR 2026] Learning Diffeomorphism for Medical Image Registration with Time-Embedded Architectures Using Semigroup Regularization [[pdf]](https://openaccess.thecvf.com/content/CVPR2026/papers/Matinkia_Learning_Diffeomorphism_for_Medical_Image_Registration_with_Time-Embedded_Architectures_Using_CVPR_2026_paper.pdf) [[code]](https://mattkia.github.io/SGDIR/)

#### Composite / Affine-to-Deformable Registration
* [MedIA 2019] A deep learning framework for unsupervised affine and deformable image registration [[pdf]](https://arxiv.org/pdf/1809.06130) [[code]](https://github.com/BDdeVos/TorchIR)
* [CVPR 2019] Networks for Joint Affine and Non-Parametric Image Registration [[pdf]](https://arxiv.org/pdf/1903.08811.pdf) [[code]](https://github.com/uncbiag/easyreg)
* [JBHI 2020] Unsupervised 3D End-to-End Medical Image Registration with Volume Tweening Network [[pdf]](https://arxiv.org/pdf/1902.05020) [[code]](https://github.com/microsoft/Recursive-Cascaded-Networks)
* [MICCAI 2021] SAME: Deformable Image Registration based on Self-supervised Anatomical Embeddings [[pdf]](https://arxiv.org/abs/2109.11572) [[code]](https://github.com/alibaba-damo-academy/same)
* [MICCAI 2023] Non-iterative Coarse-to-Fine Transformer Networks for Joint Affine and Deformable Image Registration [[pdf]](https://arxiv.org/abs/2307.03421) [[code]](https://github.com/MungoMeng/Registration-NICE-Trans)
* [Arxiv 2023] SAME++: Deformable Image Registration based on Self-supervised Anatomical Embeddings [[pdf]](https://doi.org/10.48550/arXiv.2311.14986) [[code]](https://github.com/alibaba-damo-academy/same)

### 2. Learning Methods

#### CNN-based Registration
* [MICCAI Workshop 2017] End-to-end unsupervised deformable image registration with a convolutional neural network [[pdf]](https://arxiv.org/pdf/1704.06065)
* [CVPR 2018] An Unsupervised Learning Model for Deformable Medical Image Registration [[pdf]](https://arxiv.org/abs/1802.02604) [[code]](https://github.com/voxelmorph/voxelmorph)
* [Arxiv 2018] Inverse-Consistent Deep Networks for Unsupervised Deformable Image Registration [[pdf]](https://arxiv.org/pdf/1809.03443) [[code]](https://github.com/zhangjun001/ICNet)
* [MICCAI 2019] Unsupervised Deformable Image Registration Using Cycle-Consistent CNN [[pdf]](https://arxiv.org/abs/1907.01319) [[code]](https://github.com/boahK/MEDIA_CycleMorph)
* [TMI 2019] VoxelMorph: A Learning Framework for Deformable Medical Image Registration [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-32226-7_19) [[code]](https://github.com/boahK/MEDIA_CycleMorph)

#### Pyramid / Multi-scale Registration
* [MICCAI 2019] Dual-Stream Pyramid Registration Network [[pdf]](https://arxiv.org/pdf/1909.11966) [[code]](https://github.com/kangmiao15/Dual-Stream-PRNet-Plus)
* [MICCAI 2020] Large Deformation Diffeomorphic Image Registration with Laplacian Pyramid Networks [[pdf]](https://arxiv.org/abs/2006.16148) [[code]](https://github.com/cwmok/LapIRN)
* [TMI 2021] Learning a model-driven variational network for deformable image registration [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9525092) [[code]](https://github.com/xi-jia/Learning-a-Model-Driven-Variational-Network-for-Deformable-Image-Registration)
* [MICCAI 2022] NICE-Net: a Non-Iterative Coarse-to-finE registration Network for deformable image registration [[pdf]](https://arxiv.org/abs/2206.12596) [[code]](https://github.com/MungoMeng/Registration-NICE-Net)
* [TMI 2023] Self-Distilled Hierarchical Network for Unsupervised Deformable Image Registration [[pdf]](https://ieeexplore.ieee.org/abstract/document/10042453) [[code]](https://github.com/Blcony/SDHNet)
* [TMI 2024] GroupMorph: Medical Image Registration via Grouping Network with Contextual Fusion [[pdf]](https://ieeexplore.ieee.org/document/10530124) [[code]](https://github.com/TVayne/GroupMorph)

#### Transformer & MLP Registration
* [MedIA 2022] TransMorph: Transformer for unsupervised medical image registration [[pdf]](https://arxiv.org/abs/2111.10480) [[code]](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)
* [MICCAI 2022] Swin-VoxelMorph: A Symmetric Unsupervised Learning Model for Deformable Medical Image Registration Using Swin Transformer [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_8) [[code]](https://github.com/CJSOrange/DMR-Deformer)
* [MICCAI 2022] Deformer: Towards displacement field learning for unsupervised medical image registration [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_14) [[code]](https://github.com/YongpeiZhu/Swin-VoxelMorph)
* [MICCAI 2022] XMorpher: Full Transformer for Deformable Medical Image Registration via Cross Attention [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_21) [[code]](https://github.com/Solemoon/XMorpher)
* [CVPR 2022] Affine Medical Image Registration with Coarse-to-Fine Vision Transformer [[pdf]](https://arxiv.org/abs/2203.15216) [[code]](https://github.com/cwmok/C2FViT)
* [TMI 2023] TransMatch: a transformer-based multilevel dual-stream feature matching network for unsupervised deformable image registration [[pdf]](https://ieeexplore.ieee.org/abstract/document/10158729) [[code]](https://github.com/tzayuan/TransMatch_TMI)
* [MICCAI 2023] ModeT: Learning Deformable Image Registration via Motion Decomposition Transformer [[pdf]](https://arxiv.org/pdf/2306.05688) [[code]](https://github.com/ZAX130/SmileCode)
* [CVPR 2024] H-ViT: A Hierarchical Vision Transformer for Deformable Image Registration [[pdf]](https://openaccess.thecvf.com/content/CVPR2024/papers/Ghahremani_H-ViT_A_Hierarchical_Vision_Transformer_for_Deformable_Image_Registration_CVPR_2024_paper.pdf) [[code]](https://github.com/mogvision/hvit)
* [CVPR 2024] Correlation-aware Coarse-to-fine MLPs for Deformable Medical Image Registration [[pdf]](https://openaccess.thecvf.com/content/CVPR2024/papers/Meng_Correlation-aware_Coarse-to-fine_MLPs_for_Deformable_Medical_Image_Registration_CVPR_2024_paper.pdf) [[code]](https://github.com/MungoMeng/Registration-CorrMLP)

#### Iterative / Multi-stage Registration
* [ICCV 2019] Recursive Cascaded Networks for Unsupervised Medical Image Registration [[pdf]](https://arxiv.org/abs/1907.12353) [[code]](https://github.com/microsoft/Recursive-Cascaded-Networks)
* [TPAMI 2021] Learning Deformable Image Registration from Optimization: Perspective, Modules, Bilevel Training and Beyond [[pdf]](https://arxiv.org/abs/2004.14557) [[code]](https://github.com/Alison-brie/MultiPropReg)
* [MICCAI 2023] PIViT: Large Deformation Image Registration with Pyramid-Iterative Vision Transformer [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_57) [[code]](https://github.com/Torbjorn1997/PIViT)
* [TMI 2024] Recursive Deformable Pyramid Network for Unsupervised Medical Image Registration [[pdf]](https://ieeexplore.ieee.org/document/10423043) [[code]](https://github.com/ZAX130/RDP)
* [CVPR 2024] IIRP-Net: Iterative Inference Residual Pyramid Network for Enhanced Image Registration [[pdf]](https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_IIRP-Net_Iterative_Inference_Residual_Pyramid_Network_for_Enhanced_Image_Registration_CVPR_2024_paper.pdf) [[code]](https://github.com/Torbjorn1997/IIRP-Net)

#### Discrete & Instance Optimization
* [MICCAI 2019] Closing the Gap between Deep and Conventional Image Registration using Probabilistic Dense Displacement Networks [[pdf]](https://arxiv.org/abs/1907.10931) [[code]](https://github.com/multimodallearning/pdd_net)
* [MICCAI 2020] Highly accurate and memory efficient unsupervised learning-based discrete CT registration using 2.5 D displacement search [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_19) [[code]](https://github.com/multimodallearning/pdd2.5/)
* [MICCAI Workshop 2021] Fast 3D registration with accurate optimisation and little learning for Learn2Reg 2021 [[pdf]](https://arxiv.org/abs/2112.03053) [[code]](https://github.com/multimodallearning/convexAdam)
* [MICCAI Workshop 2022] Voxelmorph++ going beyond the cranial vault with keypoint supervision and multi-channel instance optimisation [[pdf]](https://openreview.net/pdf?id=SrlgSXA3qAY) [[code]](https://github.com/mattiaspaul/VoxelMorphPlusPlus)
* [MICCAI 2023] SAMConvex: Fast Discrete Optimization for CT Registration using Self-supervised Anatomical Embedding and Correlation Pyramid [[pdf]](https://arxiv.org/abs/2307.09727) [[code]](https://github.com/alibaba-damo-academy/samconvex) 
* [TMI 2024] ConvexAdam: Self-Configuring Dual-Optimisation-Based 3D Multitask Medical Image Registration [[pdf]](https://ieeexplore.ieee.org/abstract/document/10681158) [[code]](https://github.com/multimodallearning/convexAdam)
* [MICCAI 2024] On-the-Fly Guidance Training for Medical Image Registration [[pdf]](https://arxiv.org/pdf/2308.15216) [[code]](https://github.com/cilix-ai/on-the-fly-guidance)
* [MICCAI 2025] VoxelOpt: Voxel-Adaptive Message Passing for Discrete Optimization in Deformable Abdominal CT Registration [[pdf]](https://arxiv.org/pdf/2506.19975) [[code]](https://github.com/tinymilky/VoxelOpt)

#### Hyperparameter / Adaptive Registration
* [IPMI 2021] HyperMorph: Amortized Hyperparameter Learning for Image Registration [[pdf]](https://arxiv.org/abs/2101.01035) [[code]](https://ahoopes.github.io/hypermorph/)
* [MICCAI 2021] Conditional Deformable Image Registration with Convolutional Neural Network [[pdf]](https://arxiv.org/abs/2106.12673) [[code]](https://github.com/cwmok/Conditional_LapIRN)
* [TPAMI 2021] Learning Deformable Image Registration from Optimization: Perspective, Modules, Bilevel Training and Beyond [[pdf]](https://arxiv.org/abs/2004.14557) [[code]](https://github.com/Alison-brie/MultiPropReg)
* [MedIA 2023] Hyper-Convolutions via Implicit Kernels for Medical Image Analysis [[pdf]](https://arxiv.org/abs/2202.02701) [[code]](https://github.com/tym002/Hyper-Convolution)
* [TIP 2023] Automated learning for deformable medical image registration by jointly optimizing network architectures and objective functions [[pdf]](https://arxiv.org/abs/2203.06810) [[code]](https://github.com/Alison-brie/AutoReg)

* [TNNLS 2024] Spatially covariant image registration with text prompts [[pdf]](https://arxiv.org/abs/2311.15607) [[code]](https://github.com/tinymilky/TextSCF)
* [ECCV 2024] Adaptive Correspondence Scoring for Unsupervised Medical Image Registration [[pdf]](https://arxiv.org/pdf/2312.00837) [[code]](https://github.com/Voldemort108X/AdaCS)
* [CVPR 2025] SACB-Net: Spatial-awareness Convolutions for Medical Image Registration [[pdf]](https://arxiv.org/pdf/2503.19592) [[code]](https://github.com/x-xc/SACB_Net)
* [MedIA 2026] Unsupervised learning of spatially varying regularization for diffeomorphic image registration [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S1361841525004335) [[code]](https://github.com/junyuchen245/Spatially-Varying-Regularization-ImgReg) 

#### Efficient Registration
* [CVPR 2020] DeepFLASH: An Efficient Network for Learning-based Medical Image Registration [[pdf]](https://arxiv.org/pdf/2004.02097) [[code]](https://github.com/jw4hv/deepflash)
* [TCSVT 2022] Cross-Resolution Distillation for Efficient 3D Medical Image Registration [[pdf]](https://ieeexplore.ieee.org/document/9782430) 
* [AAAI 2023] Fourier-Net: Fast Image Registration with Band-Limited Deformation [[pdf]](https://arxiv.org/pdf/2211.16342)  [[code]](https://github.com/xi-jia/Fourier-Net)
* [MICCAI 2024] WiNet: Wavelet-based Incremental Learning for Efficient Medical Image Registration [[pdf]](https://arxiv.org/abs/2407.13426)  [[code]](https://github.com/x-xc/WiNet)
* [TMI 2025] Decoder-Only Image Registration [[pdf]](https://ieeexplore.ieee.org/document/10967349) [[code]](https://github.com/xi-jia/LessNet)
* [TCSVT 2026] Encoder-Only Image Registration [[pdf]](https://arxiv.org/abs/2509.00451)  [[code]](https://github.com/XiangChen1994/EOIR)
* [CVPR 2026] Dynamic Stream Network for Combinatorial Explosion Problem in Deformable Medical Image Registration [[pdf]](https://openaccess.thecvf.com/content/CVPR2026/papers/Bi_Dynamic_Stream_Network_for_Combinatorial_Explosion_Problem_in_Deformable_Medical_CVPR_2026_paper.pdf)
 [[code]](https://github.com/ShaochenBi/DySNet)

### 3. Generalizable & Foundation Registration

#### Universal / Generalist Registration Models
* [MICCAI 2024] uniGradICON: A Foundation Model for Medical Image Registration [[pdf]](https://arxiv.org/abs/2403.05780) [[code]](https://github.com/uncbiag/uniGradICON)
* [Arxiv 2024] UniMo: Universal Motion Correction For Medical Images without Network Retraining [[pdf]](https://arxiv.org/abs/2409.14204) [[code]](https://github.com/IntelligentImaging/UNIMO/)
* [ICLR 2025] Learning General-purpose Biomedical Volume Representations using Randomized Synthesis [[pdf]](https://arxiv.org/abs/2411.02372)[[code]](https://github.com/neel-dey/anatomix)
* [MELBA 2025] BrainMorph: A Foundational Keypoint Model for Robust and Flexible Brain MRI Registration [[pdf]](https://arxiv.org/abs/2405.14019v3)[[code]](https://github.com/alanqrwang/brainmorph)
* [MICCAI 2025] PromptReg: Universal Medical Image Registration via Task Prompt Learning and Domain Knowledge Transfer [[pdf]](https://papers.miccai.org/miccai-2025/paper/1233_paper.pdf)[[code]](https://github.com/xiehousheng/PromptReg)
* [ICLR 2026] Unified Brain Surface and Volume Registration [[pdf]](https://arxiv.org/pdf/2512.19928v1)[[code]](https://github.com/mabulnaga/neuralign)
* [TCSVT 2026] UniReg: Conditional Unified Model for Medical Image Registration.[[pdf]](https://arxiv.org/pdf/2503.12868v2) [[code]](https://github.com/Alison-brie/UniReg)

#### Foundation Features for Registration
* [TMI 2022] SAM: Self-supervised Learning of Pixel-wise Anatomical Embeddings in Radiological Images [[pdf]](https://arxiv.org/abs/2012.02383) [[code]](https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2)
* [MICCAI 2023] SAMConvex: Fast Discrete Optimization for CT Registration using Self-supervised Anatomical Embedding and Correlation Pyramid [[pdf]](https://arxiv.org/abs/2307.09727) [[code]](https://github.com/alibaba-damo-academy/samconvex) 
* [MedIA 2025] UAE: Universal Anatomical Embedding on Multi-modality Medical Images [[pdf]](https://arxiv.org/abs/2311.15111)[[code]](https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2)
* [TMI 2025] Dino-Reg: Efficient Multimodal Image Registration with Distilled Features [[pdf]](https://ieeexplore.ieee.org/abstract/document/10988615)[[code]](https://github.com/RPIDIAL/DINO-Reg)
* [IPMI 2025] Medical Image Registration Meets Vision Foundation Model: Prototype Learning and Contour Awareness [[pdf]](https://arxiv.org/pdf/2502.11440)[[code]](https://github.com/HaoXu0507/IPMI25-SAM-Assisted-Registration)
* [MICCAI 2025] Guiding Registration with Emergent Similarity from Pre-Trained Diffusion Models [[pdf]](https://arxiv.org/pdf/2506.02419)[[code]](https://github.com/uncbiag/dgir)
* [MICCAI 2026] FSE-Reg: Enhancing 3D Deformable Registration with Frozen Large-Scale Pre-trained Segmentation Encoders [[code]](https://github.com/alibaba-damo-academy/FSE-Reg)

### 4. Registration Quality Assessment
* [MedIA 2019] Quantitative error prediction of medical image registration using regression forests [[pdf]](https://arxiv.org/abs/1905.07624)
* [MICCAI 2019] On the Applicability of Registration Uncertainty [[pdf]](https://arxiv.org/abs/1803.05266)
* [MICCAI 2023] FocalErrorNet: Uncertainty-aware focal modulation network for inter-modal registration error estimation in ultrasound-guided neurosurgery [[pdf]](https://conferences.miccai.org/2023/papers/278-Paper3362.html)
* * [MIDL 2024] Registration Quality Evaluation Metric with Self-Supervised Siamese Networks [[pdf]](https://proceedings.mlr.press/v250/kulkarni24b.html)
* [MedIA 2026] Contrastive Discrepancy: A label-free metric for deformable image registration supporting testing-time hyperparameter selection [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S1361841526002793)

### 5. Registration Settings

#### Multi-modal Registration
##### Classic
* [MedIA 2011] MIND: Modality independent neighborhood descriptor for multi-modal deformable registration [[pdf]](http://svg.dmi.unict.it/miss14/MISS2014-ReadingGroup00-All-Paper.pdf) [[code]](https://github.com/mattiaspaul/deedsBCV)
* [MedIA 2012] DRAMMS: Deformable registration via attribute matching and mutual-saliency weighting [[pdf]](https://pmc.ncbi.nlm.nih.gov/articles/PMC3012150/)
* [MICCAI 2013] Towards real-time multimodal fusion for image-guided interventions using self-similarities [[pdf]](https://www.researchgate.net/profile/Mattias-Heinrich/publication/260127659_Lecture_Notes_in_Computer_Science/links/0deec52eb61e9a9fdc000000/Lecture-Notes-in-Computer-Science.pdf) [[code]](https://github.com/mattiaspaul/deedsBCV)
* [MedIA 2014] Automatic ultrasound–MRI registration for neurosurgery using the 2D and 3D LC2 Metric [[pdf]](https://campar.cs.tum.edu/pub/fuerst2014media/fuerst2014media.pdf)
##### Learning-based
* [IPMI 2019] Unsupervised deformable registration for multi-modal images via disentangled representations [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-20351-1_19)
* [MICCAI 2019] Synthesis and inpainting-based MR-CT registration for image-guided thermal ablation of liver tumors [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-32254-0_57)
* [MICCAI 2020] Adversarial uni-and multi-modal stream networks for multimodal image registration [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_22)
* [NeurIPS 2020] CoMIR: Contrastive multimodal image representation for registration [[pdf]](https://proceedings.neurips.cc/paper/2020/hash/d6428eecbe0f7dff83fc607c5044b2b9-Abstract.html) [[code]](https://github.com/MIDA-group/CoMIR)
* [TPMAI 2021] SymReg-GAN: symmetric image registration with generative adversarial networks [[pdf]](https://ieeexplore.ieee.org/abstract/document/9440692)
* [TMI 2022] SynthMorph: learning contrast-invariant registration without acquired images [[pdf]](https://arxiv.org/abs/2004.10282)  [[code]](https://martinos.org/malte/synthmorph/)
* [MedIA 2022] Cross-modal attention for multi-modal image registration [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S1361841522002407)  [[code]](https://github.com/DIAL-RPI/Attention-Reg)
* [MICCAI 2022] ContraReg: Contrastive Learning of Multi-modality Unsupervised Deformable Image Registration [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_7) [[code]](https://github.com/jmtzt/ContraReg)
* [MICCAI 2023] DISA: DIfferentiable Similarity Approximation for Universal Multimodal Registration [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_72) [[code]](https://github.com/ImFusionGmbH/DISA-universal-multimodal-registration)
* [CVPR 2023] Indescribable Multi-modal Spatial Evaluator [[pdf]](https://openaccess.thecvf.com/content/CVPR2023/papers/Kong_Indescribable_Multi-Modal_Spatial_Evaluator_CVPR_2023_paper.pdf) [[code]](https://github.com/Kid-Liet/IMSE)
* [CVPR 2024] Modality-Agnostic Structural Image Representation Learning for Deformable Multi-Modality Medical Image Registration [[pdf]](https://arxiv.org/abs/2402.18933)
* [MICCAI 2025] Mono-Modalizing Extremely Heterogeneous Multi-Modal Medical Image Registration [[pdf]](https://arxiv.org/abs/2506.15596) [[code]](https://github.com/MICV-yonsei/M2M-Reg)

#### 2D–3D Registration
* [MICCAI 2020] Fluid Registration Between Lung CT and Stationary Chest Tomosynthesis Images [[pdf]](https://arxiv.org/abs/2203.04958) [[code]](https://github.com/uncbiag/2D3DFluidReg)
* [MICCAI 2020] Generalizing spatial transformers to projective geometry with applications to 2D/3D registration [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_32) [[code]](https://github.com/gaocong13/Projective-Spatial-Transformers)
* [MICCAI 2022] LiftReg: Limited Angle 2D/3D Deformable Registration [[pdf]](https://arxiv.org/abs/2203.05565) [[code]](https://github.com/uncbiag/LiftReg)
* [MICCAI 2023] X-ray to ct rigid registration using scene coordinate regression [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_74) [[code]](https://github.com/Pragyanstha/SCR-Registration)
* [MICCAI 2023] A patient-specific self-supervised model for automatic X-Ray/CT registration [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_49) [[code]](https://github.com/BaochangZhang/PSSS_registration)
* [CVPR 2024] Intraoperative 2D/3D Image Registration via Differentiable X-ray Rendering [[pdf]](https://arxiv.org/abs/2312.06358) [[code]](https://github.com/eigenvivek/DiffPose)
* [MedIA 2026] DVAP-Reg: Dual-view anatomical prior-driven cross-dimensional registration for spinal surgery navigation [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S1361841525004761) [[code]](https://github.com/TMMU-KLPOP/DVAP-Reg)
* [TMI 2026] Double-Decomposition Motion Tracking of Intraoperative 3D Structures via Cross-Spatio-Temporal Semantics Alignment [[pdf]](https://ieeexplore.ieee.org/document/11184624)
* [Nature 2026] Rapid patient-specific neural networks for intraoperative X-ray to volume registration [[pdf]](https://arxiv.org/pdf/2503.16309)

#### Longitudinal Registration
* [MedIA 2024] Longitudinally consistent registration and parcellation of cortical surfaces using semi-supervised learning [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S136184152400118X) [[code]](https://github.com/BRAIN-Lab-UNC/LongitudinalJointRegParc)

#### Histological / Microscopy Registration
* [TMI 2020] ANHIR: Automatic Non-rigid Histological Image Registration Challenge, IEEE Transactions on Medical Imaging [[pdf]](https://ieeexplore.ieee.org/document/9058666) [[code]](https://github.com/MWod/ANHIR_MW)
* [Nature Communications 2023] Virtual alignment of pathology image series for multi-gigapixel whole slide image [[pdf]](https://www.nature.com/articles/s41467-023-40218-9) [[code]](https://github.com/MathOnco/valis)
* [TMI 2024] Unsupervised Non-rigid Histological Image Registration Guided by Keypoint Correspondences Based on Learnable Deep Features with Iterative Training [[pdf]](https://ieeexplore.ieee.org/document/10643202) [[code]](https://github.com/weixy17/IKCG/tree/main/ACROBAT)
* [MedIA 2025] PViT-AIR: Puzzling vision transformer-based affine image registration for multi histopathology and faxitron images of breast tissue [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S1361841524002810) [[code]](https://github.com/pimed/PViT-AIR)

#### Pathology-aware / Missing-correspondence Registration
* [MICCAI 2022] Unsupervised Deformable Image Registration with Absent Correspondences in Pre-operative and Post-Recurrence Brain Tumor MRI Scans [[pdf]](https://arxiv.org/abs/2206.03900) [[code]](https://github.com/cwmok/DIRAC)
* [ICCV 2023] Preserving Tumor Volumes for Unsupervised Medical Image Registration [[pdf]](https://arxiv.org/abs/2309.10153) [[code]](https://github.com/dddraxxx/Medical-Reg-with-Volume-Preserving)
* [MICCAI 2024] Noise Removed Inconsistency Activation Map for Unsupervised Registration of Brain Tumor MRI between Pre-operative and Follow-up Phases [[pdf]](https://papers.miccai.org/miccai-2024/paper/1262_paper.pdf) [[code]](https://github.com/chongweiwu/NR-IAM)

#### Cortical Surface Registration
* [NeuroImage 2020] Cortical surface registration using unsupervised learning [[pdf]](https://arxiv.org/abs/2004.04617) [[code]](https://github.com/voxelmorph/spheremorph)
* [TMI 2021] S3Reg: Superfast Spherical Surface Registration Based on Deep Learning [[pdf]](https://ieeexplore.ieee.org/abstract/document/9389746/citations#citations) [[code]](https://github.com/BRAIN-Lab-UNC/S3Reg)
* [MICCAI 2022] A Deep-Discrete Learning Framework for Spherical Surface Registration [[pdf]](https://arxiv.org/abs/2203.12999) [[code]](https://github.com/mohamedasuliman/DDR)
* [MedIA 2024] SUGAR: Spherical ultrafast graph attention framework for cortical surface registration [[pdf]](https://www.sciencedirect.com/science/article/pii/S1361841524000471) [[code]](https://github.com/pBFSLab/SUGAR)
* [MedIA 2024] Longitudinally consistent registration and parcellation of cortical surfaces using semi-supervised learning [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S136184152400118X) [[code]](https://github.com/BRAIN-Lab-UNC/LongitudinalJointRegParc)
* [MedIA 2024] JOSA: Joint surface-based registration and atlas construction of brain geometry and function [[pdf]](https://arxiv.org/pdf/2311.08544) [[code]](https://voxelmorph.net)

### 6. Registration-Enabled Medical Image Analysis

#### Registration-guided Segmentation
* [CVPR 2019] Data Augmentation Using Learned Transformations for One-Shot Medical Image Segmentation [[pdf]](https://www.mit.edu/~adalca/files/papers/cvpr2019_brainstorm.pdf) [[code]](https://github.com/xamyzhao/brainstorm)
* [MICCAI 2019] DeepAtlas: Joint Semi-Supervised Learning of Image Registration and Segmentation [[pdf]](https://arxiv.org/abs/1904.08465) [[code]](https://github.com/uncbiag/DeepAtlas) 
* [MedIA 2022] Atlas-ISTN: joint segmentation, registration and atlas construction with image-and-spatial transformer networks [[pdf]](https://www.sciencedirect.com/science/article/pii/S1361841522000354) [[code]](https://github.com/biomedia-mira/atlas-istn)

#### Atlas & Template Construction
* [NeurIPS 2019] Learning conditional deformable templates with convolutional networks [[pdf]](https://proceedings.neurips.cc/paper/2019/hash/bbcbff5c1f1ded46c25d28119a85c6c2-Abstract.html) [[code]](https://github.com/voxelmorph/voxelmorph/blob/dev/scripts/tf/train_cond_template.py)
* [ICCV 2021] Generative Adversarial Registration for Improved Conditional Deformable Templates [[pdf]](https://arxiv.org/abs/2105.04349) [[code]](https://github.com/neel-dey/Atlas-GAN)
* [CVPR 2022] Aladdin: Joint Atlas Building and Diffeomorphic Registration Learning with Pairwise Alignment [[pdf]](https://arxiv.org/abs/2202.03563) [[code]](https://github.com/uncbiag/Aladdin)
* [CVPR 2022] Topology-preserving shape reconstruction and registration via neural diffeomorphic flow [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_Topology-Preserving_Shape_Reconstruction_and_Registration_via_Neural_Diffeomorphic_Flow_CVPR_2022_paper.pdf) [[code]](https://github.com/Siwensun/Neural_Diffeomorphic_Flow--NDF)
* [NeurIPS 2022] Geo-SIC: Learning Deformable Geometric Shapes in Deep Image Classifiers [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2022/file/b328c5bd9ff8e3a5e1be74baf4a7a456-Paper-Conference.pdf) [[code]](https://github.com/jw4hv/Geo-SIC)
* [CVPR 2025] MultiMorph: On-demand Atlas Construction [[pdf]](https://arxiv.org/pdf/2504.00247) [[code]](https://github.com/mabulnaga/multimorph)

#### Motion Estimation & Tracking
* [MICCAI 2018] Joint learning of motion estimation and segmentation for cardiac MR image sequences [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-00934-2_53) [[code]](https://github.com/cq615/Joint-Motion-Estimation-and-Segmentation)
* [CVPR 2021] DeepTag: An unsupervised deep learning method for motion tracking on cardiac tagging magnetic resonance images [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/html/Ye_DeepTag_An_Unsupervised_Deep_Learning_Method_for_Motion_Tracking_on_CVPR_2021_paper.html) [[code]](https://github.com/DeepTag/cardiac_tagging_motion_estimation)
* [TMI 2022] MulViMotion: Shape-aware 3D Myocardial Motion Tracking from Multi-View Cardiac MRI [[pdf]](https://ieeexplore.ieee.org/abstract/document/9721301/) [[code]](https://github.com/ImperialCollegeLondon/Multiview-Motion-Estimation-for-3D-cardiac-motion-tracking)
* [MedIA 2023] Generative myocardial motion tracking via latent space exploration with biomechanics-informed prior [[pdf]](https://www.sciencedirect.com/science/article/pii/S1361841522003103) [[code]](https://github.com/cq615/BIGM-motion-tracking)
* [MICCAI 2024] TLRN: Temporal Latent Residual Networks For Large Deformation Image Registration [[pdf]](https://arxiv.org/abs/2407.11219) [[code]](https://github.com/nellie689/TLRN)

#### Representation Learning
* [CVPR 2023] Geometric Visual Similarity Learning in 3D Medical Image Self-supervised Pre-training [[pdf]](https://arxiv.org/abs/2303.00874) [[code]](https://github.com/YutingHe-list/GVSL)
* [TPAMI 2025] Homeomorphism Prior for False Positive and Negative Problem in Medical Image Dense Contrastive Representation Learning [[pdf]](https://www.arxiv.org/abs/2502.05282) [[code]](https://github.com/YutingHe-list/GEMINI) 

#### Image-guided Intervention & Surgical Navigation
* [MICCAI 2023] X-ray to ct rigid registration using scene coordinate regression [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_74) [[code]](https://github.com/Pragyanstha/SCR-Registration)
* [MICCAI 2023] A patient-specific self-supervised model for automatic X-Ray/CT registration [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_49) [[code]](https://github.com/BaochangZhang/PSSS_registration)
* [CVPR 2024] Intraoperative 2D/3D Image Registration via Differentiable X-ray Rendering [[pdf]](https://arxiv.org/abs/2312.06358) [[code]](https://github.com/eigenvivek/DiffPose)
* [MedIA 2026] DVAP-Reg: Dual-view anatomical prior-driven cross-dimensional registration for spinal surgery navigation [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S1361841525004761) [[code]](https://github.com/TMMU-KLPOP/DVAP-Reg)
* [TMI 2026] Double-Decomposition Motion Tracking of Intraoperative 3D Structures via Cross-Spatio-Temporal Semantics Alignment [[pdf]](https://ieeexplore.ieee.org/document/11184624)
* [Nature 2026] Rapid patient-specific neural networks for intraoperative X-ray to volume registration [[pdf]](https://arxiv.org/pdf/2503.16309)

## 7. Datasets & Challenges

### Datasets

This section collects publicly available datasets commonly used in medical image registration. The datasets are grouped by imaging modality and will be continuously expanded.

> Some datasets require account registration, acceptance of data-use agreements, or challenge participation. The number of images, generated registration pairs, dataset splits, and preprocessing protocols may vary across studies. Please refer to the corresponding paper and official implementation for the exact experimental settings.

### CT Datasets

| Dataset | Anatomy | Typical registration setting | Scale and notes | Access |
|---|---|---|---|---|
| Learn2Reg Abdomen CT–CT | Abdomen | Inter-subject registration | Contains 30 abdominal CT scans with annotations of 13 anatomical structures. | [Learn2Reg Datasets](https://learn2reg.grand-challenge.org/Datasets/) |
| Abdominal DIR-QA | Abdomen | Intra-subject longitudinal CT registration | Contains 30 abdominal CT image pairs with corresponding vessel-bifurcation landmarks. | [Zenodo](https://zenodo.org/records/14362785) · [Usage Instructions](https://github.com/deshanyang/Abdominal-DIR-QA) |
| Learn2Reg Lung CT | Lung / thorax | Intra-subject inspiration–expiration registration | Contains paired inspiration and expiration lung CT scans, including 30 pairs. | [Learn2Reg Datasets](https://learn2reg.grand-challenge.org/Datasets/) |
| Medical Segmentation Decathlon – Liver | Liver | Inter-subject liver CT registration | Task03 contains portal-venous-phase CT volumes with liver and tumor annotations. | [Official Website](https://medicaldecathlon.com/) · [AWS Open Data](https://registry.opendata.aws/msd/) |
| SLIVER07 | Liver | Inter-subject liver CT registration | Contains contrast-enhanced abdominal CT scans with liver segmentation annotations. | [SLIVER07](https://sliver07.grand-challenge.org/) · [Zenodo](https://zenodo.org/records/2597575) |
| BFH Liver CT | Liver | Inter-subject liver CT registration | Contains liver CT volumes collected at Beijing Friendship Hospital. | [Preprocessed Dataset](https://github.com/microsoft/Recursive-Cascaded-Networks#datasets) |
| LSPIG | Liver | Intra-subject liver CT registration | Contains paired pig liver CT scans with liver segmentation labels. | [Preprocessed Dataset](https://github.com/microsoft/Recursive-Cascaded-Networks#datasets) |
| SegRap2023 | Head and neck | Inter-subject CT registration | Contains paired non-contrast and contrast-enhanced CT scans from 200 NPC patients, with annotations of organs and tumors. | [SegRap2023 Dataset](https://segrap2023.grand-challenge.org/dataset/) |

### MR Datasets

| Dataset | Anatomy | Typical registration setting | Scale and notes | Access |
|---|---|---|---|---|
| OASIS | Brain | Atlas-based and inter-subject T1-weighted MR registration | OASIS-1 contains 416 subjects. Anatomical segmentation maps and preprocessed versions are commonly used in registration studies. | [OASIS-1](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md) · [Neurite-preprocessed Data](https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/) |
| LUMIR | Brain | Large-scale inter-subject T1-weighted MR registration | Provides a large collection of preprocessed T1-weighted brain MR volumes for deformable registration. | [Learn2Reg 2024 – LUMIR](https://learn2reg.grand-challenge.org/learn2reg-2024/) |
| LPBA40 | Brain | Inter-subject T1-weighted MR registration | Contains 40 subjects with manually annotated anatomical structures. | [LONI Atlas Downloads](https://www.loni.usc.edu/research/atlas_downloads) |
| Mindboggle101 | Brain | Inter-subject T1-weighted MR registration | Contains 101 manually labeled brain MR scans collected from multiple public datasets. | [Mindboggle101 Data](https://mindboggle.info/data) |
| HCP Young Adult | Brain | Atlas-based and inter-subject structural MR registration | A large healthy-adult neuroimaging dataset containing structural MR scans. | [HCP Young Adult Data](https://www.humanconnectome.org/study/hcp-young-adult/data-releases) |
| ACDC | Cardiac | Intra-subject end-diastolic–end-systolic registration | Contains cardiac cine-MR examinations with annotations. | [ACDC Database](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) |
| Osteoarthritis Initiative | Knee | Inter-subject and longitudinal knee MR registration | A large longitudinal knee osteoarthritis cohort containing knee MRI and associated annotations. | [OAI Data Access](https://nda.nih.gov/oai) |

### Multi-modal Datasets

| Dataset | Anatomy | Typical registration setting | Scale and notes | Access |
|---|---|---|---|---|
| Learn2Reg Abdomen MR–CT | Abdomen | Intra-subject deformable MR–CT registration | Contains paired abdominal MR and CT scans with abdominal organ annotations for multimodal registration evaluation, together with additional unpaired scans. | [Learn2Reg Datasets](https://learn2reg.grand-challenge.org/Datasets/) |
| BraTS 2018 | Brain | Cross-contrast T1–T2 MR registration | Provides multimodal brain tumor MR scans, including T1, contrast-enhanced T1, T2, and FLAIR sequences. | [BraTS 2018 Data](https://www.med.upenn.edu/sbia/brats2018/data.html) |
| iSeg-2019 | Infant brain | Cross-contrast T1–T2 MR registration | Provides multi-site T1- and T2-weighted MR scans of 6-month-old infants with white matter, gray matter, and cerebrospinal fluid annotations. | [iSeg-2019 Data](https://iseg2019.web.unc.edu/data/) |

### Challenges
- [Learn2Reg 2025](https://learn2reg.grand-challenge.org/learn2reg-2025/)
 - [Oncoreg](https://learn2reg.grand-challenge.org/oncoreg/)
 - [Learn2Reg 2024](https://learn2reg.grand-challenge.org/learn2reg-2024/)
 - [Learn2Reg 2023](https://learn2reg.grand-challenge.org/learn2reg-2023/)
 - [Learn2Reg 2022](https://learn2reg.grand-challenge.org/learn2reg-2022/)
 - [Learn2Reg 2021](https://learn2reg.grand-challenge.org/Learn2Reg2021/)
 - [Learn2Reg 2020](https://learn2reg.grand-challenge.org/Learn2Reg2020/)
 - [AutomatiC Registration Of Breast cAncer Tissue (ACROBAT)](https://acrobat.grand-challenge.org/)
 - [Automatic Non-rigid Histological Image Registration (ANHIR)](https://anhir.grand-challenge.org/)
 - [Robust Non-rigid Registration Challenge for Expansion Microscopy (RnR-ExM)](https://rnr-exm.grand-challenge.org/)
 - [Correction of Brain shift with Intra-Operative Ultrasound (CuRIOUS 2019)](https://curious2019.grand-challenge.org/)
 - [Correction of Brain shift with Intra-Operative Ultrasound (CuRIOUS 2018)](https://curious2018.grand-challenge.org/)

## 8. Software
- [ANTs](https://manpages.ubuntu.com/manpages/trusty/man1/ANTS.1.html)
 - [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg)
 - [LDDMM](https://github.com/brianlee324/torch-lddmm)
 - [DeedsBCV](https://github.com/mattiaspaul/deedsBCV)
 - [Elastix](https://github.com/SuperElastix/elastix)
 - [FireANTs (GPU)](https://github.com/rohitrango/fireants)
