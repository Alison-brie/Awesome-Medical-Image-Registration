
# Awesome Medical Image Registration [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/Alison-brie/Awesome-ImageRegistration/)

:wave: Hi! This repo is a collection of AWESOME things about :star2:**Medical Image Registration**:star2:, including useful materials, papers, code. Feel free to star and fork.

TODO:
- [ ] Add a list people in medical image registration listed with their academic genealogy
- [ ] Add a list organs in medical image registration



## Contributing
Please feel free to send me [pull requests](https://github.com/Alison-brie/Awesome-ImageRegistration/pulls) or email (alisonbrielee@gmail.com) to add links.

## Table of Contents

 - [Related Awesome Lists](#related-awesome-lists)
 - [Papers](#papers)
 - [Chanllenges](#chanllenges)
 - [Software](#software)


## Related Awesome Lists
 - [Awesome Optical Flow](https://github.com/hzwer/Awesome-Optical-Flow)
 - [Awesome Image-to-Image Translation](https://github.com/weihaox/awesome-image-translation)
 - [Awesome Medical Imaging](https://github.com/fepegar/awesome-medical-imaging)
 - [Awesome Machine Learning in Biomedical(Healthcare) Imaging](https://github.com/XindiWu/Awesome-Machine-Learning-in-Biomedical-Healthcare-Imaging)
 
 

## Papers

### Table of Contents
 - [Original CNN model for Registration](#original-cnn-model-for-registration)
 - [Deep Diffeomorphic Registration](#deep-diffeomorphic-registration)
 - [Pyramid CNN for Registration](#pyramid-cnn-for-registration)
 - [Transformer for Registration](#transformer-for-registration)
 - [Hyperparameter Learning in Registration](#hyperparameter-learning-in-registration)
 - [Discrete Registration and Instance Optimization](#discrete-registration-and-instance-optimization)
 - [Joint Affine and Deformable Registration](#joint-affine-and-deformable-registration)
 - [Multi-Modality Registration](#multi-modality-registration)
 - [2D-3D Registration](#2d-3d-registration)
 - [Histological Image Registration](#histological-image-registration)
 - [Applications of image registration](#applications-of-image-registration)



#### Original CNN model for Registration
* [MICCAI Workshop 2017] End-to-end unsupervised deformable image registration with a convolutional neural network [[pdf]](https://arxiv.org/pdf/1704.06065)
* [CVPR 2018] An Unsupervised Learning Model for Deformable Medical Image Registration [[pdf]](https://arxiv.org/abs/1802.02604) [[code]](https://github.com/voxelmorph/voxelmorph)
* [Arxiv 2018] Inverse-Consistent Deep Networks for Unsupervised Deformable Image Registration [[pdf]](https://arxiv.org/pdf/1809.03443) [[code]](https://github.com/zhangjun001/ICNet)
* [MICCAI 2019] Unsupervised Deformable Image Registration Using Cycle-Consistent CNN [[pdf]](https://arxiv.org/pdf/1809.03443) [[code]](https://github.com/zhangjun001/ICNet)
* [TMI 2019] VoxelMorph: A Learning Framework for Deformable Medical Image Registration [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-32226-7_19) [[code]](https://github.com/boahK/MEDIA_CycleMorph)


#### Deep Diffeomorphic Registration 
* [MICCAI 2018] Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration [[pdf]](https://arxiv.org/abs/1805.04605) [[code]](https://github.com/voxelmorph/voxelmorph)
* [MedIA 2019] Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces [[pdf]](https://arxiv.org/abs/1903.03545) [[code]](https://github.com/voxelmorph/voxelmorph) 
* [NeurIPS 2019] Region-specific Diffeomorphic Metric Mapping [[pdf]](https://drive.google.com/file/d/1kIuunw6FP2ek8ZsLw92zL6RJw02YU7Nx) [[code]](https://github.com/uncbiag/easyreg)
* [CVPR 2019] Metric Learning for Image Registration [[pdf]](https://drive.google.com/file/d/1kIuunw6FP2ek8ZsLw92zL6RJw02YU7Nx) [[code]](https://github.com/uncbiag/mermaid)
* [CVPR 2020] DeepFLASH: An Efficient Network for Learning-based Medical Image Registration [[pdf]](https://arxiv.org/pdf/2004.02097)  [[code]](https://github.com/jw4hv/deepflash)
* [CVPR 2020] Fast Symmetric Diffeomorphic Image Registration with Convolutional Neural Networks [[pdf]](https://arxiv.org/abs/2003.09514) [[code]](https://github.com/cwmok/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks)
* [ICCV 2023] ICON: Learning Regular Maps Through Inverse Consistency [[pdf]](https://arxiv.org/pdf/2105.04459) [[code]](https://github.com/uncbiag/ICON)
* [CVPR 2023] GradICON: Approximate Diffeomorphisms via Gradient Inverse Consistency [[pdf]](https://drive.google.com/file/d/1j8u5n50knQUxhnHp1OMGEwsl8CX-lODX) [[code]](https://github.com/uncbiag/ICON)
* [MedIA 2019] R2Net: Efficient and flexible diffeomorphic image registration using Lipschitz continuous residual networks [[pdf]](https://www.sciencedirect.com/science/article/pii/S1361841523001779) [[code]](https://github.com/ankitajoshi15/R2Net)
* [ECCV 2024] NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration [[pdf]](https://arxiv.org/abs/2309.07322) [[code]](https://github.com/uncbiag/NePhi)



#### Pyramid CNN for Registration
* [ICCV 2019] Recursive Cascaded Networks for Unsupervised Medical Image Registration [[pdf]](https://arxiv.org/abs/1907.12353) [[code]](https://github.com/microsoft/Recursive-Cascaded-Networks)
* [MICCAI 2019] Dual-Stream Pyramid Registration Network [[pdf]](https://arxiv.org/pdf/1909.11966) [[code]](https://github.com/kangmiao15/Dual-Stream-PRNet-Plus)
* [MICCAI 2020] Large Deformation Diffeomorphic Image Registration with Laplacian Pyramid Networks [[pdf]](https://arxiv.org/abs/2006.16148) [[code]](https://github.com/cwmok/LapIRN)
* [TMI 2021] Learning a model-driven variational network for deformable image registration [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9525092) [[code]](https://github.com/xi-jia/Learning-a-Model-Driven-Variational-Network-for-Deformable-Image-Registration)
* [MICCAI 2022] NICE-Net: a Non-Iterative Coarse-to-finE registration Network for deformable image registration [[pdf]](https://arxiv.org/abs/2206.12596) [[code]](https://github.com/MungoMeng/Registration-NICE-Net)
* [TMI 2023] Self-Distilled Hierarchical Network for Unsupervised Deformable Image Registration [[pdf]](https://ieeexplore.ieee.org/abstract/document/10042453) [[code]](https://github.com/Blcony/SDHNet)
* [CVPR 2024] IIRP-Net: Iterative Inference Residual Pyramid Network for Enhanced Image Registration [[pdf]](https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_IIRP-Net_Iterative_Inference_Residual_Pyramid_Network_for_Enhanced_Image_Registration_CVPR_2024_paper.pdf) [[code]](https://github.com/Torbjorn1997/IIRP-Net)



#### Transformer for Registration
* [MedIA 2022] TransMorph: Transformer for unsupervised medical image registration [[pdf]](https://arxiv.org/abs/2111.10480) [[code]](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)
* [MICCAI 2022] Swin-VoxelMorph: A Symmetric Unsupervised Learning Model for Deformable Medical Image Registration Using Swin Transformer [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_8) [[code]](https://github.com/CJSOrange/DMR-Deformer)
* [MICCAI 2022] Deformer: Towards displacement field learning for unsupervised medical image registration [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_14) [[code]](https://github.com/YongpeiZhu/Swin-VoxelMorph)
* [MICCAI 2022] XMorpher: Full Transformer for Deformable Medical Image Registration via Cross Attention [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_21) [[code]](https://github.com/Solemoon/XMorpher)
* [CVPR 2022] Affine Medical Image Registration with Coarse-to-Fine Vision Transformer [[pdf]](https://arxiv.org/abs/2203.15216) [[code]](https://github.com/cwmok/C2FViT)
* [MICCAI 2023] PIViT: Large Deformation Image Registration with Pyramid-Iterative Vision Transformer [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_57) [[code]](https://github.com/Torbjorn1997/PIViT)
* [TMI 2023] TransMatch: a transformer-based multilevel dual-stream feature matching network for unsupervised deformable image registration [[pdf]](https://ieeexplore.ieee.org/abstract/document/10158729) [[code]](https://github.com/tzayuan/TransMatch_TMI)
* [CVPR 2024] Correlation-aware Coarse-to-fine MLPs for Deformable Medical Image Registration [[pdf]](https://openaccess.thecvf.com/content/CVPR2024/papers/Meng_Correlation-aware_Coarse-to-fine_MLPs_for_Deformable_Medical_Image_Registration_CVPR_2024_paper.pdf) [[code]](https://github.com/MungoMeng/Registration-CorrMLP)



#### Hyperparameter learning in Registration 
* [IPMI 2021] HyperMorph: Amortized Hyperparameter Learning for Image Registration [[pdf]](https://arxiv.org/abs/2101.01035) [[code]](https://ahoopes.github.io/hypermorph/)
* [MICCAI 2021] Conditional Deformable Image Registration with Convolutional Neural Network [[pdf]](https://arxiv.org/abs/2106.12673) [[code]](https://github.com/cwmok/Conditional_LapIRN)
* [TPAMI 2021] Learning Deformable Image Registration from Optimization: Perspective, Modules, Bilevel Training and Beyond [[pdf]](https://arxiv.org/abs/2004.14557) [[code]](https://github.com/Alison-brie/MultiPropReg)
* [MedIA 2023] Hyper-Convolutions via Implicit Kernels for Medical Image Analysis [[pdf]](https://arxiv.org/abs/2202.02701) [[code]](https://github.com/tym002/Hyper-Convolution)
* [TIP 2023] Automated learning for deformable medical image registration by jointly optimizing network architectures and objective functions [[pdf]](https://arxiv.org/abs/2203.06810) [[code]](https://github.com/Alison-brie/AutoReg)





#### Discrete Registration and Instance Optimization
* [MICCAI 2019] Closing the Gap between Deep and Conventional Image Registration using Probabilistic Dense Displacement Networks [[pdf]](https://arxiv.org/abs/1907.10931) [[code]](https://github.com/multimodallearning/pdd_net)
* [MICCAI 2020] Highly accurate and memory efficient unsupervised learning-based discrete CT registration using 2.5 D displacement search [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_19) [[code]](https://github.com/multimodallearning/pdd2.5/)
* [MICCAI Workshop 2021] Fast 3D registration with accurate optimisation and little learning for Learn2Reg 2021 [[pdf]](https://arxiv.org/abs/2112.03053) [[code]](https://github.com/multimodallearning/convexAdam)
* [MICCAI Workshop 2022] Voxelmorph++ going beyond the cranial vault with keypoint supervision and multi-channel instance optimisation [[pdf]](https://openreview.net/pdf?id=SrlgSXA3qAY) [[code]](https://github.com/mattiaspaul/VoxelMorphPlusPlus)
* [MICCAI 2023] SAMConvex: Fast Discrete Optimization for CT Registration using Self-supervised Anatomical Embedding and Correlation Pyramid [[pdf]](https://arxiv.org/abs/2307.09727) [[code]](https://github.com/alibaba-damo-academy/samconvex) 
* [MICCAI Workshop 2024] Deformable Medical Image Registration Under Distribution Shifts with Neural Instance Optimization [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-45673-2_13)
* [TMI 2024] ConvexAdam: Self-Configuring Dual-Optimisation-Based 3D Multitask Medical Image Registration [[pdf]](https://ieeexplore.ieee.org/abstract/document/10681158) [[code]](https://github.com/multimodallearning/convexAdam)




#### Joint Affine and Deformable Registration
* [CVPR 2019] Networks for Joint Affine and Non-Parametric Image Registration [[pdf]](https://arxiv.org/pdf/1903.08811.pdf) [[code]](https://github.com/uncbiag/easyreg)
* [JBHI 2020] Unsupervised 3D End-to-End Medical Image Registration with Volume Tweening Network [[pdf]](https://arxiv.org/pdf/1902.05020) [[code]](https://github.com/microsoft/Recursive-Cascaded-Networks)
* [MICCAI 2021] SAME: Deformable Image Registration based on Self-supervised Anatomical Embeddings [[pdf]](https://arxiv.org/abs/2109.11572) [[code]](https://github.com/alibaba-damo-academy/same)
* [MICCAI 2023] Non-iterative Coarse-to-Fine Transformer Networks for Joint Affine and Deformable Image Registration [[pdf]](https://arxiv.org/abs/2307.03421) [[code]](https://github.com/MungoMeng/Registration-NICE-Trans)
* [Arxiv 2023] SAME++: Deformable Image Registration based on Self-supervised Anatomical Embeddings [[pdf]](https://doi.org/10.48550/arXiv.2311.14986) [[code]](https://github.com/alibaba-damo-academy/same)



#### Multi-Modality Registration
* [IPMI 2019] Unsupervised deformable registration for multi-modal images via disentangled representations [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-20351-1_19)
* [MICCAI 2019] Synthesis and inpainting-based MR-CT registration for image-guided thermal ablation of liver tumors [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-32254-0_57)
* [MICCAI 2020] Adversarial uni-and multi-modal stream networks for multimodal image registration [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_22)
* [NeurIPS 2020] CoMIR: Contrastive multimodal image representation for registration [[pdf]](https://proceedings.neurips.cc/paper/2020/hash/d6428eecbe0f7dff83fc607c5044b2b9-Abstract.html) [[code]](https://github.com/MIDA-group/CoMIR)
* [TPMAI 2021] SymReg-GAN: symmetric image registration with generative adversarial networks [[pdf]](https://ieeexplore.ieee.org/abstract/document/9440692)
* [MedIA 2022] Deformable MR-CT image registration using an unsupervised, dual-channel network for neurosurgical guidance [[pdf]](https://www.sciencedirect.com/science/article/pii/S1361841521003376)
* [TMI 2022] SynthMorph: learning contrast-invariant registration without acquired images [[pdf]](https://arxiv.org/abs/2004.10282)  [[code]](https://martinos.org/malte/synthmorph/)
* [MICCAI 2022] ContraReg: Contrastive Learning of Multi-modality Unsupervised Deformable Image Registration [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_7) [[code]](https://github.com/jmtzt/ContraReg)
* [MICCAI 2023] DISA: DIfferentiable Similarity Approximation for Universal Multimodal Registration [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_72) [[code]](https://github.com/ImFusionGmbH/DISA-universal-multimodal-registration)
* [CVPR 2024] Modality-Agnostic Structural Image Representation Learning for Deformable Multi-Modality Medical Image Registration [[pdf]](https://arxiv.org/abs/2402.18933) 



#### 2D-3D Registration
* [MICCAI 2020] Fluid Registration Between Lung CT and Stationary Chest Tomosynthesis Images [[pdf]](https://arxiv.org/abs/2203.04958) [[code]](https://github.com/uncbiag/2D3DFluidReg)
* [MICCAI 2020] Generalizing spatial transformers to projective geometry with applications to 2D/3D registration [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_32) [[code]](https://github.com/gaocong13/Projective-Spatial-Transformers)
* [MICCAI 2022] LiftReg: Limited Angle 2D/3D Deformable Registration [[pdf]](https://arxiv.org/abs/2203.05565) [[code]](https://github.com/uncbiag/LiftReg)
* [MICCAI 2023] X-ray to ct rigid registration using scene coordinate regression [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_74) [[code]](https://github.com/Pragyanstha/SCR-Registration)
* [MICCAI 2023] A patient-specific self-supervised model for automatic X-Ray/CT registration [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_49) [[code]](https://github.com/BaochangZhang/PSSS_registration)
* [CVPR 2024] Intraoperative 2D/3D Image Registration via Differentiable X-ray Rendering [[pdf]](https://arxiv.org/abs/2312.06358) [[code]](https://github.com/eigenvivek/DiffPose)


#### Histological Image Registration
* [TMI 2022] Unsupervised Histological Image Registration Using Structural Feature Guided Convolutional Neural Network [[pdf]](https://ieeexplore.ieee.org/document/9745959)
* [Nature Communications 2023] Virtual alignment of pathology image series for multi-gigapixel whole slide image [[pdf]](https://www.nature.com/articles/s41467-023-40218-9) [[code]](https://github.com/MathOnco/valis)
* [TMI 2024] Unsupervised Non-rigid Histological Image Registration Guided by Keypoint Correspondences Based on Learnable Deep Features with Iterative Training [[pdf]](https://ieeexplore.ieee.org/document/10643202) [[code]](https://github.com/weixy17/IKCG/tree/main/ACROBAT)


#### Applications of image registration
* [CVPR 2019] Data Augmentation Using Learned Transformations for One-Shot Medical Image Segmentation [[pdf]](https://www.mit.edu/~adalca/files/papers/cvpr2019_brainstorm.pdf) [[code]](https://github.com/xamyzhao/brainstorm)
* [MICCAI 2019] DeepAtlas: Joint Semi-Supervised Learning of Image Registration and Segmentation [[pdf]](https://arxiv.org/abs/1904.08465) [[code]](https://github.com/uncbiag/DeepAtlas) 
* [MICCAI 2019] Adversarial optimization for joint registration and segmentation in prostate CT radiotherapy [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-32226-7_41)
* [MedIA 2019] Adversarial learning for mono-or multi-modal registration [[pdf]](https://www.sciencedirect.com/science/article/pii/S1361841519300805)
* [ICCV 2021] Generative Adversarial Registration for Improved Conditional Deformable Templates [[pdf]](https://arxiv.org/abs/2105.04349) [[code]](https://github.com/neel-dey/Atlas-GAN)
* [CVPR 2022] Aladdin: Joint Atlas Building and Diffeomorphic Registration Learning with Pairwise Alignment [[pdf]](https://arxiv.org/abs/2202.03563) [[code]](https://github.com/uncbiag/Aladdin)
* [CVPR 2022] Topology-preserving shape reconstruction and registration via neural diffeomorphic flow [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_Topology-Preserving_Shape_Reconstruction_and_Registration_via_Neural_Diffeomorphic_Flow_CVPR_2022_paper.pdf) [[code]](https://github.com/Siwensun/Neural_Diffeomorphic_Flow--NDF)
* [MICCAI 2022] Unsupervised Deformable Image Registration with Absent Correspondences in Pre-operative and Post-Recurrence Brain Tumor MRI Scans [[pdf]](https://arxiv.org/abs/2206.03900) [[code]](https://github.com/cwmok/DIRAC)
* [MICCAI 2023] Implicit neural representations for joint decomposition and registration of gene expression images in the marmoset brain [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_61) [[code]](https://github.com/BrainImageAnalysis/ImpRegDec)
* [ICCV 2023] Preserving Tumor Volumes for Unsupervised Medical Image Registration [[pdf]](https://arxiv.org/abs/2309.10153) [[code]](https://github.com/dddraxxx/Medical-Reg-with-Volume-Preserving)
* [MedIA 2023] Anatomically constrained and attention-guided deep feature fusion for joint segmentation and deformable medical image registration [[pdf]](https://www.sciencedirect.com/science/article/pii/S1361841523000725)




## Chanllenges
 - [Learn2Reg 2020](https://learn2reg.grand-challenge.org/Learn2Reg2020/)
 - [Learn2Reg 2021](https://learn2reg.grand-challenge.org/Learn2Reg2021/)
 - [Learn2Reg 2022](https://learn2reg.grand-challenge.org/Learn2Reg2022/)
 - [Learn2Reg 2023](https://learn2reg.grand-challenge.org/Learn2Reg2023/)
 - [Oncoreg](https://learn2reg.grand-challenge.org/oncoreg/)
 - [AutomatiC Registration Of Breast cAncer Tissue (ACROBAT)](https://acrobat.grand-challenge.org/)
 - [Automatic Non-rigid Histological Image Registration (ANHIR)](https://anhir.grand-challenge.org/)
 - [Robust Non-rigid Registration Challenge for Expansion Microscopy (RnR-ExM)](https://rnr-exm.grand-challenge.org/)


## Software
 - [ANTs](https://manpages.ubuntu.com/manpages/trusty/man1/ANTS.1.html)
 - [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg)
 - [LDDMM](https://github.com/brianlee324/torch-lddmm)
 - [DeedsBCV](https://github.com/mattiaspaul/deedsBCV)
 - [Elastix](https://github.com/SuperElastix/elastix)
