
# Awesome Medical Image Registration [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/Alison-brie/Awesome-ImageRegistration/)

:wave: Hi! This repo is a collection of AWESOME things about :star2:**Medical Image Registration**:star2:, including useful materials, papers, code. Feel free to star and fork.

TODO:
- [ ] Add a list organs in medical image registration




## Contributing
:running: **We will keep updating it.** :running:    

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
 - [Original and Fancier Network Architecture](#original-and-fancier-network-architecture)
    - [Original CNN model for Registration](#original-cnn-model-for-registration)
    - [Pyramid CNN for Registration](#pyramid-cnn-for-registration)
    - [Transformer for Registration](#transformer-for-registration)
 - [Deep Diffeomorphic Registration](#deep-diffeomorphic-registration)
 - [Joint Affine and Deformable Registration](#joint-affine-and-deformable-registration)
 - [Hyperparameter Learning in Registration](#hyperparameter-learning-in-registration)
 - [Discrete Registration and Instance Optimization](#discrete-registration-and-instance-optimization)
 - [Registration on Specific Task](#registration-on-specific-task)
   - [Multi-Modality Registration](#multi-modality-registration)
   - [2D-3D Registration](#2d-3d-registration)
   - [Histological Image Registration](#histological-image-registration)
   - [Registration with Tumor in Image](#registration-with-tumor-in-image)
 - [Fundation Model for Registration](#fundation-model-for-registration)
 - [Registration Related/Based Application](#registration-relatedbased-application)
    - [Segmentation](#segmentation)
    - [Template Construction](#template-construction)
    - [Cardiac Motion Estimation](#cardiac-motion-estimation)

    


### Original and Fancier Network Architecture
#### Original CNN model for Registration
* [MICCAI Workshop 2017] End-to-end unsupervised deformable image registration with a convolutional neural network [[pdf]](https://arxiv.org/pdf/1704.06065)
* [CVPR 2018] An Unsupervised Learning Model for Deformable Medical Image Registration [[pdf]](https://arxiv.org/abs/1802.02604) [[code]](https://github.com/voxelmorph/voxelmorph)
* [Arxiv 2018] Inverse-Consistent Deep Networks for Unsupervised Deformable Image Registration [[pdf]](https://arxiv.org/pdf/1809.03443) [[code]](https://github.com/zhangjun001/ICNet)
* [MICCAI 2019] Unsupervised Deformable Image Registration Using Cycle-Consistent CNN [[pdf]](https://arxiv.org/abs/1907.01319) [[code]](https://github.com/boahK/MEDIA_CycleMorph)
* [TMI 2019] VoxelMorph: A Learning Framework for Deformable Medical Image Registration [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-32226-7_19) [[code]](https://github.com/boahK/MEDIA_CycleMorph)

#### Pyramid and Iterative CNN for Registration
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
* [MICCAI Workshop 2022] Coordinate Translator for Learning Deformable Medical Image Registration [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-18814-5_10) [[code]](https://github.com/ZAX130/SmileCode)
* [MICCAI 2023] PIViT: Large Deformation Image Registration with Pyramid-Iterative Vision Transformer [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_57) [[code]](https://github.com/Torbjorn1997/PIViT)
* [MICCAI 2023] ModeT: Learning Deformable Image Registration via Motion Decomposition Transformer [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_70) [[code]](https://github.com/ZAX130/SmileCode)
* [TMI 2023] TransMatch: a transformer-based multilevel dual-stream feature matching network for unsupervised deformable image registration [[pdf]](https://ieeexplore.ieee.org/abstract/document/10158729) [[code]](https://github.com/tzayuan/TransMatch_TMI)
* [MICCAI Workshop 2023] Deformable Cross-Attention Transformer for Medical Image Registration [[pdf]](https://link.springer.com/chapter/10.1007/978-3-031-45673-2_12) [[code]](https://github.com/junyuchen245/TransMorph_DCA)
* [CVPR 2024] Correlation-aware Coarse-to-fine MLPs for Deformable Medical Image Registration [[pdf]](https://openaccess.thecvf.com/content/CVPR2024/papers/Meng_Correlation-aware_Coarse-to-fine_MLPs_for_Deformable_Medical_Image_Registration_CVPR_2024_paper.pdf) [[code]](https://github.com/MungoMeng/Registration-CorrMLP)




### Deep Diffeomorphic Registration 
* [MICCAI 2018] Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration [[pdf]](https://arxiv.org/abs/1805.04605) [[code]](https://github.com/voxelmorph/voxelmorph)
* [TMI 2019] Learning a Probabilistic Model for Diffeomorphic Registration [[pdf]](https://arxiv.org/pdf/1812.07460)
* [MedIA 2019] Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces [[pdf]](https://arxiv.org/abs/1903.03545) [[code]](https://github.com/voxelmorph/voxelmorph) 
* [NeurIPS 2019] Region-specific Diffeomorphic Metric Mapping [[pdf]](https://drive.google.com/file/d/1kIuunw6FP2ek8ZsLw92zL6RJw02YU7Nx) [[code]](https://github.com/uncbiag/easyreg)
* [CVPR 2019] Metric Learning for Image Registration [[pdf]](https://drive.google.com/file/d/1kIuunw6FP2ek8ZsLw92zL6RJw02YU7Nx) [[code]](https://github.com/uncbiag/mermaid)
* [CVPR 2020] DeepFLASH: An Efficient Network for Learning-based Medical Image Registration [[pdf]](https://arxiv.org/pdf/2004.02097)  [[code]](https://github.com/jw4hv/deepflash)
* [CVPR 2020] Fast Symmetric Diffeomorphic Image Registration with Convolutional Neural Networks [[pdf]](https://arxiv.org/abs/2003.09514) [[code]](https://github.com/cwmok/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks)
* [ICCV 2023] ICON: Learning Regular Maps Through Inverse Consistency [[pdf]](https://arxiv.org/pdf/2105.04459) [[code]](https://github.com/uncbiag/ICON)
* [CVPR 2023] GradICON: Approximate Diffeomorphisms via Gradient Inverse Consistency [[pdf]](https://drive.google.com/file/d/1j8u5n50knQUxhnHp1OMGEwsl8CX-lODX) [[code]](https://github.com/uncbiag/ICON)
* [MedIA 2019] R2Net: Efficient and flexible diffeomorphic image registration using Lipschitz continuous residual networks [[pdf]](https://www.sciencedirect.com/science/article/pii/S1361841523001779) [[code]](https://github.com/ankitajoshi15/R2Net)
* [ECCV 2024] NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration [[pdf]](https://arxiv.org/abs/2309.07322) [[code]](https://github.com/uncbiag/NePhi)
* [ArXiv 2024] SITReg: Multi-resolution architecture for symmetric, inverse consistent, and topology preserving image registration [[pdf]](https://arxiv.org/abs/2303.10211) [[code]](https://github.com/honkamj/SITReg?tab=readme-ov-file)




### Joint Affine and Deformable Registration
* [CVPR 2019] Networks for Joint Affine and Non-Parametric Image Registration [[pdf]](https://arxiv.org/pdf/1903.08811.pdf) [[code]](https://github.com/uncbiag/easyreg)
* [JBHI 2020] Unsupervised 3D End-to-End Medical Image Registration with Volume Tweening Network [[pdf]](https://arxiv.org/pdf/1902.05020) [[code]](https://github.com/microsoft/Recursive-Cascaded-Networks)
* [MICCAI 2021] SAME: Deformable Image Registration based on Self-supervised Anatomical Embeddings [[pdf]](https://arxiv.org/abs/2109.11572) [[code]](https://github.com/alibaba-damo-academy/same)
* [MICCAI 2023] Non-iterative Coarse-to-Fine Transformer Networks for Joint Affine and Deformable Image Registration [[pdf]](https://arxiv.org/abs/2307.03421) [[code]](https://github.com/MungoMeng/Registration-NICE-Trans)
* [Arxiv 2023] SAME++: Deformable Image Registration based on Self-supervised Anatomical Embeddings [[pdf]](https://doi.org/10.48550/arXiv.2311.14986) [[code]](https://github.com/alibaba-damo-academy/same)



### Hyperparameter learning in Registration 
* [IPMI 2021] HyperMorph: Amortized Hyperparameter Learning for Image Registration [[pdf]](https://arxiv.org/abs/2101.01035) [[code]](https://ahoopes.github.io/hypermorph/)
* [MICCAI 2021] Conditional Deformable Image Registration with Convolutional Neural Network [[pdf]](https://arxiv.org/abs/2106.12673) [[code]](https://github.com/cwmok/Conditional_LapIRN)
* [TPAMI 2021] Learning Deformable Image Registration from Optimization: Perspective, Modules, Bilevel Training and Beyond [[pdf]](https://arxiv.org/abs/2004.14557) [[code]](https://github.com/Alison-brie/MultiPropReg)
* [MedIA 2023] Hyper-Convolutions via Implicit Kernels for Medical Image Analysis [[pdf]](https://arxiv.org/abs/2202.02701) [[code]](https://github.com/tym002/Hyper-Convolution)
* [TIP 2023] Automated learning for deformable medical image registration by jointly optimizing network architectures and objective functions [[pdf]](https://arxiv.org/abs/2203.06810) [[code]](https://github.com/Alison-brie/AutoReg)



### Discrete Registration and Instance Optimization
* [MICCAI 2019] Closing the Gap between Deep and Conventional Image Registration using Probabilistic Dense Displacement Networks [[pdf]](https://arxiv.org/abs/1907.10931) [[code]](https://github.com/multimodallearning/pdd_net)
* [MICCAI 2020] Highly accurate and memory efficient unsupervised learning-based discrete CT registration using 2.5 D displacement search [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_19) [[code]](https://github.com/multimodallearning/pdd2.5/)
* [MICCAI Workshop 2021] Fast 3D registration with accurate optimisation and little learning for Learn2Reg 2021 [[pdf]](https://arxiv.org/abs/2112.03053) [[code]](https://github.com/multimodallearning/convexAdam)
* [MICCAI Workshop 2022] Voxelmorph++ going beyond the cranial vault with keypoint supervision and multi-channel instance optimisation [[pdf]](https://openreview.net/pdf?id=SrlgSXA3qAY) [[code]](https://github.com/mattiaspaul/VoxelMorphPlusPlus)
* [MICCAI 2023] SAMConvex: Fast Discrete Optimization for CT Registration using Self-supervised Anatomical Embedding and Correlation Pyramid [[pdf]](https://arxiv.org/abs/2307.09727) [[code]](https://github.com/alibaba-damo-academy/samconvex) 
* [TMI 2024] ConvexAdam: Self-Configuring Dual-Optimisation-Based 3D Multitask Medical Image Registration [[pdf]](https://ieeexplore.ieee.org/abstract/document/10681158) [[code]](https://github.com/multimodallearning/convexAdam)






### Registration on Specific Task
#### Multi-Modality Registration
* [IPMI 2019] Unsupervised deformable registration for multi-modal images via disentangled representations [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-20351-1_19)
* [MICCAI 2019] Synthesis and inpainting-based MR-CT registration for image-guided thermal ablation of liver tumors [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-32254-0_57)
* [MICCAI 2020] Adversarial uni-and multi-modal stream networks for multimodal image registration [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_22)
* [NeurIPS 2020] CoMIR: Contrastive multimodal image representation for registration [[pdf]](https://proceedings.neurips.cc/paper/2020/hash/d6428eecbe0f7dff83fc607c5044b2b9-Abstract.html) [[code]](https://github.com/MIDA-group/CoMIR)
* [TPMAI 2021] SymReg-GAN: symmetric image registration with generative adversarial networks [[pdf]](https://ieeexplore.ieee.org/abstract/document/9440692)
* [MedIA 2022] Deformable MR-CT image registration using an unsupervised, dual-channel network for neurosurgical guidance [[pdf]](https://www.sciencedirect.com/science/article/pii/S1361841521003376)
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
* [TMI 2020] ANHIR: Automatic Non-rigid Histological Image Registration Challenge, IEEE Transactions on Medical Imaging [[pdf]](https://ieeexplore.ieee.org/document/9058666) [[code]](https://github.com/MWod/ANHIR_MW)
* [Nature Communications 2023] Virtual alignment of pathology image series for multi-gigapixel whole slide image [[pdf]](https://www.nature.com/articles/s41467-023-40218-9) [[code]](https://github.com/MathOnco/valis)
* [TMI 2024] Unsupervised Non-rigid Histological Image Registration Guided by Keypoint Correspondences Based on Learnable Deep Features with Iterative Training [[pdf]](https://ieeexplore.ieee.org/document/10643202) [[code]](https://github.com/weixy17/IKCG/tree/main/ACROBAT)


#### Registration with Tumor in Image
* [MICCAI 2022] Unsupervised Deformable Image Registration with Absent Correspondences in Pre-operative and Post-Recurrence Brain Tumor MRI Scans [[pdf]](https://arxiv.org/abs/2206.03900) [[code]](https://github.com/cwmok/DIRAC)
* [ICCV 2023] Preserving Tumor Volumes for Unsupervised Medical Image Registration [[pdf]](https://arxiv.org/abs/2309.10153) [[code]](https://github.com/dddraxxx/Medical-Reg-with-Volume-Preserving)



### Fundation Model for Registration 
* [TMI 2022] SynthMorph: learning contrast-invariant registration without acquired images [[pdf]](https://arxiv.org/abs/2004.10282)  [[code]](https://martinos.org/malte/synthmorph/)
* [ArXiv 2023] UAE: Universal Anatomical Embedding on Multi-modality Medical Images [[pdf]](https://arxiv.org/pdf/2311.15111.pdf) [[code]](https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2)
* [MICCAI 2024] uniGradICON: A Foundation Model for Medical Image Registration [[pdf]](https://arxiv.org/abs/2403.05780) [[code]](https://github.com/uncbiag/uniGradICON)
* [MICCAI 2024 Workshop] multiGradICON: A Foundation Model for Multimodal Medical Image Registration [[pdf]](https://arxiv.org/abs/2408.00221) [[code]](https://github.com/uncbiag/uniGradICON)



### Registration Related/Based Application
#### Segmentation
* [CVPR 2019] Data Augmentation Using Learned Transformations for One-Shot Medical Image Segmentation [[pdf]](https://www.mit.edu/~adalca/files/papers/cvpr2019_brainstorm.pdf) [[code]](https://github.com/xamyzhao/brainstorm)
* [MICCAI 2019] DeepAtlas: Joint Semi-Supervised Learning of Image Registration and Segmentation [[pdf]](https://arxiv.org/abs/1904.08465) [[code]](https://github.com/uncbiag/DeepAtlas) 
* [MedIA 2022] Atlas-ISTN: joint segmentation, registration and atlas construction with image-and-spatial transformer networks [[pdf]](https://www.sciencedirect.com/science/article/pii/S1361841522000354) [[code]](https://github.com/biomedia-mira/atlas-istn)


#### Template Construction
* [NeurIPS 2019] Learning conditional deformable templates with convolutional networks [[pdf]](https://proceedings.neurips.cc/paper/2019/hash/bbcbff5c1f1ded46c25d28119a85c6c2-Abstract.html) [[code]](https://github.com/voxelmorph/voxelmorph/blob/dev/scripts/tf/train_cond_template.py)
* [ICCV 2021] Generative Adversarial Registration for Improved Conditional Deformable Templates [[pdf]](https://arxiv.org/abs/2105.04349) [[code]](https://github.com/neel-dey/Atlas-GAN)
* [CVPR 2022] Aladdin: Joint Atlas Building and Diffeomorphic Registration Learning with Pairwise Alignment [[pdf]](https://arxiv.org/abs/2202.03563) [[code]](https://github.com/uncbiag/Aladdin)
* [CVPR 2022] Topology-preserving shape reconstruction and registration via neural diffeomorphic flow [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_Topology-Preserving_Shape_Reconstruction_and_Registration_via_Neural_Diffeomorphic_Flow_CVPR_2022_paper.pdf) [[code]](https://github.com/Siwensun/Neural_Diffeomorphic_Flow--NDF)
* [NeurIPS 2022] Geo-SIC: Learning Deformable Geometric Shapes in Deep Image Classifiers [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2022/file/b328c5bd9ff8e3a5e1be74baf4a7a456-Paper-Conference.pdf) [[code]](https://github.com/jw4hv/Geo-SIC)


#### Cardiac Motion Estimation
* [MICCAI 2018] Joint learning of motion estimation and segmentation for cardiac MR image sequences [[pdf]](https://link.springer.com/chapter/10.1007/978-3-030-00934-2_53) [[code]](https://github.com/cq615/Joint-Motion-Estimation-and-Segmentation)
* [CVPR 2021] DeepTag: An unsupervised deep learning method for motion tracking on cardiac tagging magnetic resonance images [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/html/Ye_DeepTag_An_Unsupervised_Deep_Learning_Method_for_Motion_Tracking_on_CVPR_2021_paper.html) [[code]](https://github.com/DeepTag/cardiac_tagging_motion_estimation)
* [TMI 2022] MulViMotion: Shape-aware 3D Myocardial Motion Tracking from Multi-View Cardiac MRI [[pdf]](https://ieeexplore.ieee.org/abstract/document/9721301/) [[code]](https://github.com/ImperialCollegeLondon/Multiview-Motion-Estimation-for-3D-cardiac-motion-tracking)
* [MedIA 2023] Generative myocardial motion tracking via latent space exploration with biomechanics-informed prior [[pdf]](https://www.sciencedirect.com/science/article/pii/S1361841522003103) [[code]](https://github.com/cq615/BIGM-motion-tracking)
* [MICCAI 2024] TLRN: Temporal Latent Residual Networks For Large Deformation Image Registration [[pdf]](https://arxiv.org/abs/2407.11219) [[code]](https://github.com/nellie689/TLRN)


## Chanllenges
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


## Software
 - [ANTs](https://manpages.ubuntu.com/manpages/trusty/man1/ANTS.1.html)
 - [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg)
 - [LDDMM](https://github.com/brianlee324/torch-lddmm)
 - [DeedsBCV](https://github.com/mattiaspaul/deedsBCV)
 - [Elastix](https://github.com/SuperElastix/elastix)
 - [FireANTs (GPU)](https://github.com/rohitrango/fireants)

