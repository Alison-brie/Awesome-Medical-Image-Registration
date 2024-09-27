
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
 - [Discrete Optimization-based Registration](#discrete-optimization-based-registration)
 - [Joint Affine and Deformable Registration](#joint-affine-and-deformable-registration)
 - [Multi-Modality Registration](#multi-modality-registration)
 - [2D-3D Registration](#2d-3d-registration)
 - [Histological Image Registration](#histological-image-registration)



#### Original CNN model for Registration
* [End-to-end unsupervised deformable image registration with a convolutional neural network](https://arxiv.org/pdf/1704.06065) - Bob D. de Vos, Floris F. Berendsen, Max A. Viergever, Marius Staring, Ivana Išgum. MICCAI Workshop 2017
* [An Unsupervised Learning Model for Deformable Medical Image Registration](https://arxiv.org/abs/1802.02604) - Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca. CVPR 2018
* [Inverse-Consistent Deep Networks for Unsupervised Deformable Image Registration](https://arxiv.org/pdf/1809.03443) - Jun Zhang. arxiv 2018
* [VoxelMorph: A Learning Framework for Deformable Medical Image Registration](https://arxiv.org/abs/1809.05231) - Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca. IEEE Transactions on Medical Imaging 2019


#### Deep Diffeomorphic Registration 
* [Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration](https://arxiv.org/abs/1805.04605) - Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu. MICCAI 2018
* [Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces](https://arxiv.org/abs/1903.03545) - Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu. Medial Image Analysis 2019
* [Region-specific Diffeomorphic Metric Mapping](https://drive.google.com/file/d/1kIuunw6FP2ek8ZsLw92zL6RJw02YU7Nx) - Zhengyang Shen, François-Xavier Vialard, Marc Niethammer. NeurIPS 2019
* [Metric Learning for Image Registration](https://drive.google.com/file/d/1kIuunw6FP2ek8ZsLw92zL6RJw02YU7Nx) - Marc Niethammer, Roland Kwitt, François-Xavier Vialard. CVPR 2019
* [DeepFLASH: An Efficient Network for Learning-based Medical Image Registration](https://arxiv.org/pdf/2004.02097) - Jian Wang, Miaomiao Zhang. CVPR 2020
* [Fast Symmetric Diffeomorphic Image Registration with Convolutional Neural Networks](https://arxiv.org/abs/2003.09514) - Tony C. W. Mok, Albert C. S. Chung. CVPR 2020
* [ICON: Learning Regular Maps Through Inverse Consistency](https://arxiv.org/pdf/2105.04459) - Hastings Greer, Roland Kwitt, Francois-Xavier Vialard, Marc Niethammer. ICCV 2023
* [GradICON: Approximate Diffeomorphisms via Gradient Inverse Consistency](https://drive.google.com/file/d/1j8u5n50knQUxhnHp1OMGEwsl8CX-lODX) - Lin Tian, Hastings Greer, François-Xavier Vialard, Roland Kwitt, Raúl San José Estépar, Marc Niethammer. CVPR 2023


#### Pyramid CNN for Registration
* [Dual-Stream Pyramid Registration Network](https://arxiv.org/pdf/1909.11966) - Miao Kang, Xiaojun Hu, Weilin Huang, Matthew R. Scott, Mauricio Reyes.  MICCAI 2019
* [Large Deformation Diffeomorphic Image Registration with Laplacian Pyramid Networks](https://arxiv.org/abs/2006.16148) - Tony C. W. Mok, Albert C. S. Chung. MICCAI 2020
* [Recursive Cascaded Networks for Unsupervised Medical Image Registration](https://arxiv.org/abs/1907.12353) - Shengyu Zhao, Yue Dong, Eric I-Chao Chang, Yan Xu. ICCV 2019

#### Transformer for Registration
* [TransMorph: Transformer for unsupervised medical image registration]() - Medical Image Analysis 2022
* [Swin-VoxelMorph: A Symmetric Unsupervised Learning Model for Deformable Medical Image Registration Using Swin Transformer]() - MICCAI 2022
* [XMorpher: Full Transformer for Deformable Medical Image Registration via Cross Attention]() - MICCAI 2022
* [Affine Medical Image Registration with Coarse-to-Fine Vision Transformer](https://arxiv.org/abs/2203.15216) - CVPR 2022
* [PIViT: Large Deformation Image Registration with Pyramid-Iterative Vision Transformer]() - MICCAI 2023
* [Correlation-aware Coarse-to-fine MLPs for Deformable Medical Image Registration]() - CVPR 2024




#### Hyperparameter learning in Registration 
* [HyperMorph: Amortized Hyperparameter Learning for Image Registration](https://arxiv.org/abs/2101.01035) - Andrew Hoopes, Malte Hoffmann, Bruce Fischl, John Guttag, Adrian V. Dalca. IPMI 2021
* [Conditional Deformable Image Registration with Convolutional Neural Network](https://arxiv.org/abs/2106.12673) - Tony C. W. Mok, Albert C. S. Chung. MICCAI 2021
* [Learning Deformable Image Registration from Optimization: Perspective, Modules, Bilevel Training and Beyond](https://arxiv.org/abs/2004.14557) - Risheng Liu, Zi Li, Xin Fan, Chenying Zhao, Hao Huang, Zhongxuan Luo. IEEE Transactions on Pattern Analysis and Machine Intelligence 2021
* [Learning the Effect of Registration Hyperparameters with HyperMorph](https://arxiv.org/abs/2203.16680) - Andrew Hoopes, Malte Hoffmann, Bruce Fischl, John Guttag, Adrian V. Dalca. Machine Learning for Biomedical Imaging 2022
* [Hyper-Convolutions via Implicit Kernels for Medical Image Analysis.](https://arxiv.org/abs/2202.02701) - T. Ma, A.Q. Wang, A. V. Dalca, M.R. Sabuncu. Medical Image Analysis 2023
* [Automated learning for deformable medical image registration by jointly optimizing network architectures and objective functions](https://arxiv.org/abs/2203.06810) - Xin Fan*, Zi Li*, Ziyang Li, Xiaolin Wang, Risheng Liu, Zhongxuan Luo, Hao Huang. IEEE Transactions on Image Processing 2023






#### Discrete Optimization-based Registration
* [Fast 3D registration with accurate optimisation and little learning for Learn2Reg 2021]() - MICCAI Workshop 2021
* [SAMConvex: Fast Discrete Optimization for CT Registration using Self-supervised Anatomical Embedding and Correlation Pyramid](https://arxiv.org/abs/2307.09727) - Zi Li*, Lin Tian*, Tony C. W. Mok, Xiaoyu Bai, Puyang Wang, Jia Ge, Jingren Zhou, Le Lu, Xianghua Ye, Ke Yan, Dakai Jin. MICCAI 2023
* [Unsupervised 3D registration through optimization-guided cyclical self-training](https://arxiv.org/abs/2306.16997) - Alexander Bigalke, Lasse Hansen, Tony C. W. Mok, Mattias P. Heinrich. MICCAI 2023





#### Joint Affine and Deformable Registration
* [Networks for Joint Affine and Non-Parametric Image Registration](https://drive.google.com/file/d/1fybx_qI9PNW14w8C2gOmBN6GoYn3G_Mu) - Zhengyang Shen, Xu Han, Zhenlin Xu, Marc Niethammer. CVPR 2019
* [Unsupervised 3D End-to-End Medical Image Registration with Volume Tweening Network](https://arxiv.org/pdf/1902.05020) - Shengyu Zhao*, Tingfung Lau*, Ji Luo*, Eric I-Chao Chang, Yan Xu. IEEE Journal of Biomedical and Health Informatics 2020
* [Non-iterative Coarse-to-Fine Transformer Networks for Joint Affine and Deformable Image Registration]() - MICCAI 2023





#### Multi-Modality Registration
* [MIND: Modality independent neighbourhood descriptor for multi-modal deformable registration]() - Medical Image Analysis 2012
* [Globally Optimal Deformable Registration on a Minimum Spanning Tree Using Dense Displacement Sampling]() - MICCAI 2012
* [SynthMorph: learning contrast-invariant registration without acquired images.](https://arxiv.org/abs/2004.10282) - Malte Hoffmann, Benjamin Billot, Juan Eugenio Iglesias, Bruce Fischl, Adrian V. Dalca. IEEE Transactions on Medical Imaging. 2022
* [Modality-Agnostic Structural Image Representation Learning for Deformable Multi-Modality Medical Image Registration](https://arxiv.org/abs/2402.18933) - Tony C. W. Mok*, Zi Li*, Yunhao Bai, Jianpeng Zhang, Wei Liu, Yan-Jie Zhou, Ke Yan, Dakai Jin, Yu Shi, Xiaoli Yin, Le Lu, Ling Zhang. CVPR 2024



#### 2D-3D Registration
* [LiftReg: Limited Angle 2D/3D Deformable Registration](https://drive.google.com/file/d/13Dw3RO1ZhF3vtLr9TyJGhJkF7wz8DrjC) - Lin Tian, Yueh Z. Lee, Raúl San José Estépar, Marc Niethammer. MICCAI 2022
* [Intraoperative 2D/3D Image Registration via Differentiable X-ray Rendering]() - CVPR 2024


#### Histological Image Registration
* [ANHIR: Automatic non-rigid histological image registration challenge]() - TMI 2020
* [Unsupervised Histological Image Registration Using Structural Feature Guided Convolutional Neural Network]() - TMI 2022
* [Virtual alignment of pathology image series for multi-gigapixel whole slide image]() - Nature Communications 2023
* [Unsupervised Non-rigid Histological Image Registration Guided by Keypoint Correspondences Based on Learnable Deep Features with Iterative Training]() - TMI 2024
* [The ACROBAT 2022 challenge: Automatic registration of breast cancer tissue]() - MIA 2024



