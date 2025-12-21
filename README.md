<!-- # Fast3DHPE -->
<p align="center">
  <img src="readme_main.gif"> 
</p>

**Fast3DHPE** is a unified and efficient repository designed to help researchers quickly get started with **3D Human Pose Estimation**, providing a consistent framework for training, evaluation, and comparison across multiple representative methods.
This repository is supported by [BNU-IVC](https://github.com/BNU-IVC) and [WATRIX.AI](http://www.watrix.ai).

## 🆕 What's New

-- **[Dec 2025]** 🚀🚀🚀 Our paper **FastDDHPose: Towards Unified, Efficient, and Disentangled 3D Human Pose Estimation** is released, and the corresponding code for our model (**FastDDHPose**) and the unified framework (**Fast3DHPE**) is being progressively made available.  
  👉 Paper: [arXiv](https://arxiv.org/abs/2512.14162)

-- **[Feb 2024]** 🔥🔥🔥 Our paper **Disentangled Diffusion-Based 3D Human Pose Estimation with Hierarchical Spatial and Temporal Denoiser** has been accepted by **AAAI 2024**!  
  👉 Paper: [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/27847) ｜ [arXiv](https://arxiv.org/abs/2403.04444) ｜ Code: [GitHub](https://github.com/Andyen512/DDHPose)
<!-- (https://ojs.aaai.org/index.php/AAAI/article/view/27847).   -->

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2403.04444-b31b1b.svg)](https://arxiv.org/abs/2403.04444)   -->
<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangled-diffusion-based-3d-human-pose/multi-hypotheses-3d-human-pose-estimation-on)](https://paperswithcode.com/sota/multi-hypotheses-3d-human-pose-estimation-on?p=disentangled-diffusion-based-3d-human-pose)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangled-diffusion-based-3d-human-pose/monocular-3d-human-pose-estimation-on-human3)](https://paperswithcode.com/sota/monocular-3d-human-pose-estimation-on-human3?p=disentangled-diffusion-based-3d-human-pose)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffab41)](https://huggingface.co/Andyen512/DDHpose) -->


## ✅ Supports
The following algorithms and datasets are currently supported. More methods and benchmarks will be continuously integrated.
### Supported Algorithms
- [x] [VideoPose3D (ICCV 2019)](https://github.com/facebookresearch/VideoPose3D)
- [x] [PoseFormer (ICCV 2021)](https://github.com/zczcwh/PoseFormer)
- [x] [PoseFormerV2 (CVPR 2023)](https://github.com/QitaoZhao/PoseFormerV2)
- [x] [MixSTE (CVPR 2022)](https://github.com/JinluZhang1126/MixSTE)
- [x] [STCFormer (CVPR 2023)](https://github.com/zhenhuat/STCFormer)
- [x] [D3DP (ECCV 2022)](https://github.com/paTRICK-swk/D3DP)
- [x] [KTPFormer (CVPR 2024)](https://github.com/JihuaPeng/KTPFormer)
- [x] [FinePOSE (CVPR 2024)](https://github.com/PKU-ICST-MIPL/FinePOSE_CVPR2024)
- [x] [DDHPose (AAAI 2024)](https://github.com/Andyen512/DDHPose)
- [x] [FastDDHPose (Arxiv 2025)](https://github.com/Andyen512/Fast3DHPE)

### Supported Datasets
- [x] [Human3.6M (CVPR 2014)](http://vision.imar.ro/human3.6m/)
- [x] [MPI-INF-3DHP (3DV 2017)](http://gvv.mpi-inf.mpg.de/3dhp-dataset/)

## 🚀 Getting Started

Please see [0.get_started.md](docs/0.get_started.md). We also provide the following tutorials for your reference:

### 🧩 TODO
- [ ] 🛠️ Complete training and evaluation code
- [ ] 📦 Release pre-trained model checkpoints
- [ ] 📊 Provide visualization scripts

## **Citation**
```bibtex
@inproceedings{cai2024disentangled,
  title={Disentangled Diffusion-Based 3D Human Pose Estimation with Hierarchical Spatial and Temporal Denoiser},
  author={Cai, Qingyuan and Hu, Xuecai and Hou, Saihui and Yao, Li and Huang, Yongzhen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={2},
  pages={882--890},
  year={2024}
}
```
```bibtex
@article{cai2025fastddhpose,
  title={FastDDHPose: Towards Unified, Efficient, and Disentangled 3D Human Pose Estimation},
  author={Cai, Qingyuan and Zhang, Linxin and Hu, Xuecai and Hou, Saihui and Huang, Yongzhen},
  journal={arXiv preprint arXiv:2512.14162},
  year={2025}
}
```

## Acknowledgement
Our code refers to the following repositories.
* [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
* [PoseFormer](https://github.com/zczcwh/PoseFormer)
* [PoseFormerV2](https://github.com/QitaoZhao/PoseFormerV2)
* [MixSTE](https://github.com/JinluZhang1126/MixSTE)
* [STCFormer](https://github.com/zhenhuat/STCFormer)
* [D3DP](https://github.com/paTRICK-swk/D3DP)
* [DDHPose](https://github.com/Andyen512/DDHPose)
* [KTPFormer](https://github.com/JihuaPeng/KTPFormer)
* [FinePOSE](https://github.com/PKU-ICST-MIPL/FinePOSE_CVPR2024)

We thank the authors for releasing their codes