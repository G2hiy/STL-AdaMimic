# AdaMimic: Adaptive Motion Tracking
[![arXiv](https://img.shields.io/badge/arXiv-2510.14454-brown)](https://arxiv.org/abs/2510.14454)
[![](https://img.shields.io/badge/Website-%F0%9F%9A%80-yellow)](https://taohuang13.github.io/adamimic.github.io/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)]()
[![](https://img.shields.io/badge/Youtube-🎬-red)](https://www.youtube.com/watch?v=OGDoPvs7GS0)


This is the official PyTorch implementation of the paper "[**Towards Adaptable Humanoid Control via Adaptive Motion Tracking**]()" by 

[Tao Huang](https://taohuang13.github.io/), [Huayi Wang](https://why618188.github.io/), [Junli Ren](https://renjunli99.github.io/), [Kangning Yin](https://yinkangning0124.github.io/), [Zirui Wang](https://scholar.google.com/citations?user=Vc3DCUIAAAAJ&hl=zh-TW), [Xiao Chen](https://xiao-chen.tech/), [Feiyu Jia](https://trap-1.github.io/), [Wentao Zhang](), [Junfeng Long](https://junfeng-long.github.io/), [Jingbo Wang](https://wangjingbo1219.github.io/)†, [Jiangmiao Pang](https://oceanpang.github.io/)†

<p align="left">
  <img width="98%" src="docs/teaser_website.png" style="box-shadow: 1px 1px 6px rgba(0, 0, 0, 0.3); border-radius: 4px;">
</p>

## 🛠️ Installation Instructions
Clone this repository:
```bash
git clone https://github.com/InternRobotics/AdaMimic.git
cd AdaMimic
```
Create a conda environment:
```bash
conda env create -f conda_env.yml 
conda activate adamimic
```

Download and install [Isaac Gym](https://developer.nvidia.com/isaac-gym):
```bash
cd isaacgym/python && pip install -e .
```

Install rsl_rl (PPO implementation) and legged gym:
```bash
cd rsl_rl && pip install -e . && cd .. 
cd legged_gym &&  pip install -e . && cd .. 
```

## Usage
### Commands for AdaMimic
Training stage 1:
```bash
python legged_gym/scripts/train.py +dataset=g1_dof27/${task} +algorithm=adamimic/stage1 
```
Here, `${task}` can be one of task in [this list](./legged_gym/legged_gym/configs/dataset/g1_dof27/).

Training stage 2:
```bash
python legged_gym/scripts/train.py +dataset=g1_dof27/${task} +algorithm=adamimic/stage2 checkpoint_path=${path/to/stage1_ckpt} 
```
The `${path/to/stage1_ckpt} ` should be replaced with checkpoints trained in the stage 1.

Play policies:
```bash
python legged_gym/scripts/play.py +dataset=g1_dof27/${task} +algorithm=adamimic/stage2 resume_path=${path/to/stage2_ckpt} 
```

### Commands for baselines
Train baselines:
```bash
python legged_gym/scripts/train.py +dataset=g1_dof27/${task} +algorithm=${baseline}
```
All configurations of `${baseline} ` are implemented in [this folder](./legged_gym/legged_gym/configs/algorithm/).

Play policies
```bash
python legged_gym/scripts/play.py +dataset=g1_dof27/${task} +algorithm=${baseline} resume_path=${path/to/baseline_ckpt} 
```

## ✉️ Contact
For any questions, please feel free to email taou.cs13@gmail.com. We will respond to it as soon as possible.


## 🎉 Acknowledgments
This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

* [legged_gym](https://github.com/leggedrobotics/legged_gym) and [HIMLoco](https://github.com/OpenRobotLab/HIMLoco): The foundation for training and running codes.
* [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): Reinforcement learning algorithm implementation.
* [ASAP](https://github.com/LeCAR-Lab/ASAP): Motion tracking implementation.
* [AMP for hardware](https://github.com/escontra/AMP_for_hardware): AMP implementation.
* [GVHMR](https://github.com/zju3dv/GVHMR): SMPL motion reconstruction algorithom.

## 📝 Citation

If you find our work useful, please consider citing:
```
@article{huang2025adaptive,
  title={Towards Adaptable Humanoid Control via Adaptive Motion Tracking},
  author={Huang, Tao and Wang, Huayi and Ren, Junli and Yin, Kangning and Wang, Zirui and Chen, Xiao and Jia, Feiyu and Zhang, Wentao and Long, Jungfeng and Wang, Jingbo and Pang, Jiangmiao},
  year={2025}
}
```

## 📄 License

The code is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0 International License</a> <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>.
Commercial use is not allowed without explicit authorization.
