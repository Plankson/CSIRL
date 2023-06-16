# Curricular Subgoal for Inverse Reinforcement Learning

[![License: Apache](https://img.shields.io/badge/License-Apache-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2306.08232-b31b1b.svg)](https://arxiv.org/abs/2306.08232)


Official codebase for paper [Curricular Subgoal for Inverse Reinforcement Learning](https://arxiv.org/abs/2306.08232).

<div align="center">
<img src="https://github.com/Plankson/CSIRL/blob/master/introduction.png" width="50%">
</div>

## Overview

**TLDR:** Our main contribution is a dedicated curricular subgoal-based IRL framework that enables multi-stage imitation based on expert demonstrations. Extensive experiments conducted on the D4RL and autonomous driving benchmarks show that our proposed CSIRL framework yields significantly superior performance to state-of-the-art competitors, as well as better interpretability in the training process. Moreover, the robustness analysis experiments show that CSIRL still maintains high performance even with only one expert trajectory.

**Abstract:** Inverse Reinforcement Learning (IRL) aims to reconstruct the reward function from expert demonstrations to facilitate policy learning, and has demonstrated its remarkable success in imitation learning. To promote expert-like behavior, existing IRL methods mainly focus on learning global reward functions to minimize the trajectory difference between the imitator and the expert. However, these global designs are still limited by the redundant noise and error propagation problems, leading to the unsuitable reward assignment and thus downgrading the agent capability in complex multi-stage tasks. In this paper, we propose a novel Curricular Subgoal-based Inverse Reinforcement Learning (CSIRL) framework, that explicitly disentangles one task with several local subgoals to guide agent imitation. Specifically, CSIRL firstly introduces decision uncertainty of the trained agent over expert trajectories to dynamically select subgoals, which directly determines the exploration boundary of different task stages. To further acquire local reward functions for each stage, we customize a meta-imitation objective based on these curricular subgoals to train an intrinsic reward generator. Experiments on the D4RL and autonomous driving benchmarks demonstrate that the proposed methods yields results superior to the state-of-the-art counterparts, as well as better interpretability.

![image](https://github.com/Plankson/CSIRL/blob/master/framework.png)

## Prerequisites

#### Install dependencies

See `requirment.txt` file for more information about how to install the dependencies.

#### Install highway-env
It should be noted that we make some modification on the original [highway-env](https://github.com/eleurent/highway-env) to make it more fit the real driving environment. The modified highway-env is provided by `highway_modify`, which can be installed by running:

```bash
cd highway_modify
pip install -e .
```


## Usage
Detailed instructions to replicate the results in the paper are contained in `scripts` directory. 
Here we give the form of the instructions. 

```bash
# highway-fast
python main.py env=highway-fast-continues-v0_s35_d1 expert.tra=<EXPERT_DATASET_PATH> seed=<RANDOM_SEED>

# merge
python main.py env=merge-continues-v0 expert.tra=<EXPERT_DATASET_PATH> seed=<RANDOM_SEED>

# roundabout
python main.py env=roundabout-continues-v1 expert.tra=<EXPERT_DATASET_PATH> seed=<RANDOM_SEED>

# intersection
python main.py env=intersection-continues-v0-o1 expert.tra=<EXPERT_DATASET_PATH> seed=<RANDOM_SEED>
```

Make sure to replace `EXPERT_DATASET_PATH` with the path to the corresponding dataset in `expert_data`.


![image](https://github.com/Plankson/CSIRL/blob/master/exp-highway.png)


![image](https://github.com/Plankson/CSIRL/blob/master/exp-highway-table.png)

## Citation

If you find this work useful for your research, please cite our paper:

```
@article{liu2023CSIRL,
  title={Curricular Subgoal for Inverse Reinforcement Learning},
  author={Liu, Shunyu and Qing, Yunpeng and Xu, Shuqi and Wu, Hongyan and Zhang, Jiangtao and Cong, Jingyuan and Liu, Yunfu and Song, Mingli},
  journal={arXiv preprint arXiv:2306.08232},
  year={2023}
}
```

## Contact

Please feel free to contact me via email (<liushunyu@zju.edu.cn>, <qingyunpeng@zju.edu.cn>) if you are interested in my research :)
