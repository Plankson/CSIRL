# Curricular Subgoal for Inverse Reinforcement Learning

Official codebase for paper [Curricular Subgoal for Inverse Reinforcement Learning](). This code is implemented based on Pytorch and evaluated on tasks from modified [highway-env](https://github.com/eleurent/highway-env) and please refer to that repo for more documentation.



## 1. Prerequisites

#### Install dependencies

See `requirment.txt` file for more information about how to install the dependencies.

#### Install Highway-env
It should be noted that we make some modification on the original [highway-env](https://github.com/eleurent/highway-env) to make it more fit the real driving environment.
The updated highway-env is provided by highway_modify, which can be installed by running `pip install -e.` in `highway_modify` directory.


## 2. Usage
Detailed instructions to replicate the results in the paper are contained in `scripts` directory. 
Here we give the form of the instructions. 


```bash
python main.py env=[env_name]  expert.tra=[expert_dataset_path] seed=[seed_id]

# env_name:
# highway-fast-continues-v0_s35_d1
# merge-continues-v0
# roundabout-continues-v1
# intersection-continues-v0-o1

# expert_dataset_path:  expert dataset path of npy file

# seed_id: random seed
```
