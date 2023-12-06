
# Chameleon: Towards Update-Efficient Learned Indexing for Locally Skewed Data
##### This work is a novel learned index constructed base on Multi-Agent Reinforcement Learning.

- `benchmark` Includes comparative experiments with previous work.
- `includeinclude` Auxiliary.
- `index` Core components of Chameleon. Workspace is used for training Chameleon.
- `others` Baselines.

## 1 Before running
### 1.1 Prerequisites
- C++17.
- NVIDIA GPU.
- PyTorch with the PyTorch C++ frontend (libtorch).

### 1.2 Experimental Environments
- Ubuntu 22.04.3 LTS
- AMD 7900X CPU @ 4.7GHz
- 128 GB DDR5 5200Mhz main memory
- NVIDIA GeForce RTX 4070 12GB
- CUDA 12.1
- PyTorch 2.1.0

### 1.3 Datasets
##### Binary dataset of type <double,long>.The data size of 200M is 3.2GB.
##### Detailed information can be found in the paper

### 1.4 The paths required to run the program
##### Configured  in `index/include/Parameter.h`
- `data_father_path` path for storing datasets
- `model_father_path` path for storing models of RL
- `experience_father_path` path for storing experiments of DARE

### 1.5 Note
-  The example can be seen in any file that includes "CHA:: Index" in the benchmark test.
-  To successfully run the code, you need to add the source code baselines of b+tree, alex, pgm, lipp to the others folder. Or delete the baseline related benchmark code and only run the chameleon.
-  Before running the chameleon, please make sure that the computer has the necessary cuda, libtorch, and boost libraries installed.


## 2 Running

### 2.1 How to Build
```
mkdir Release
cd Release
cmake -DCMAKE_BUILD_TYPE=Release..
make
```
### 2.2 How to Execute Chameleon
#### 2.2.1 Execute within Multi-Agent RL
- `train TSMDP` index/workspace/train_TSMDP.cpp 
- `train DARE`  index/workspace/train_DARE.cpp
- `run a benchmark` Core components of Chameleon. Workspace is used for training Chameleon.
#### 2.2.2 Execute within a default parameter
- `Given a default parameter` The constructor for the index requires one parameter. Default parameters can be used, as shown in CHA::Configuration::default_configuration.

# 3 Classes & Function overview

This library provides the following classes:

- `CHA::Index | index/include/Index.hpp` Chameleon index structure.
- `Global_Q_network | index/include/RL_network.hpp` The DQN structure for DARE.
- `Small_Q_network | index/include/RL_network.hpp` The DQN structure for TSMDP.
- `RewardScalar | index/include/RL_network.hpp` A simple standardized scaling method.
- `GlobalController | index/include/Controller.hpp` Actor in DARE. The Implementation Process of Combining DQN and GA Algorithms.
- `GlobalController::get_best_action_GA | index/include/Controller.hpp` Get the best action for a given state.
- `CHA::Configuration | index/include/Configuration.hpp` Implementation class of parameter matrix M.
- `CHA::Index::get_fanout | index/include/Index.hpp` Get a fanout with the parameter matrix M with linear interpolation algorithm.
- `main | index/workspace/train_DARE.hpp` Train DARE with multiple threads. If there are multiple GPUs and sufficient CPU and main memory resources, the strategy for deploying processes on different GPUs can be appropriately changed.
- `main | index/workspace/train_TSMDP.hpp` Train TSMDP.
- `experience_t | index/include/experience.hpp` Implementation class of experience of DARE.
- `create_dataset | include/DataSet.hpp` Create dataset with local-skew distribution where the parameter `skew` is the skewness degree.
- `local_skew | include/DataSet.hpp` The function to measure the local skewness of a dataset.

# 4 References
[1] https://pytorch.org/
