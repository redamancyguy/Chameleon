
# Chameleon: Towards Update-Efficient Learned Indexing for Locally Skewed Data
### This work is a novel learned index constructed base on Multi-Agent Reinforcement Learning.
## benchmark: Includes comparative experiments with previous work.
## include: Auxiliary tools.
## index: Core components of Chameleon. Workspace is used for training Chameleon.
## others: Baselines.

# Before running 
### 1.The example can be seen in any file that includes "CHA:: Index" in the benchmark test.
### 2.To successfully run the code, you need to add the source code baselines of b+tree, alex, pgm, lipp to the others folder. Or delete the baseline related benchmark code and only run the chameleon.
### 3.Before running the chameleon, please make sure that the computer has the necessary cuda, libtorch, and boost libraries installed.

## Classes overview

This library provides the following classes:

- `CHA::Index | index/include/Index.hpp` Chameleon index structure.
- `Global_Q_network | index/include/RL_network.hpp` The DQN structure for DARE.
- `Small_Q_network | index/include/RL_network.hpp` The DQN structure for TSMDP.
- `pgm::CompressedPGMIndex` compresses the segments to reduce the space usage of the index.
- `pgm::OneLevelPGMIndex` uses a binary search on the segments rather than a recursive structure.
- `pgm::BucketingPGMIndex` uses a top-level lookup table to speed up the search on the segments.
- `pgm::EliasFanoPGMIndex` uses a top-level succinct structure to speed up the search on the segments.
