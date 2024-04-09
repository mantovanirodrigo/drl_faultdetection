# Deep Reinforcement Learning framework for fault detection in continuous chemical processes

## Master thesis
### Abstract:

Modern industrial processes are subject to increasingly higher quality, safety, environmental and economical standards. The implementation of closed-loop control aims to guarantee that these standards are met by compensating the effects of disturbances and changes that might occur in the process. However, faults, i.e., unpermitted deviations of at least one characteristic property or variable of the system, can still happen in most industrial processes. Faults can significantly impact the process and make it more difficult to meet several requirements. Therefore, process monitoring tasks that
detect, diagnose and remove faults are increasingly more necessary. In the field of Machine Learning, some of the most relevant approaches to fault detection are supervised learning-based methods, in which the learning process focuses on determining the mapping between input variables, i.e., process data, and some discrete set of classes, in this case the fault types. Recently, there has been an increase in industrial fault detection approaches based on Deep Reinforcement Learning (DRL), which consists on building intelligent agents that interact with the environment by performing actions
and receiving rewards based on the quality of the actions performed. However, the majority of the methods currently found in literature focuses in equipments such as industrial rotating machinery and rely on datasets with a small number of variables. Hence, this paper’s objective is to develop a reinforcement learning framework that can be used to perform fault detection in continuous chemical process, which have a large number of variables and complex, non-linear relations among them. The proposed framework was able to train, validate and test DRL agents with the Deep QLearning algorithm and use these agents to detect 20 different fault types in the simulated Tennessee Eastman Process benchmark. The model reached overall higher performance according to four selected metrics if compared to a Principal Component Analysis baseline model.

*Keywords*: Continuous chemical process, Fault detection, Reinforcement learning, Deep Q-Learning, Machine learning
## Folders
- agents: contains RL agents
- models: contains trained models
- logs: contains training and validation logs
- environments: contains the custom environment

## Files
- main.py: main training script
- utils.py: auxiliary functions
- test.py: test script

