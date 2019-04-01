---
layout: post    
title: Q-learning 和 Double Q-learning
categories: RL RL_Feeling    
description: Q-learning 和 Double Q-learning的区别
keywords: Machine Learning, Reinforcement Learning, Q-learning, Double Q-learning
---

Q-learning是一种无模型的、异步策略、时间差分（TD）控制方法，关于无模型、异步策略、时间差分、预测和控制等概念，在先前的强化学习笔记中有详细描述。Double Q-learning是针对Q-learning的缺点提出的一种改进方法。

Q-learning算法的步骤为

![](https://github.com/feedliu/feedliu.github.io/blob/master/images/blog/Q-learning-algorithm.png?raw=true)

更新公式为

$$Q(S,A) \leftarrow Q(S,A)+\alpha[R + \gamma \max_aQ(S',a) - Q(S,A)]$$

但是Q-learning更新公式中 $\max_aQ(S',a)$ 会对 $Q(S', a)$  过估计(overestimation)，用最大状态-动作值来替代最大期望状态-动作值。这使得Q-learning在一些随机MDP环境中表现的很差。
Double Q-learning通过双估计(double estimator)来解决这一问题，具体证明参考[Double Q-learning paper](https://papers.nips.cc/paper/3964-double-q-learning.pdf)，论文中会涉及到统计学的一些知识。

Double Q-learning算法的步骤为

![](https://github.com/feedliu/feedliu.github.io/blob/master/images/blog/Double-Q-learning-algorithm.png?raw=true)

Double Q-learning会保存两个状态-动作值函数，每次更新时随机选择一个函数值更新，在更新其中一个函数值时会用到另一个函数值。
