---
layout: post    
title: 【强化学习笔记5】无模型控制
categories: RL RL_Notes    
description: 强化学习笔记第五章
keywords: Machine Learning, Reinforcement Learning, Model-Free Control
---

上一章中我们讲到了无模型的预测(Model-Free Prediction)，这一讲我们讲无模型的控制(Model-Free Control)。无模型的预测讲的是我们在不知道MDP环境信息的条件下如何评估一个策略，无模型的控制讲的是我们在不知道MDP环境信息的条件下如何优化一个策略。

# Model-Free Control

我们上一章讲了在MDP环境信息未知的情况下，使用MC和TD方法来评估一个策略，那我们如何提升一个策略呢？

根据状态值函数： 

$$\pi'(s) = \underset{a \in A}{\operatorname{argmax}}{R_s^a + P_{ss'}^a V(s')}$$

根据状态-动作值函数：

$$\pi'(s) = \underset{a \in A}{\operatorname{argmax}}{Q(s, a)}$$

我们发现如果根据状态值函数来提升策略，需要知道环境信息，而这里我们又不知道环境信息，所以我们应该根据状态-动作值函数来提升策略。

我们前面在讲到探索-利用问题时说到了，在采样的时候应该采用随机策略，而不是使用确定性策略，这里介绍常用的 $\epsilon$贪婪策略。

$\epsilon$ 贪婪策略：

- 简单的想法来平衡探索和利用
- 保证所有行为的概率不为0
- 以 $1 - \epsilon$ 的概率选择贪婪行为
- 以 $\epsilon$ 的概率随机选择一个行为

$$\pi(a \mid s) = 
\begin{cases}
\epsilon / m + 1 - \epsilon & if & a^\ast=\underset{a \in A}{\operatorname{argmax}} Q(s,a)\\
\epsilon / m & otherwise & \\
\end{cases}
$$

**同步策略学习**：我们学习的策略与采样时使用的策略是同一个策略，比如都是采用 $\epsilon$ 贪婪策略。
**异步策略学习**：我们学习的策略与采样时使用的策略是不同策略，比如学习确定性策略，采样使用 $\epsilon$ 贪婪策略。

>那么为什么我们学习的策略和采样时用的策略要采用不同策略呢？这里大概直观的解释一下，我们要学习的策略是一个确定性策略，前面说过每个MDP一定存在一个最优的确定性策略，然而如果我们采样时也使用这个确定性策略，那我们就无法探索其他的策略和状态了，因为我们每次到该状态都只会采取一个行为，所以异步策略学习一般在采样时采用 $\epsilon$ 贪婪策略，而学习的是确定性策略。

那么采用同步策略学习和异步策略学习能保证提升吗？

$\epsilon$ 贪婪策略的同步策略学习的理论保证：

>证明：对于一个 $\epsilon$ 贪婪策略 $\pi$ ，和提升之后的 $\epsilon$ 贪婪策略 $\pi'$ ，$v_{\pi'}(s) \ge v_\pi(s)$
>
>$$
>\begin{aligned}
>q_\pi(s, \pi'(s)) &= \sum_{a \in A}{\pi'(a \mid s)q_\pi(s, a)} \\
>&= \epsilon / m \sum_{a \in A}{q_\pi(s, a)} + (1-\epsilon) \max_{a \in A}{q_\pi(s, a)} \\
>&= \epsilon / m \sum_{a \in A}{q_\pi(s, a)} + (1-\epsilon) \frac{\sum_{a \in A} (\pi(a \mid s) - \epsilon / m) }{1 - \epsilon} \max_{a \in A}{q_\pi(s, a)} \\
>&\ge \epsilon / m \sum_{a \in A}{q_\pi(s,a)} + (1-\epsilon) \sum_{a \in A} \frac{(\pi(a \mid s) - \epsilon / m) }{1 - \epsilon}{q_\pi(s, a)} \\
>&= \sum_{a \in A}\pi(a \mid s)q_\pi(s,a) \\
>&= v_\pi(s)
>\end{aligned} 
>$$
>
>而 $q_\pi(s, \pi'(s)) = v_{\pi'}(s)$, 所以 $v_{\pi'}(s) \ge v_\pi(s)$

关于异步策略学习的理论证明，原教程中没有给出证明，这里也不给出证明，感兴趣的请自行探索, 哈哈。

GLIE(Greedy in the Limit with Infinite Exploration)理论： 所有的状态-动作对都被探索无穷次，那么策略会收敛到最优贪婪策略。

比如，如果 $\epsilon_k = \frac{1}{k}$，那么 $\epsilon$ 贪婪是GLIE。

>我们直观的理解，就是说 $\epsilon$ 需要随着探索次数的增大而减小，因为探索次数越来越多，所有的状态也都探索的差不多了，不再需要探索。

## On-Policy Control

### On-Policy Monte-Carlo Control

算法流程：

- 按照策略 $\pi$ 生成经验片段
- 对经验片段中的每个状态和行为对 $(S_t, A_t)$
$$
\begin{aligned}
N(S_t, A_t) &\leftarrow N(S_t, A_t) + 1 \\
Q(S_t, A_t) &\leftarrow Q(S_t, A_t) + \frac{1}{N(S_t, A_t)}(G_t - Q(S_t, A_t))
\end{aligned}
$$

- 策略提升
$$
\begin{aligned}
\epsilon &\leftarrow \frac{1}{k} \\
\pi &\leftarrow \epsilon-greedy(Q)
\end{aligned}
$$

>GLIE的蒙特卡洛控制收敛到最优的状态-动作函数, $Q(s, a) \rightarrow q_\ast(s,a)$

### On-Policy Temporal-Difference Control

算法流程：

- 初始化状态S, 按照 $\epsilon$ 贪婪策略选择动作 A
- 观察奖励R和转移后的状态 S', 按照 $\epsilon$ 贪婪策略选择动作 A'
$$
\begin{aligned}
& Q(S,A) \leftarrow Q(S, A) + \alpha [R + \gamma Q(S', A') - Q(S,A)] \\
&S \leftarrow S'; A \leftarrow A'
\end{aligned}
$$

- 如果S'是终止状态，那么进行策略提升
$$
\begin{aligned}
\epsilon &\leftarrow \frac{1}{k} \\
\pi &\leftarrow \epsilon-greedy(Q)
\end{aligned}
$$
否则返回第2步
$S \leftarrow S', A \leftarrow A'$

因为同步TD控制方法需要<S,A,R,S',A'>，所以也称作 **Sarsa**方法。

原教程中给出了Sarsa的收敛性证明以及Sarsa($\lambda$)算法(Sarsa$\lambda$的策略评估部分使用的TD($\lambda$)算法)，感兴趣的请移步原教程。

## Off-Policy Control

对于异步策略学习方法，我们如何评估不同概率分布的期望呢？

**Importance Sampling**:

$$
\begin{aligned}
E_{X \sim P}[f(X)] &= \sum_{}P(x)f(X) \\
&= \sum_{}Q(X)\frac{P(X)}{Q(X)}f(X) \\
&= E_{X \sim Q} \left[ \frac{P(X)}{Q(X)}f(X) \right]
\end{aligned}
$$

### Monte_Carlo Off-Policy Control

- 使用按照策略 $\mu$ 生成的回报来评估策略 $\pi$
- 对回报进行加权
$$G_t^{\pi / \mu} = \frac{\pi(A_t \mid S_t)}{\mu(A_t \mid S_t)}\frac{\pi(A_{t+1} \mid S_{t+1})}{\mu(A_{t+1} \mid S_{t+1})} \cdots \frac{\pi(A_T \mid S_T)}{\mu(A_T \mid S_T)} G_t$$

- 更新公式
$$V(S_t) \leftarrow V(S_t) + \alpha (G_t^{\pi / \mu} - V(S_t))$$

- 如果策略 $\mu$ 的分布存在某点为0，而策略 $\pi$ 不为0，那么这种方式不能使用

- 这种方法显著增加了方差

### Temporal-Difference Off-Policy Control

- 使用按照策略 $\mu$ 生成的回报来评估策略 $\pi$
- 对回报进行加权，更新公式为
$$V(S_t) \leftarrow V(S_t) + \alpha \left( \frac{\pi(A_t \mid S_t)}{\mu(A_t \mid S_t)}(R_{t+1} + \gamma V(S_{t+1})) - V(S_t) \right)$$

- 比蒙特卡洛方差低，异步策略只需要在这单独的一步想似

### Q-Learning

Q-Learning思想相对于异步TD的两点改进：

- 下一个动作 $A'$，我们不根据策略 $\mu$ 选择，而是根据策略 $\pi$ 选择；
$$A' \sim \pi(\cdot \mid S')$$

- 在上述选择动作 $A'$ 的同时提升策略
$$A' = \pi(S') = \underset{a'}{\operatorname{argmax}}Q(S',a')$$

>回顾前面的策略迭代和值迭代方法，可以看到Q-Learning是一种值迭代方法。

所以Q-Learning的更新公式为

$$Q(S,A) \leftarrow Q(S, A) + \alpha\left( R + \gamma \max_{a'}Q(S',a')-Q(S,A) \right)$$

>Q-Learning控制收敛到最优状态-动作值函数 $Q(s,a) \rightarrow q_\ast(s,a)$

Q-Learning算法：

![](https://github.com/feedliu/feedliu.github.io/blob/master/images/blog/Q-Learning.png?raw=true)

最后总结一下DP和TD之间的关系：

![](https://github.com/feedliu/feedliu.github.io/blob/master/images/blog/DP-TD.png?raw=true)
