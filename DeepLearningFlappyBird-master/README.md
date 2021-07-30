# 这是用deep Q-NetWork来训练AI玩flappy bird游戏



### 注意：项目中的目录要改成自己的

## Installation Dependencies(安装依赖库):
* Python  3
* TensorFlow 2.0
* pygame
* OpenCV-Python

## How to Run(如何运行)?
1.训练神经网络只需运行 dqn_flappy.py 程序
2.测试和用训练好的模型玩游戏则运行 game_test.py ,可以在 game_test.py 中自行更换加载的checkpoint


## 什么是Deep Q-Network?
DQN是一个深度强化学习的神经网络，它有卷积神经网络和全连接层组成。

## Deep Q-Network Algorithm
具体的训练算法如下：

```
Initialize replay memory D to size N
Initialize action-value function Q with random weights
for episode = 1, M do
    Initialize state s_1
    for t = 1, T do
        With probability ϵ select random action a_t
        otherwise select a_t=max_a  Q(s_t,a; θ_i)
        Execute action a_t in emulator and observe r_t and s_(t+1)
        Store transition (s_t,a_t,r_t,s_(t+1)) in D
        Sample a minibatch of transitions (s_j,a_j,r_j,s_(j+1)) from D
        Set y_j:=
            r_j for terminal s_(j+1)
            r_j+γ*max_(a^' )  Q(s_(j+1),a'; θ_i) for non-terminal s_(j+1)
        Perform a gradient step on (y_j-Q(s_j,a_j; θ_i))^2 with respect to θ
    end for
end for
```



