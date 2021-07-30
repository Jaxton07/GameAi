# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
import cv2
import sys
import pygame
import csv

sys.path.append('game/')
import wrapped_flappy_bird as fb
from collections import deque

# 定义参数
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 10000
EXPLORE = 3000000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH = 32
IMAGE_SIZE = 80

oldScore = 0   # 未更新的得分
S_loss = 0
A_loss = 0
Y_loss = 0

# 1. 创建文件对象,保存得分日志
input_scorePath = r'D:\Coding\Pycharm\Python file\AI\DeepLearningFlappyBird-master\my_logs\get_score.csv'
f2 = open(input_scorePath, 'a', encoding='utf-8', newline='' "")        # 追加写入
csv_writer = csv.writer(f2)                     # 2. 基于文件对象构建 csv写入对象
csv_writer.writerow(["rounds", "score"])     # 3. 构建列表头
f2.close()

# 2保存loss数据
lossPath = r'D:\Coding\Pycharm\Python file\AI\DeepLearningFlappyBird-master\my_logs\loss_log.csv'
f = open(lossPath, 'a', encoding='utf-8', newline='' "")        # 追加写入
csv_writer = csv.writer(f)
csv_writer.writerow(["step", "loss"])
f.close()


# 定义一些网络输入和辅助函数，每一个S由连续的四帧游戏截图组成
S = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 4], name='S')
A = tf.placeholder(dtype=tf.float32, shape=[None, ACTIONS], name='A')
Y = tf.placeholder(dtype=tf.float32, shape=[None], name='Y')
k_initializer = tf.truncated_normal_initializer(0, 0.01)
b_initializer = tf.constant_initializer(0.01)


# 卷积  "same"表示不够卷积核大小的块就补0
def conv2d(inputs, kernel_size, filters, strides):
    return tf.layers.conv2d(inputs, kernel_size=kernel_size, filters=filters, strides=strides, padding='same',
                            kernel_initializer=k_initializer, bias_initializer=b_initializer)


# 最大池化
def max_pool(inputs):
    return tf.layers.max_pooling2d(inputs, pool_size=2, strides=2, padding='same')


# 激活
def relu(inputs):
    return tf.nn.relu(inputs)


# 定义网络结构，卷积、池化、两层全连接层结构

# 三个卷积层
h0 = max_pool(relu(conv2d(S, 8, 32, 4)))
h0 = relu(conv2d(h0, 4, 64, 2))
h0 = relu(conv2d(h0, 3, 64, 1))

h0 = tf.contrib.layers.flatten(h0)      # 输入全连接层前的预处理，将输入数据变为一个向量

# 两个全连接层
h0 = tf.layers.dense(h0, units=512, activation=tf.nn.relu, bias_initializer=b_initializer)
Q = tf.layers.dense(h0, units=ACTIONS, bias_initializer=b_initializer, name='Q')
Q_ = tf.reduce_sum(tf.multiply(Q, A), axis=1)

# tensorboard保存 log日志
with tf.name_scope('loss'):
    loss = tf.losses.mean_squared_error(Y, Q_)
    tf.summary.scalar('loss', loss)  # 损失函数用scalar_summary函数

# loss = tf.losses.mean_squared_error(Y, Q_)

optimizer = tf.train.AdamOptimizer(1e-6).minimize(loss)         # 采用Adam优化算法，也是基于梯度

# 用一个队列实现记忆模块，开始游戏，对于初始状态选择什么都不做
game_state = fb.GameState()
D = deque()

do_nothing = np.zeros(ACTIONS)
do_nothing[0] = 1
img, reward, terminal = game_state.frame_step(do_nothing)    # 输入动作
img = cv2.cvtColor(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2GRAY)     # 将彩色图像处理成黑白的二值图像
_, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
S0 = np.stack((img, img, img, img), axis=2)

# 继续进行游戏并训练模型
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()   # #合并所有的summary data的获取函数
# 保存图
writer = tf.summary.FileWriter("D:/Coding/Pycharm/Python file/AI/DeepLearningFlappyBird-master/logs/", sess.graph)

t = 0
success = 0
saver = tf.train.Saver()
epsilon = INITIAL_EPSILON

while True:
    if epsilon > FINAL_EPSILON and t > OBSERVE:     # 更新概率
        epsilon = INITIAL_EPSILON - (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE * (t - OBSERVE)

    Qv = sess.run(Q, feed_dict={S: [S0]})[0]
    Av = np.zeros(ACTIONS)
    if np.random.random() <= epsilon:           # 按epsilon的概率随机生成动作
        action_index = np.random.randint(ACTIONS)
    else:
        action_index = np.argmax(Qv)
    Av[action_index] = 1

    img, reward, terminal = game_state.frame_step(Av)   # 输入动作，返回状态
    if reward == 1:
        success += 1

    # 灰度处理和二值化
    img = cv2.cvtColor(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    img = np.reshape(img, (IMAGE_SIZE, IMAGE_SIZE, 1))
    S1 = np.append(S0[:, :, 1:], img, axis=2)

    D.append((S0, Av, reward, S1, terminal))    # 添加游戏记录数据
    if len(D) > REPLAY_MEMORY:
        D.popleft()

    if t > OBSERVE:
        minibatch = random.sample(D, BATCH)     # 过了观察期之后就随机从队列中选取小批量样本数据,样本大小为32
        S_batch = [d[0] for d in minibatch]
        A_batch = [d[1] for d in minibatch]
        R_batch = [d[2] for d in minibatch]
        S_batch_next = [d[3] for d in minibatch]
        T_batch = [d[4] for d in minibatch]         # terminal

        Y_batch = []
        Q_batch_next = sess.run(Q, feed_dict={S: S_batch_next})
        for i in range(BATCH):
            if T_batch[i]:
                Y_batch.append(R_batch[i])      # reward奖励
            else:
                Y_batch.append(R_batch[i] + GAMMA * np.max(Q_batch_next[i]))

        sess.run(optimizer, feed_dict={S: S_batch, A: A_batch, Y: Y_batch})     # 梯度下降，反向传播更新参数

        # 暂存数据
        S_loss = S_batch
        A_loss = A_batch
        Y_loss = Y_batch

        # 保存loss到tensorboard日志
        rs = sess.run(merged, feed_dict={S: S_batch, A: A_batch, Y: Y_batch})
        writer.add_summary(rs, t)   # 把数据添加到文件中
        # print(sess.run(loss, feed_dict={S: S_batch, A: A_batch, Y: Y_batch}))  # 打印loss

    S0 = S1
    t += 1

    if t > OBSERVE and t % 10000 == 0:
        saver.save(sess, './saved_model/flappy_bird_dqn', global_step=t)

        # 保存loss到csv    000000000**************
        lossValue = sess.run(loss, feed_dict={S: S_loss, A: A_loss, Y: Y_loss})
        f4 = open(lossPath, 'a', encoding='utf-8', newline='' "")  # 追加写入
        csv_writer = csv.writer(f4)  # 2. 基于文件对象构建 csv写入对象
        csv_writer.writerow([t, lossValue])  # 保存得分数据
        f4.close()

    if t <= OBSERVE:
        state = 'observe'
    elif t <= OBSERVE + EXPLORE:
        state = 'explore'
    else:
        state = 'train'

    # 保存得分
    newScore = fb.getThisRoundScore()
    rounds = fb.getRounds()
    if oldScore != newScore:
        oldScore = newScore
        if t > OBSERVE and rounds % 10 == 0:
            f4 = open(input_scorePath, 'a', encoding='utf-8', newline='' "")  # 追加写入
            csv_writer = csv.writer(f4)                 # 2. 基于文件对象构建 csv写入对象
            csv_writer.writerow([rounds, oldScore])     # 保存得分数据
            f4.close()

    print('Current Step %d Rounds %d Success %d State %s Epsilon %.6f Action %d Reward %f Q_MAX %f' % (
        t, rounds, success, state, epsilon, action_index, reward, np.max(Qv)))


