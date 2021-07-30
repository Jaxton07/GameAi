# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import cv2
import sys
import csv

sys.path.append('game/')
import wrapped_flappy_bird as game

ACTIONS = 2
IMAGE_SIZE = 80

oldRound = 0   # 未更新的得分


# 1. 创建文件对象,保存得分日志
input_scorePath = r'D:\Coding\Pycharm\Python file\AI\DeepLearningFlappyBird-master\my_logs\test_score509.csv'
f2 = open(input_scorePath, 'a', encoding='utf-8', newline='' "")        # 追加写入
csv_writer = csv.writer(f2)                     # 2. 基于文件对象构建 csv写入对象
csv_writer.writerow(["rounds", "score"])     # 3. 构建列表头
f2.close()

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# 加载模型
saver = tf.compat.v1.train.import_meta_graph('./save_model_backup/flappy_bird_dqn-5090000.meta')
saver.restore(sess, tf.train.latest_checkpoint('./save_model_backup/'))
graph = tf.compat.v1.get_default_graph()

S = graph.get_tensor_by_name('S:0')
Q = graph.get_tensor_by_name('Q/BiasAdd:0')

game_state = game.GameState()

do_nothing = np.zeros(ACTIONS)
do_nothing[0] = 1
img, reward, terminal = game_state.frame_step(do_nothing)
img = cv2.cvtColor(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2GRAY)
_, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
S0 = np.stack((img, img, img, img), axis=2)


while game.rounds <= 100:
    Qv = sess.run(Q, feed_dict={S: [S0]})[0]
    Av = np.zeros(ACTIONS)
    Av[np.argmax(Qv)] = 1

    img, reward, terminal = game_state.frame_step(Av)
    img = cv2.cvtColor(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    img = np.reshape(img, (IMAGE_SIZE, IMAGE_SIZE, 1))
    S0 = np.append(S0[:, :, 1:], img, axis=2)

    # 保存得分
    rounds = game.getRounds()
    if oldRound != rounds:
        newScore = game.getThisRoundScore()
        f4 = open(input_scorePath, 'a', encoding='utf-8', newline='' "")  # 追加写入
        csv_writer = csv.writer(f4)  # 2. 基于文件对象构建 csv写入对象
        csv_writer.writerow([rounds, newScore])  # 保存得分数据
        f4.close()
        oldRound = rounds

    # 打印分数
    print("rounds:" + str(game.rounds) + "    bestScore:" + str(game.getBestScore()) + "    aveScore:"
          + str(game.getAvescore()) + "    score:" + str(game_state.score))