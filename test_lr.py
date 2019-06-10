# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:06:30 2019

@author: Administrator
"""

# coding:utf-8
import os
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
import tensorflow as tf
 
LEARNING_RATE_BASE = 0.1  # 最初学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
LEARNING_RATE_STEP = 10  # 喂入多少轮BATCH-SIZE以后，更新一次学习率。一般为总样本数量/BATCH_SIZE
gloabl_steps = tf.Variable(0, trainable=False)  # 计数器，用来记录运行了几轮的BATCH_SIZE，初始为0，设置为不可训练
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE
                                           , gloabl_steps,
                                           LEARNING_RATE_STEP,
                                           LEARNING_RATE_DECAY,
                                           staircase=True)
# 定义指数下降学习率。
# 如果staircase=True，那就表明每decay_steps次计算学习速率变化，更新原始学习速率，如果是False，那就是每一步都更新学习速率。红色表示False，绿色表示True。
 
 
w = tf.Variable(tf.constant(5, tf.float32))
 
loss = tf.square(w + 1)
 
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=gloabl_steps)
print('start')
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(100):
        sess.run(train_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        learning_rate_val = sess.run(learning_rate)
        print('After %s steps:learning_rate is %f ,w is %f ,loss is %f ' % (i, learning_rate_val, w_val, loss_val))
