# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:44:20 2019

@author: Administrator
"""

import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
# parameters

#image_shape
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot = True)

print(mnist.train.images.shape)
print(mnist.test.images.shape)
print(mnist.train.labels.shape)
print(mnist.test.labels.shape)

# 网络输入输出
x = tf.placeholder(shape = [None,784],dtype = tf.float32)
y = tf.placeholder(shape = [None,10], dtype = tf.float32)

# 创建网络
def weights_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev = 0.1, dtype = tf.float32)
    return tf.Variable(initial)

def biases_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, trainable=True)

# 定义卷积层

def conv2d(x, W):
    # 默认 strides[0]=strides[3]=1, strides[1]为x方向步长，strides[2]为y方向步长
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', data_format="NHWC")

# pooling 层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# 创建卷积网络
    
def buid_2fc_net():
    #创建两层fc层，第二层为softmax层
    fc1_W = weights_variable([784,1024])
    fc1_b = biases_variable([1024])
    fc1 = tf.nn.relu(tf.matmul(x,fc1_W)+fc1_b)

    fc2_W = weights_variable([1024,10])
    fc2_b = biases_variable([10])
    fc2 = tf.nn.softmax(tf.matmul(fc1,fc2_W)+fc2_b)
    return fc2

def build_conv_net(x,keep_prob):
    # 把X转为卷积所需要的形式
    X = tf.reshape(x, [-1, 28, 28, 1])
    # 第一层卷积：5×5×1卷积核32个 [5，5，1，32],h_conv1.shape=[-1, 28, 28, 32]
    W_conv1 = weights_variable([5,5,1,32])
    b_conv1 = biases_variable([32])
    h_conv1 = tf.nn.relu(conv2d(X, W_conv1)+b_conv1)

    # 第一个pooling 层[-1, 28, 28, 32]->[-1, 14, 14, 32]
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积：5×5×32卷积核64个 [5，5，32，64],h_conv2.shape=[-1, 14, 14, 64]
    W_conv2 = weights_variable([5,5,32,64])
    b_conv2 = biases_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # 第二个pooling 层,[-1, 14, 14, 64]->[-1, 7, 7, 64] 
    h_pool2 = max_pool_2x2(h_conv2)

    # flatten层，[-1, 7, 7, 64]->[-1, 7*7*64],即每个样本得到一个7*7*64维的样本
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    # fc1
    W_fc1 = weights_variable([7*7*64, 1024])
    b_fc1 = biases_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout: 输出的维度和h_fc1一样，只是随机部分值被值为零
    
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 输出层
    W_fc2 = weights_variable([1024, 10])
    b_fc2 = biases_variable([10])
    out_put = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    return out_put

# 评估函数
def loss(out_put):
    cross_entropy = -tf.reduce_sum(y * tf.log(out_put))
    return cross_entropy

def main():
    keep_prob = tf.placeholder(tf.float32)
    out_put = build_conv_net(x,keep_prob)
    
    # 2.优化函数：AdamOptimizer, 优化速度要比 GradientOptimizer 快很多
    cross_entropy = loss(out_put)
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(out_put, 1), tf.arg_max(y, 1))  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for i in range(10000):
        x_batch, y_batch = mnist.train.next_batch(64)
        acc,_ = sess.run([accuracy,train_step], feed_dict = {x:x_batch, y: y_batch,keep_prob: 0.5})
        if (i+1) % 200 == 0:
            #x_batch, y_batch = mnist.train.next_batch(64)
            train_accuracy = sess.run([accuracy],feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})
            print ("step %d, training acc %g" % (i+1, train_accuracy[0]))
        if (i+1) % 1000 == 0:
            test_accuracy = sess.run(accuracy,feed_dict={x: mnist.test.images[0:256], y: mnist.test.labels[0:256], keep_prob: 1.0})
            print ("= " * 10, "step %d, testing acc %g" % (i+1, test_accuracy))
            
if __name__=='__main__':
    main()