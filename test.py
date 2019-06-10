# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:35:28 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
from models_tf.VGG16_TF import VGG
from models_tf.vgg16_tflayers import VGG16
from models_tf.Resnets_TF import Resnet

def test_softmax():
    logits = tf.constant([[1.,3., 5.],[4,2,1]])
    labels = tf.constant([2,2])
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)
    cost = tf.reduce_mean(loss)
    with tf.Session() as sess:
        print(sess.run(cost))

 
def test_tf_train_slice_input_producer():
    images = ['img1', 'img2', 'img3', 'img4', 'img5']
    labels= [1,2,3,4,5]
     
    epoch_num=8
     
    f = tf.train.slice_input_producer([images, labels],num_epochs=None,shuffle=True)
     
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(epoch_num):
            k = sess.run(f)
            print ('*'*18)
            print (i,k)
     
        coord.request_stop()
        coord.join(threads)
        
def test_tf_train_batch():
    #batch(tensors, batch_size, num_threads=1, capacity=32,enqueue_many=False,\
    #shapes=None, dynamic_pad=False,allow_smaller_final_batch=False, shared_name=None, name=None)
    # -*- coding:utf-8 -*-
 
    # 样本个数
    sample_num=5
    # 设置迭代次数
    epoch_num = 2
    # 设置一个批次中包含样本个数
    batch_size = 3
    # 计算每一轮epoch中含有的batch个数
    batch_total = int(sample_num/batch_size)+1
     
    # 生成5个数据和标签
    def generate_data(sample_num=sample_num):
        labels = np.asarray(range(0, sample_num))
        images = np.random.random([sample_num, 224, 224, 3])
        print('image size {},label size :{}'.format(images.shape, labels.shape))
     
        return images,labels
     
    def get_batch_data(batch_size=batch_size):
        images, label = generate_data()
        # 数据类型转换为tf.float32
        images = tf.cast(images, tf.float32)
        label = tf.cast(label, tf.int32)
     
        #从tensor列表中按顺序或随机抽取一个tensor
        input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
     
        image_batch, label_batch = tf.train.shuffle_batch(input_queue, batch_size=batch_size, num_threads=1, capacity=64)
        return image_batch, label_batch
     
    image_batch, label_batch = get_batch_data(batch_size=batch_size)
     
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        try:
            for i in range(epoch_num):  # 每一轮迭代
                print('************')
                for j in range(batch_total): #每一个batch
                    print ('--------')
                    # 获取每一个batch中batch_size个样本和标签
                    image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
                    # for k in
                    print(image_batch_v.shape, label_batch_v)
        except tf.errors.OutOfRangeError:
            print("done")
        finally:
            coord.request_stop()
        coord.join(threads)

def test_arg_max():
    #a = [[0],[2],[3],[4],[3],[2]]
    a= [1,2,0,5,5,6,9]
    print(a)
    y=tf.one_hot(a,10,1,0)
    #print(y)
    with tf.Session() as sess:
        b = tf.arg_max(y,1)
        print(sess.run(y))
        print(sess.run(b))
        
a=tf.constant(2)    
b=tf.constant(3)    
x=tf.constant(4)    
y=tf.constant(5)    
z = tf.multiply(a, b)    
t=True
result = tf.cond(tf.cast(t, tf.bool), lambda: tf.add(x, z), lambda: tf.square(y))    
with tf.Session() as session:    
    print(result.eval())
    
if __name__=='__main__':
    #test_tf_train_slice_input_producer()
    #test_tf_train_batch()
    pass
    #test_arg_max()