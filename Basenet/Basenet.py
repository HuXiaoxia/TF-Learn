# -*- coding: utf-8 -*-
"""
Created on Wed May 15 18:21:53 2019

@author: Administrator
"""

import os
import sys
currentUrl = os.path.dirname(__file__)
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
#print(parentUrl)
sys.path.append(parentUrl)

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from tensorflow.python.training import moving_averages

class Basenet(object):
    def __init__(self,input_tensor):
        self.input_tensor = input_tensor
        
    # 权重参数初始化函数
    def weights_variable(self,shape,name):
        """
        :param shape:当输出卷积层的权重参数时，该shape为卷积层的卷积核的shape，
        如[3,3,3,64],卷积核大小3*3，卷积网络的通道数输入和输出3，64；
        当输出为全连接层的权重参数时，shape为[d_in,d_out],全连接层输入维度与输出维度
        
        """
        weights=tf.get_variable(name,initializer=tf.random_normal(shape,stddev=0.01))
        #weights =tf.Variable( tf.random_normal(shape=shape,dtype =tf.float32,stddev=5e-2),name='weights')
        return weights
    # 偏置参数初始化函数
    def biases_variable(self,shape,name):
        """
        :param shape: 偏置维度，卷积核的输出通道数或者全连接层的输出维度，shape均为[d]一维列表
        """
        bias=tf.get_variable(name,initializer=tf.constant(0.0, shape=shape),trainable=True)
        #bias = tf.Variable(tf.constant(0.0, shape=shape),trainable=True)
        return bias
    # 卷积函数
    def conv2d(self,x,W,strides,padding='SAME'):
        return tf.nn.conv2d(x,W,strides,padding=padding)
    
    # maxpool函数
    def maxpool(self,x,kernel_size,strides,padding='VALID'):
        return tf.nn.max_pool(x, ksize=kernel_size, strides=strides, padding=padding)
    
    # 卷积层的bn函数
    def batch_norm_layer(self,Input,is_training,out_channels,name):
        gamma = tf.Variable(tf.ones([out_channels]))
        beta = tf.Variable(tf.zeros([out_channels]))
    
        pop_mean = tf.Variable(tf.zeros([out_channels]), trainable=False)
        pop_variance = tf.Variable(tf.ones([out_channels]), trainable=False)
    
        epsilon = 1e-3
    
        def batch_norm_training():
            # 一定要使用正确的维度确保计算的是每个特征图上的平均值和方差而不是整个网络节点上的统计分布值
            batch_mean, batch_variance = tf.nn.moments(Input, [0, 1, 2], keep_dims=False)
    
            decay = 0.99
            train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1 - decay))
            train_variance = tf.assign(pop_variance, pop_variance*decay + batch_variance*(1 - decay))
    
            with tf.control_dependencies([train_mean, train_variance]):
                return tf.nn.batch_normalization(Input, batch_mean, batch_variance, beta, gamma, epsilon,name='batch_norm_'+name)
    
        def batch_norm_inference():
            return tf.nn.batch_normalization(Input, pop_mean, pop_variance, beta, gamma, epsilon)
    
        batch_normalized_output = tf.cond(tf.cast(is_training, tf.bool), batch_norm_training, batch_norm_inference)
        return batch_normalized_output
    # 全连接层的bn函数
    def batch_norm_layer_1(self,Input,is_training,out_dim,name):
        gamma = tf.Variable(tf.ones([out_dim]))
        beta = tf.Variable(tf.zeros([out_dim]))
    
        pop_mean = tf.Variable(tf.zeros([out_dim]), trainable=False)
        pop_variance = tf.Variable(tf.ones([out_dim]), trainable=False)
    
        epsilon = 1e-3
    
        def batch_norm_training():
            batch_mean, batch_variance = tf.nn.moments(Input, [0])
    
            decay = 0.99
            train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1 - decay))
            train_variance = tf.assign(pop_variance, pop_variance*decay + batch_variance*(1 - decay))
    
            with tf.control_dependencies([train_mean, train_variance]):
                return tf.nn.batch_normalization(Input, batch_mean, batch_variance, beta, gamma, epsilon,name='batch_norm1_'+name)
    
        def batch_norm_inference():
            return tf.nn.batch_normalization(Input, pop_mean, pop_variance, beta, gamma, epsilon,name='batch_norm1_'+name)
    
        batch_normalized_output = tf.cond(tf.cast(is_training, tf.bool), batch_norm_training, batch_norm_inference)
        return batch_normalized_output
    # 卷积层  
    def conv_layer(self,x,conv_shape,bias_shape,strides,name,is_training,padding='SAME'):
        with tf.name_scope(name):
            weights = self.weights_variable(shape = conv_shape,name='weights_'+name)
            bias = self.biases_variable(shape=bias_shape,name='bias_'+name)
            conv = tf.nn.bias_add(self.conv2d(x,weights,strides=strides,padding=padding),bias)
            conv = self.batch_norm_layer(conv,is_training=is_training,out_channels=conv.get_shape().as_list()[-1],name=name)  #加入bn层
            out = tf.nn.relu(conv,name='conv_out_'+name)
        return out
    # 全连接层
    def fc_layer(self,x,d_in,d_out,name,activation=None,is_training=True):
        with tf.name_scope(name):
            weights = self.weights_variable(shape=[d_in,d_out],name='weights_'+name)
            bias = self.biases_variable([d_out],name='bias_'+name)
            out =tf.matmul(x,weights)+bias
            #out = self.batch_norm_layer_1(out,is_training=is_training,out_dim=d_out,name=name)  #加入bn层,实践证明不需要
            out = tf.nn.relu(out,name='fc_out_'+name)
        return out
    # maxpooling层
    def maxpool_layer(self,x,name,ksize,strides,padding='VALID'):
        with tf.name_scope(name):
            ksize = ksize
            strides = strides
            out = self.maxpool(x,ksize,strides,padding)
        return out

    def conv_block(self,x,shape,name,is_training,strides_=(1,1,1,1)):
        """
        resnet网络的conv_block函数注释
        :param x:block的输入
        :param shape: 该block的卷积层的各层卷积核的数目列表，例如resnet-50第一个block是[64,64,156], resnet-34的第一个block是[64,64]
        查看resnet各网络结构可以看出，resnet18与resnet-34的各个block的卷积是一样的区别在于每个stage的block数是不一样的，
        从而层数不一样，resnet-50、resnet-101与resnet-152是同样情况
        而且，shape也就是每个block的卷积核的数目列表的长度为2时，该block的卷积核大小都是3*3
        而shape长度为3时，卷积核大小依次为1*1，3*3，1*1，从而有下面的if len(shape)==2和if len(shape)==3
        :param name:网络block的name
        :param is_training:是否训练阶段
        :param strides: 主要是block第一个或第二个(长度为2时，第一个卷积可能步长为2，当长度为3时第二个卷积层步长可能为2)卷积层的步长
        其他的卷积层步长均为1
        """
        
        input_channel = x.get_shape().as_list()[-1]
        #print(strides_)
        #print('input_channel:'+str(input_channel))
        with tf.name_scope(name):
            if len(shape)==2:
                conv_shape = [3,3,input_channel,shape[0]]
                bias_shape = [shape[0]]
                strides = strides_
                net1 = self.conv_layer(x,conv_shape,bias_shape,strides=strides_,name=name+'_a')
                
                conv_shape = [3,3,shape[0],shape[1]]
                bias_shape = [shape[0]]
                strides = (1,1,1,1)
                net_out = self.conv_layer(net1,conv_shape,bias_shape,strides=strides,name=name+'_b',is_training=is_training)
                
            if len(shape)==3:
                conv_shape = [1,1,input_channel,shape[0]]
                bias_shape = [shape[0]]
                strides = (1,1,1,1)
                net1 = self.conv_layer(x,conv_shape,bias_shape,strides=strides,name=name+'_a',is_training=is_training)
                
                
                conv_shape = [3,3,shape[0],shape[1]]
                bias_shape = [shape[1]]
                strides = strides_
                net2 = self.conv_layer(net1,conv_shape,bias_shape,strides=strides_,name=name+'_b',is_training=is_training)
                
                conv_shape = [1,1,shape[1],shape[2]]
                bias_shape = [shape[2]]
                strides = (1,1,1,1)
                net_out = self.conv_layer(net2,conv_shape,bias_shape,strides=strides,name=name+'_c',is_training=is_training)
            #print('add_out:'+str(net_out.get_shape().as_list()))
            #print('add_in:'+str(x.get_shape().as_list()))
            if net_out.get_shape().as_list()[-2] == x.get_shape().as_list()[-2] and \
            net_out.get_shape().as_list()[-1] == x.get_shape().as_list()[-1]:
                x = tf.add(net_out, x)
                x = tf.nn.relu(x)
            else:
                x = self.conv_layer(x,conv_shape=[1,1,input_channel,shape[-1]],bias_shape =[shape[-1]], strides =strides_,name=name+'conv_input',is_training=is_training)
                x = tf.add(net_out, x)
                x = tf.nn.relu(x)               
            return x
