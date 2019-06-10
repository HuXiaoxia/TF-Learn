# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:07:50 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import time

class VGG16:
    def __init__(self,is_training = True):
        #self.input_image = tf.layers.Input(dtype=tf.float32,shape=[None,None,3],name='input_images')
        #self.input_image = tf.placeholder(dtype=tf.float32,shape=[None,224,224,3],name='input_images')
        self.is_training = is_training
        #print(self.input_image.shape.as_list())
    def conv_layers(self, input_tensor):
        conv1_1 = tf.layers.conv2d(input_tensor, filters = 64, kernel_size = (3,3), strides =(1,1), \
                                   activation = 'relu', padding = 'same' , \
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), \
                                   bias_initializer=tf.constant_initializer(),\
                                   name = 'block1_conv1')
        
        conv1_2 = tf.layers.conv2d(conv1_1, filters = 64, kernel_size =(3,3), strides =(1,1),\
                                   activation = 'relu', padding = 'same', \
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), \
                                   bias_initializer=tf.constant_initializer(),\
                                   name = 'block1_conv2')
        pool1 = tf.layers.max_pooling2d(conv1_2,pool_size=(2,2), strides = (2,2), padding='valid', name = 'block1_pool')
        
        conv2_1 = tf.layers.conv2d(pool1, filters = 128, kernel_size = (3,3), strides = (1,1), \
                                   activation = 'relu', padding = 'same', 
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), \
                                   bias_initializer=tf.constant_initializer(),\
                                   name='block2_conv1')
        conv2_2 = tf.layers.conv2d(conv2_1, filters = 128, kernel_size = (3,3), strides = (1,1),\
                                   activation = 'relu', padding = 'same', \
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), \
                                   bias_initializer=tf.constant_initializer(),\
                                   name = 'block2_conv2')
        pool2 = tf.layers.max_pooling2d(conv2_2, pool_size = (2,2), strides = (2,2), padding = 'valid', name = 'block2_pool')
        
        conv3_1 = tf.layers.conv2d(pool2, filters = 256, kernel_size = (3,3), strides = (1,1), \
                                   activation = 'relu', padding = 'same', \
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), \
                                   bias_initializer=tf.constant_initializer(),\
                                   name = 'block3_conv1')
        conv3_2 = tf.layers.conv2d(conv3_1, filters = 256, kernel_size = (3,3), strides = (1,1),\
                                   activation = 'relu', padding = 'same',\
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), \
                                   bias_initializer=tf.constant_initializer(),\
                                   name = 'blocke3_conv2')
        conv3_3 = tf.layers.conv2d(conv3_2, filters = 256, kernel_size = (3,3), strides =(1,1),\
                                   activation ='relu', padding = 'same',\
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), \
                                   bias_initializer=tf.constant_initializer(),\
                                   name = 'block3_conv3')
        
        pool3 = tf.layers.max_pooling2d(conv3_3, pool_size = (2,2), strides = (2,2), padding = 'valid', name = 'block3_pool')
        
        conv4_1 = tf.layers.conv2d(pool3, filters = 512, kernel_size = (3,3), strides = (1,1),\
                                   activation = 'relu', padding = 'same', \
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), \
                                   bias_initializer=tf.constant_initializer(),\
                                   name ='block4_conv1')
        
        conv4_2 = tf.layers.conv2d(conv4_1, filters = 512, kernel_size = (3,3), strides = (1,1),\
                                   activation = 'relu', padding = 'same', \
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), \
                                   bias_initializer=tf.constant_initializer(),\
                                   name = 'block4_conv2')
        conv4_3 = tf.layers.conv2d(conv4_2, filters = 512, kernel_size = (3,3), strides =(1,1),\
                                   activation = 'relu', padding = 'same', \
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), \
                                   bias_initializer=tf.constant_initializer(),\
                                   name = 'block4_conv3')

        pool4 = tf.layers.max_pooling2d(conv4_3, pool_size = (2,2), strides = (2,2), padding = 'valid', name = 'block4_pool')
        
        conv5_1 = tf.layers.conv2d(pool4, filters  = 512, kernel_size = (3,3), strides = (1,1),
                                   activation = 'relu', padding = 'same', \
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), \
                                   bias_initializer=tf.constant_initializer(),\
                                   name = 'block5_conv1')
        
        conv5_2 = tf.layers.conv2d(conv5_1, filters = 512, kernel_size = (3,3), strides = (1,1),\
                                   activation ='relu', padding = 'same', \
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), \
                                   bias_initializer=tf.constant_initializer(),\
                                   name = 'block5_conv2')
        conv5_3 = tf.layers.conv2d(conv5_2, filters = 512, kernel_size = (3,3), strides = (1,1),\
                                   activation = 'relu', padding = 'same', \
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), \
                                   bias_initializer=tf.constant_initializer(),\
                                   name = 'block5_conv3')
        
        pool5 = tf.layers.max_pooling2d(conv5_3, pool_size = (2,2), strides = (2,2), padding = 'valid', name = 'block5_pool')
        print(pool5.shape.as_list())
        
        return pool5
    
    def fc_layers(self,input_tensor):
        flatten = tf.layers.Flatten()(input_tensor)
        #print(flatten.shape.as_list()[1])
        #shape = input_tensor.get_shape()
        #print(shape)
        #flattened_shape = shape[3]*shape[1]*shape[2]
        #flatten = tf.reshape(input_tensor, [-1, flattened_shape])
        
        fc1 = tf.layers.dense(flatten, 4096, name = 'fc1')
        
        fc1_dropout = tf.layers.dropout(fc1, 0.5)
        
        fc2 = tf.layers.dense(fc1_dropout, 4096, name = 'fc2')
        
        fc3 = tf.layers.dense(fc2, 1000, name = 'fc3')
        fc3_dropout = tf.layers.dropout(fc3, 0.5)
        #print(fc2.shape.as_list())
        return fc3_dropout
    
    def softmax_layer(self,input_tensor,num_class):
        out_put = tf.layers.dense(input_tensor, num_class,  name = 'softmax_class')
        #activation = tf.nn.sigmoid,
        return out_put
    
    def build_vgg16(self,input_image,num_class):
        net1 = self.conv_layers(input_image)
        net2 = self.fc_layers(net1)
        net3 = self.softmax_layer(net2, num_class)
        return net3
            
# 以上代码是VGG16的网络实现，至于为啥将卷积pooling层与全连接层分开，是因为
# 卷积层对图像大小没有要求，而全连接层有，然后这里可以看到，当我们将输入定义为：
# x = tf.placeholder(dtype = tf.float32, shape = [None,224,224,3], name = 'input_image')时，
# 定义网络的卷积pooling,全连接均没有问题
# 而改为：x = tf.placeholder(dtype = tf.float32, shape = [None,None,None,3], name = 'input_image')时
# 只有卷积层没有问题， 可以尝试下面的main代码
if __name__=='__main__':
    vgg16 = VGG16()
    '''
    x = tf.placeholder(dtype=tf.float32, shape=(None,224,224,3), name='input_images')
    num_class = 4
    net = vgg19.build_network(x,num_class)
    print(net.get_shape())
    '''
    x = tf.placeholder(dtype=tf.float32, shape=(None,224,224,3), name='input_images')
    num_class = 4
    net1 = vgg16.conv_layers(x)
    print(net1.get_shape())
    net2 = vgg16.fc_layers(net1)
    print(net2.get_shape())
    net3 = vgg16.softmax_layer(net2,num_class)
    print(net3.get_shape())
            
            
