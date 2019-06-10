# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:36:45 2019

@author: Administrator
"""
import os
import sys
currentUrl = os.path.dirname(__file__)
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
sys.path.append(parentUrl)

import tensorflow as tf
import numpy as np
from Basenet.Basenet import Basenet

class VGG(Basenet):
    def __init__(self,input_tensor,num_class,keep_prob=None, is_training=True):
        Basenet.__init__(self,input_tensor)
    
        self.keep_prob = keep_prob
        self.num_class = num_class
        self.is_training =is_training
        self.kernel_sizes={'block1_conv1':[3,3,3,64],'block1_conv2':[3,3,64,64]\
                           ,'block2_conv1':[3,3,64,128],'block2_conv2':[3,3,128,128]\
                           ,'block3_conv1':[3,3,128,256],'block3_conv2':[3,3,256,256]
                           ,'block3_conv3':[3,3,256,256],'block4_conv1':[3,3,256,512]\
                           ,'block4_conv2':[3,3,512,512],'block4_conv3':[3,3,512,512]\
                           ,'block5_conv1':[3,3,512,512],'block5_conv2':[3,3,512,512]\
                           ,'block5_conv3':[3,3,512,512]}
        self.bias_shapes = {'block1_conv1':[64],'block1_conv2':[64]\
                           ,'block2_conv1':[128],'block2_conv2':[128]\
                           ,'block3_conv1':[256],'block3_conv2':[256]
                           ,'block3_conv3':[256],'block4_conv1':[512]\
                           ,'block4_conv2':[512],'block4_conv3':[512]\
                           ,'block5_conv1':[512],'block5_conv2':[512]\
                           ,'block5_conv3':[512]}
        self.layers_name = ['block1_conv1','block1_conv2','block1_pool',\
                            'block2_conv1','block2_conv2','block2_pool',\
                            'block3_conv1','block3_conv2','block3_conv3','block3_pool',\
                            'block4_conv1','block4_conv2','block4_conv3','block4_pool',\
                            'block5_conv1','block5_conv2','block5_conv3','block5_pool']
        self.strides1=[1,1,1,1]
        self.strides2=[1,2,2,1]
        self.ksize = [1,2,2,1]
        
    
        #shape1 = x.get_shpae()[1]*shape[2]
    def vgg16(self):
        for i in range(len(self.layers_name)):
            print(self.layers_name[i])
            if i==0:
                net = self.conv_layer(self.input_tensor,conv_shape=self.kernel_sizes[self.layers_name[i]],\
                                      bias_shape=[self.kernel_sizes[self.layers_name[i]][-1]],strides=(1,1,1,1),\
                                      is_training=True,name=self.layers_name[i],padding='SAME')
            else:
                if self.layers_name[i].endswith('pool'):
                    net = self.maxpool_layer(net,name=self.layers_name[i],ksize=(1,2,2,1),strides=(1,2,2,1))
                else:
                    net = self.conv_layer(net,conv_shape=self.kernel_sizes[self.layers_name[i]],\
                                      bias_shape=[self.kernel_sizes[self.layers_name[i]][-1]],strides=(1,1,1,1),\
                                      is_training=True,name=self.layers_name[i],padding='SAME')
            print(net.shape)
            
        # fc1 layer
        # fc_layer(self,x,d_in,d_out,name,activation=None,is_training=True)
        num1 = int(np.prod(net.get_shape()[1:]))
        net = tf.reshape(net,[-1,num1])
        d_in=num1
        d_out=4096
        fc1 = self.fc_layer(net,d_in,d_out,name='fc1')
        
        fc1_drop=tf.nn.dropout(fc1,self.keep_prob,name="fc1_drop")
        print(fc1_drop.shape) 
        
        d_in = 4096
        d_out = 1000
        fc2 = self.fc_layer(fc1_drop,d_in,d_out,name='fc2')
        fc2_drop =tf.nn.dropout(fc2,self.keep_prob,name="fc2_drop")
        print(fc2_drop.shape) 
        
        d_in=1000
        d_out = self.num_class
        fc3 = self.fc_layer(fc2_drop,d_in,d_out,name='fc3')
        print(fc3.shape)        
        return fc3


if __name__=='__main__':
    x= tf.placeholder(shape=[None,224,224,3],dtype=tf.float32,name='input_image')
    net =VGG(input_tensor=x,num_class=5,keep_prob=0.5).vgg16()
    