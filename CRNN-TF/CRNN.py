# -*- coding: utf-8 -*-
"""
Created on Tue May 21 18:37:29 2019

@author: Administrator
"""
import os
import sys
currentUrl = os.path.dirname(__file__)
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
sys.path.append(parentUrl)


import tensorflow as tf
from Basenet.Basenet import Basenet


class CRNN(Basenet):
    def __init__(self,input_tensor,NUM_CLASSES,is_training=True):
        Basenet.__init__(self,input_tensor)
        self.is_training = is_training
        self.NUM_CLASSES = NUM_CLASSES
    def build_conv(self):
        conv1 = self.conv_layer(self.input_tensor,[3,3,3,64],[64],[1,1,1,1],'conv1',self.is_training,padding='SAME')
        print(conv1.get_shape())
        pool1 = self.maxpool_layer(conv1,'pool1',(1,2,2,1),(1,2,2,1))
        print(pool1.get_shape())
        conv2 = self.conv_layer(pool1,[3,3,64,128],[128],[1,1,1,1],'conv2',self.is_training)
        pool2 = self.maxpool_layer(conv2,'pool2',[1,2,2,1],(1,2,2,1),padding='SAME')
        print(pool2.get_shape())
        conv3 = self.conv_layer(pool2,[3,3,128,256],[256],[1,1,1,1],'conv3',self.is_training,'SAME')
        conv4 = self.conv_layer(conv3,[3,3,256,256],[256],[1,1,1,1],'conv4',self.is_training,'SAME')
        pool3 =self.maxpool_layer(conv4,'pool3',[1,2,1,1],[1,2,1,1],padding='SAME')
        print(pool3.get_shape())
        conv5 = self.conv_layer(pool3,[3,3,256,512],[512],[1,1,1,1],'conv5',self.is_training,'SAME')
        conv6 = self.conv_layer(conv5,[3,3,512,512],[512],[1,1,1,1],'conv6',self.is_training,'SAME')
        pool4 =self.maxpool_layer(conv6,'pool4',[1,2,1,1],[1,2,1,1],padding='SAME')
        print(pool4.get_shape())
        conv7 = self.conv_layer(pool4,[2,2,512,512],[512],[1,1,1,1],'conv7',self.is_training,'VALID')
        print(conv7.get_shape())
        conv7 = tf.squeeze(conv7,squeeze_dims=1)
        batch_size,_,out_dim = conv7.get_shape().as_list()
        print(batch_size)
        conv7_reshaped = tf.reshape(conv7,[-1,out_dim])
        fc = self.fc_layer(conv7_reshaped,512,self.NUM_CLASSES,'fc')
        print(fc.get_shape())
        fc_reshaped = tf.reshape(fc,[batch_size,-1,self.NUM_CLASSES])
        raw_pred = tf.argmax(tf.nn.softmax(fc_reshaped), axis=2, name='raw_prediction')
        # Swap batch and batch axis
        net_out = tf.transpose(fc_reshaped, (1, 0, 2), name='transpose_time_major')
        return raw_pred,net_out
    
if __name__=='__main__':
    batch_size = 64
    NUM_CLASSES = 100
    x = tf.placeholder(dtype = tf.float32, shape = [None,None,32,3], name = 'input_image')
    pred,net_out = CRNN(x,NUM_CLASSES).build_conv()
    print(pred.get_shape().as_list())
    print(net_out.get_shape().as_list())