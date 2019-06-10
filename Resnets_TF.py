# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:48:50 2019

@author: Administrator
"""
import os
import sys
currentUrl = os.path.dirname(__file__)
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
sys.path.append(parentUrl)
import tensorflow as tf
from Basenet.Basenet import Basenet


class Resnet(Basenet):
    def __init__(self,num_layers,input_tensor,num_class,keep_prob=0.5,is_training=True):
        Basenet.__init__(self,input_tensor)
        self.is_training=is_training
        self.num_layers = num_layers
        self.num_class = num_class
        self.keep_prob = keep_prob
        
        self.kernels_channels = [[[64,64],[128,128],[256,256],[512,512]],
           [[64,64,256],[128,128,512],[256,256,1024],[512,512,2048]]]
        
        self.nums = [[2,2,2,2],[3,4,6,3],[3,4,6,3],[3,4,23,3],[3,8,36,3]]
        
        self.num_names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
             'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        
    def build_18layer(self):
        x = tf.pad(self.input_tensor,paddings=[[0,0],[3,3],[3,3],[0,0]])
        #print(x.get_shape().as_list())
        x = self.conv_layer(x,conv_shape=[7,7,3,64],bias_shape=[64],strides=(1,2,2,1),name='conv1',padding='VALID',is_training=self.is_training)
        #print(x.get_shape().as_list())
        x = self.maxpool_layer(x,name='maxpool1',ksize=(1,3,3,1),strides=(1,2,2,1),padding='SAME')
        #print(x.get_shape().as_list())
        for stage_i in range(len(self.nums[0])):
            stage_name = 'stage'+str(stage_i+2)
            num = self.nums[0][stage_i]
            shape = self.kernels_channels[0][stage_i]
            print('-'*20)
            #print(stage_name,num,shape)
            for i in range(num):
                name =stage_name+'_block_'+self.num_names[i]
                if stage_i !=0 and i ==0:
                    x = self.conv_block(x,shape=shape,name=name,strides_=(1,2,2,1),is_training=self.is_training)
                else:
                    x = self.conv_block(x,shape=shape,name= name,is_training=self.is_training)
                print(name)
                print(x.get_shape().as_list())
            #shape=self.kernels_channels[0][i]
        #x = self.conv_block1(x,[64,64],name='block1',strides=(1,2,2,1))
        #print(x.get_shape().as_list())
        return x
    
    def build_34layer(self):
        x = tf.pad(self.input_tensor,paddings=[[0,0],[3,3],[3,3],[0,0]])
        #print(x.get_shape().as_list())
        x = self.conv_layer(x,conv_shape=[7,7,3,64],bias_shape=[64],strides=(1,2,2,1),name='conv1',padding='VALID',is_training=self.is_training)
        #print(x.get_shape().as_list())
        x = self.maxpool_layer(x,name='maxpool1',ksize=(1,3,3,1),strides=(1,2,2,1),padding='SAME')
        #print(x.get_shape().as_list())
        for stage_i in range(len(self.nums[1])):
            stage_name = 'stage'+str(stage_i+2)
            num = self.nums[1][stage_i]
            shape = self.kernels_channels[0][stage_i]
            print('-'*20)
            #print(stage_name,num,shape)
            for i in range(num):
                name =stage_name+'_block_'+self.num_names[i]
                if stage_i !=0 and i ==0:
                    x = self.conv_block(x,shape=shape,name=name,strides_=(1,2,2,1),is_training=self.is_training)
                else:
                    x = self.conv_block(x,shape=shape,name= name,is_training=self.is_training)
                print(name)
                print(x.get_shape().as_list())
            #shape=self.kernels_channels[0][i]
        #x = self.conv_block1(x,[64,64],name='block1',strides=(1,2,2,1))
        #print(x.get_shape().as_list())
        return x
    
    def build_50layer(self):
        x = tf.pad(self.input_tensor,paddings=[[0,0],[3,3],[3,3],[0,0]])
        print(x.get_shape().as_list())
        x = self.conv_layer(x,conv_shape=[7,7,3,64],bias_shape=[64],strides=(1,2,2,1),name='conv1',padding='VALID',is_training=self.is_training)
        print(x.get_shape().as_list())
        x = self.maxpool_layer(x,name='maxpool1',ksize=(1,3,3,1),strides=(1,2,2,1),padding='SAME')
        print(x.get_shape().as_list())
        for stage_i in range(len(self.nums[2])):
            stage_name = 'stage'+str(stage_i+2)
            num = self.nums[2][stage_i]
            shape = self.kernels_channels[1][stage_i]
            print('-'*20)
            print(stage_name,num,shape)
            for i in range(num):
                name =stage_name+'_block_'+self.num_names[i]
                if stage_i !=0 and i ==0:
                    x = self.conv_block(x,shape=shape,name=name,strides_=(1,2,2,1),is_training=self.is_training)
                else:
                    x = self.conv_block(x,shape=shape,name= name,is_training=self.is_training)
                print(name)
                print(x.get_shape().as_list())
            #shape=self.kernels_channels[0][i]
        #x = self.conv_block1(x,[64,64],name='block1',strides=(1,2,2,1))
        #print(x.get_shape().as_list())
        return x
    
    def build_101layer(self):
        x = tf.pad(self.input_tensor,paddings=[[0,0],[3,3],[3,3],[0,0]])
        print(x.get_shape().as_list())
        x = self.conv_layer(x,conv_shape=[7,7,3,64],bias_shape=[64],strides=(1,2,2,1),name='conv1',padding='VALID',is_training=self.is_training)
        print(x.get_shape().as_list())
        x = self.maxpool_layer(x,name='maxpool1',ksize=(1,3,3,1),strides=(1,2,2,1),padding='SAME')
        print(x.get_shape().as_list())
        for stage_i in range(len(self.nums[3])):
            stage_name = 'stage'+str(stage_i+2)
            num = self.nums[3][stage_i]
            shape = self.kernels_channels[1][stage_i]
            print('-'*20)
            print(stage_name,num,shape)
            for i in range(num):
                name =stage_name+'_block_'+self.num_names[i]
                if stage_i !=0 and i ==0:
                    x = self.conv_block(x,shape=shape,name=name,strides_=(1,2,2,1),is_training=self.is_training)
                else:
                    x = self.conv_block(x,shape=shape,name= name,is_training=self.is_training)
                print(name)
                print(x.get_shape().as_list())
            #shape=self.kernels_channels[0][i]
        #x = self.conv_block1(x,[64,64],name='block1',strides=(1,2,2,1))
        #print(x.get_shape().as_list())
        return x
    
    def build_152layer(self):
        x = tf.pad(self.input_tensor,paddings=[[0,0],[3,3],[3,3],[0,0]])
        print(x.get_shape().as_list())
        x = self.conv_layer(x,conv_shape=[7,7,3,64],bias_shape=[64],strides=(1,2,2,1),name='conv1',padding='VALID',is_training=self.is_training)
        print(x.get_shape().as_list())
        x = self.maxpool_layer(x,name='maxpool1',ksize=(1,3,3,1),strides=(1,2,2,1),padding='SAME',is_training=self.is_training)
        print(x.get_shape().as_list())
        for stage_i in range(len(self.nums[4])):
            stage_name = 'stage'+str(stage_i+2)
            num = self.nums[4][stage_i]
            shape = self.kernels_channels[1][stage_i]
            print('-'*20)
            print(stage_name,num,shape)
            for i in range(num):
                name =stage_name+'_block_'+self.num_names[i]
                if stage_i !=0 and i ==0:
                    x = self.conv_block(x,shape=shape,name=name,strides_=(1,2,2,1),is_training=self.is_training)
                else:
                    x = self.conv_block(x,shape=shape,name= name,is_training=self.is_training)
                print(name)
                print(x.get_shape().as_list())
            #shape=self.kernels_channels[0][i]
        #x = self.conv_block1(x,[64,64],name='block1',strides=(1,2,2,1))
        #print(x.get_shape().as_list())
        return x
    def build_net(self):
        if self.num_layers ==18:
            resnet = self.build_18layer()
        elif self.num_layers==34:
            resnet = self.build_34layer()
        elif self.num_layers==50:
            resnet = self.build_50layer()
        elif self.num_layers ==101:
            resnet = self.build_101layer()
        elif self.num_layers==152:
            resnet = self.build_152layer()
        print('-'*20)
        shapes = resnet.get_shape().as_list()[1:]
        d_in = shapes[0]*shapes[1]*shapes[2]
        resnet = tf.reshape(resnet,[-1,d_in])
        print(resnet.get_shape().as_list())
        d_out = 1000
        print('-'*20)
        fc1 = self.fc_layer(resnet,d_in,d_out,name ='fc1')
        print(fc1.get_shape().as_list())
        fc1_drop=tf.nn.dropout(fc1,self.keep_prob,name="fc1_drop")
        print('-'*20)
        fc2 = self.fc_layer(fc1_drop,1000,self.num_class,name='fc_out')
        print(fc2.get_shape().as_list())
        return fc2
    
if __name__=='__main__':
    input_tensor = tf.placeholder(shape=[None,224,224,3],dtype=tf.float32,name='input_image')
    resnet = Resnet(num_layers=50,input_tensor=input_tensor,num_class=7)
    out = resnet.build_net()