# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:49:56 2019

@author: Administrator
"""
from keras import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Input,Dense,Dropout,Flatten


class vgg:
    def __init__(self, input_tensor, num_class, locked_layers=True, weights =None, num_locked = 2, include_top = False):
        self.locked_layers = locked_layers
        self.weights = weights
        self.num_class = num_class
        self.input_tensor = input_tensor
        self.num_locked = num_locked
        self.include_top = include_top
        self.net_out = self.build_net(self.input_tensor)
        
    def build_net(self,input_tensor):
        vgg16 = VGG16(input_tensor=self.input_tensor,
                      weights= self.weights,
                      include_top=self.include_top)
        if self.locked_layers:
            # locked first two conv layers
            locked_layers = [vgg16.get_layer('block1_conv1'),
                             vgg16.get_layer('block2_conv2')]
            for layer in locked_layers:
                layer.trainable = False
        vgg16_out = vgg16.output
        vgg16_out_reshape =Flatten()(vgg16_out)
        fc1 = Dense(4096)(vgg16_out_reshape)
        # 增加 DropOut layer
        fc1_dropout = Dropout(0.5)(fc1)
        fc2 = Dense(1000)(fc1_dropout)
        fc2_dropout = Dropout(0.5)(fc2)
        fc_out = Dense(self.num_class,activation='softmax')(fc2_dropout)
        return Model(inputs=self.input_tensor, outputs=fc_out)
    
        
        #vgg16_reshaped = tf.reshape(vgg16,[-1,num])
        #fc1 = Dense(4096)(vgg16_reshaped)
if __name__=='__main__':
    x = Input(name='input_img',shape=(224, 224, 3),dtype='float32')
    vgg16 = vgg(x,5).net_out
    vgg16.summary()