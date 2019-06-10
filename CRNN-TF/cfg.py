# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:56:23 2019

@author: Administrator
"""
import os

Batch_size=128
learning_rate = 1e-4
decay_steps = 1000

step_per_eval = 10
decay_rate = 0.95
epoch = 10

_IMAGE_HEIGHT = 32
_IMAGE_WIDTH = 100

num_threads= 2



labels_path = './labels_characters.txt'
CHARACTERS=''
with open(labels_path,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        if not line.split('\n')[0]==None:
            #print(line.split('\n')[0])
            CHARACTERS +=line.split('\n')[0]
NUM_CLASS = len(CHARACTERS)+1

model_dir = 'crnn_model'
log_dir = 'logs'

data_dir = 'crnn_data'
images_train = 'crnn_data/train_data'
images_test = 'crnn_data/test'
ims = os.listdir(images_train)
ims_test = os.listdir(images_test)
total_train_images = len(ims)
total_test_images = len(ims_test)
#print(total_train_images)
#print(total_test_images)

Max_strps = 10 *(int(total_train_images/Batch_size))
step_per_save = int(total_train_images/Batch_size)
#print(Max_strps)