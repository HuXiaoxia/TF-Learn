# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:33:58 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import cv2
import os

import cfg
from CRNN import CRNN
from utils_crnn import _read_tfrecord,_sparse_matrix_to_list

model_path = 'crnn_model'

image_path = './crnn_data/test/ADDIE_5977.jpg'

def main():
    
    #定义网络
    input_image = tf.placeholder(dtype=tf.float32, shape=[1, cfg._IMAGE_HEIGHT, None, 3])
    crnn = CRNN(input_image,cfg.NUM_CLASS,is_training=False)
    with tf.variable_scope('CRNN_CTC', reuse=False):
        pred,net_out = crnn.build_conv()
        
    input_sequence_length = tf.placeholder(tf.int32, shape=[1], name='input_sequence_length')

    ctc_decoded, ct_log_prob = tf.nn.ctc_beam_search_decoder(net_out, input_sequence_length, merge_repeated=True)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess,tf.train.latest_checkpoint(model_path))
        total = 0
        for im in os.listdir('./crnn_data/test'):
            # 图像预处理    
            print(im)
            image = cv2.imread('./crnn_data/test/'+im)
            h, w, c = image.shape
            height = cfg._IMAGE_HEIGHT
            width = int(w * height / h)
            image = cv2.resize(image, (width, height))
            image = np.expand_dims(image, axis=0)
            image = np.array(image, dtype=np.float32)
            # 长度
            seq_len = np.array([width / 4 -1], dtype=np.int32)
            ctc_decode  = sess.run([ctc_decoded],feed_dict={input_image:image,input_sequence_length:seq_len})
            #print(ctc_decode[0][0])
            preds = _sparse_matrix_to_list(ctc_decode[0][0])
            print(preds[0])
            if preds[0]==im.split('_')[0]:
                total+=1
                print(True)
            else:
                print(False)
        print(total)
        

if __name__=='__main__':
    main()