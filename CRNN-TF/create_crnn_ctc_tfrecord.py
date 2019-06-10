# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:00:05 2019

@author: Administrator
"""

#create_crnn_ctc_tfrecord.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import cv2

import cfg
import tensorflow as tf

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _string_to_int(label):
    int_list = []
    for c in label:
        int_list.append(cfg.CHARACTERS.index(c))
    return int_list


def _write_tfrecord(dataset_split, path):
    if not os.path.exists(cfg.data_dir):
        os.makedirs(cfg.data_dir)

    tfrecords_path = os.path.join(cfg.data_dir, dataset_split + '.tfrecord')
    ims = os.listdir(path)
    random.shuffle(ims)
    with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
        for im in ims:
            label = im.split('_')[0]
            image = cv2.imread(path+'/'+im)
            if image is None:
                continue
            image = cv2.resize(image,(cfg._IMAGE_WIDTH,cfg._IMAGE_HEIGHT))
            is_success, image_buffer = cv2.imencode('.jpg', image)
            if not is_success:
                continue
            im = im if sys.version_info[0] < 3 else im.encode('utf-8') 
            features = tf.train.Features(feature={
                   'labels': _int64_feature(_string_to_int(label)),
                   'images': _bytes_feature(image_buffer.tostring()),
                   'imagenames': _bytes_feature(im)
                })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
        

def _convert_dataset():
    
    images_paths = {'train':cfg.images_train,'test':cfg.images_test}
    
    for dataset_split in ['train', 'test']:
        _write_tfrecord(dataset_split, images_paths[dataset_split])

def main(unused_argv):
    _convert_dataset()

if __name__ == '__main__':
    tf.app.run()