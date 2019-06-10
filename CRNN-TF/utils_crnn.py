# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:17:04 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import cfg
import os
def _int_to_string(label_index):
    s = ''
    for i in range(len(label_index)):
        if label_index[i] < cfg.NUM_CLASS-1:
            s+=cfg.CHARACTERS[label_index[i]]
        else:
            continue
    return s

def _sparse_matrix_to_list(sparse_matrix):
    indices = sparse_matrix.indices
    values = sparse_matrix.values
    dense_shape = sparse_matrix.dense_shape

    dense_matrix =  len(cfg.CHARACTERS) * np.ones(dense_shape, dtype=np.int32)
    
    for i, indice in enumerate(indices):
        dense_matrix[indice[0], indice[1]] = values[i]
    string_list = []
    for row in dense_matrix:
        string_list.append(_int_to_string(row))
    return string_list

def _read_tfrecord(tfrecord_path, num_epochs=None):
    if not os.path.exists(tfrecord_path):
        raise ValueError('cannott find tfrecord file in path: {:s}'.format(tfrecord_path))

    filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'images': tf.FixedLenFeature([], tf.string),
                                           'labels': tf.VarLenFeature(tf.int64),
                                           'imagenames': tf.FixedLenFeature([], tf.string),
                                       })
    images = tf.image.decode_jpeg(features['images'])
    images.set_shape([32, 100, 3])
    images = tf.cast(images, tf.float32)
    labels = tf.cast(features['labels'], tf.int32)
    sequence_length = tf.cast(tf.shape(images)[-2] / 4 -1, tf.int32)
    imagenames = features['imagenames']
    return images, labels, sequence_length, imagenames


