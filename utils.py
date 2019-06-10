# -*- coding: utf-8 -*-
"""
Created on Thu May  9 09:43:24 2019

@author: Administrator
"""
import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
import random
#import create_and_read_TFRecord2 as reader2


def get_label_dict():
    label_dict, label_dict_res = {}, {}
    # 手动指定一个从类别到label的映射关系
    with open("labels_index_bk.txt", 'r') as f:
        for line in f.readlines():
            folder, label = line.strip().split(':')[0], line.strip().split(':')[1]
            label_dict[folder] = label
            label_dict_res[label] = folder
    print(label_dict)
    return label_dict
'''
def get_label_dict(filedir):
    i=0
    label_dict = {}
    for label in os.listdir(filedir):
        label_dict[label]=i
        i+=1
    return label_dict
'''

##################        获取数据集图像路径以及label的list       ###########
# label_dict: 字典，label相对应的标签
def get_files(file_dir):
    label_dict = get_label_dict()
    image_list, label_list = [], []
    for label in os.listdir(file_dir):
        for img in os.listdir(os.path.join(file_dir, label)):
            image_list.append(os.path.join(file_dir, label,img))
            label_list.append(int(label_dict[label]))
    print('There are %d data' %(len(image_list)))
    
    # 得到list之后需要shuffle
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    # 因为image_list是图像位置，是字符串，
    #所以label在transpose之后也会变为字符串，所以需要强制转换类型
    label_list = [int(i) for i in label_list]
    return label_dict, image_list, label_list

#############     数据读取    #########################
# tf.train.batch
# tf.image.random_flip_left_righ，tf.image.random_flip_up_down, tf.image.random_brightness, tf.image.random_contrast, tf.image.random_hue, tf.image.random_saturation
# tensorboard
# tf.summary.image

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label], shuffle=True)
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # 数据增强
    #image = tf.image.resize_image_with_pad(image, target_height=image_W, target_width=image_H)
    image = tf.image.resize_images(image, (image_W, image_H))
    # 随机左右翻转
   # image = tf.image.random_flip_left_right(image)
    # 随机上下翻转
    #image = tf.image.random_flip_up_down(image)
    # 随机设置图片的亮度
    #image = tf.image.random_brightness(image, max_delta=32/255.0)
    # 随机设置图片的对比度
    #image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    # 随机设置图片的色度
    #image = tf.image.random_hue(image, max_delta=0.05)
    # 随机设置图片的饱和度
    #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # 标准化,使图片的均值为0，方差为1
    #image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64,
                                                capacity = capacity)
    #tf.summary.image("input_img", image_batch, max_outputs=5)
    #print(label_batch.shape)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch

def gen_tfrecords(dirs_path,image_W, image_H,ratio = 0.9):
    writer= tf.python_io.TFRecordWriter("flower_train.tfrecords") #要生成的文件
    writer2=tf.python_io.TFRecordWriter('flower_test.tfrecords')
    label_dict, image_list, label_list = get_files(dirs_path)
    #print(image_list[0:10])
    print(label_list[0:10])
    #print(len(image_list))
    
    num = int(len(image_list)*ratio)
    
    for i in range(len(image_list)):
        img = Image.open(image_list[i])
        img = img.resize((image_W, image_H))
        img_raw=img.tobytes()#将图片转化为二进制格式
        label = int(label_list[i])
        
        example = tf.train.Example(features=tf.train.Features(feature={
                 "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                 'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                 })) #example对象对label和image数据进行封装
        if i < num:
            writer.write(example.SerializeToString())  #序列化为字符串
        else:
            writer2.write(example.SerializeToString())  #序列化为字符串

    writer.close()
    writer2.close()
    
def read_and_decode(filename,image_W, image_H,batch_size): # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename],num_epochs= None, shuffle=True)#生成一个queue队列
 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#将image数据和label取出来
 
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [image_W, image_H, 3])  #reshape为128*128的3通道图片
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #在流中抛出img张量
    label = tf.cast(features['label'], tf.int32) #在流中抛出label张量
    
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size = batch_size, capacity = 1000, 
                                                 min_after_dequeue = 64)
    return img_batch, label_batch

if __name__=='__main__':
    gen_tfrecords('./flower_photos',224,224)
    
    '''
    imgs, lbls = read_and_decode('flower_test.tfrecords',224,224,64)
    #img_batch, label_batch = tf.train.shuffle_batch([imgs, lbls], batch_size = 64, capacity = 1000, min_after_dequeue = 64)
    
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #saver.restore(sess, logs_train_dir+'/model.ckpt-174000') 
        try:
            for step in range(1000):
                if coord.should_stop():
                        break
                images, labels = sess.run([imgs, lbls])
                print(labels)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        
        coord.join(threads)
        sess.close()
    '''
    