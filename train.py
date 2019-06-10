# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:19:47 2019

@author: Administrator
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
import numpy as np
from utils import get_files,get_batch,read_and_decode
from vgg16_models.vgg16_tflayers import VGG16
from vgg16_models.VGG16_TF import VGG
from Resnets_models.Resnets_TF import Resnet

train_dir = "./Dataset/Datasets_horror_normal/train"
test_dir =  "./Dataset/Datasets_horror_normal/val"
tf_train_records_path = './Dataset/flower_train.tfrecords'
tf_test_records_path = './Dataset/flower_test.tfrecords'

logs_dir = './logs'
model_dir = './models_saved1'
pb_model_dir = './pb_models'
init_lr = 1e-5
BATCH_SIZE = 40
IMG_W = 224
IMG_H = 224
CAPACITY = 32

label_dict, train, train_label = get_files(train_dir)
label_dict, test, test_label = get_files(test_dir)

one_epoch_step = int(len(train) / BATCH_SIZE)
decay_steps = 3*one_epoch_step
MAX_STEP = 20*one_epoch_step
N_CLASSES = len(label_dict)

config = tf.ConfigProto()
#config.gpu_options.allow_growth = True # 设置最小gpu使用量

def main():
    # 网络输入
    x = tf.placeholder(dtype = tf.float32, shape = [None,224,224,3], name = 'input_image')
    # 定义网络  
    #model =VGG16().build_vgg16(x,N_CLASSES)  # 用tf.layers实现的vgg网络
    #model = VGG(input_tensor=x,num_class=N_CLASSES,keep_prob=0.5).vgg16()  # 用tf.Variable实现的vgg网络
    model = Resnet(num_layers=50,input_tensor=x,num_class=N_CLASSES).build_net()  # 用tf.Variable实现的Resnet网络
    # 网络输出shape
    #print(model.get_shape())
    
    gloabl_steps = tf.Variable(tf.constant(0))    
    # label without one-hot
    
    #数据读取
    
    #batch_train, batch_labels = read_and_decode(tf_train_records_path,IMG_W,IMG_H,BATCH_SIZE)
    #test_images_batch,test_labels_batch  = read_and_decode(tf_test_records_path,IMG_W,IMG_H,BATCH_SIZE)
     
    
    batch_train, batch_labels = get_batch(train,train_label,IMG_W,IMG_H,BATCH_SIZE, CAPACITY)
    test_images_batch,test_labels_batch = get_batch(test,test_label,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)


    labels_true = tf.placeholder(tf.int32,shape = [None])
    #soft_max_out = tf.nn.softmax(model)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=labels_true )
    loss = tf.reduce_mean(cross_entropy, name='loss')
    tf.summary.scalar('train_loss', loss)
    # optimizer
    #
    lr = tf.train.exponential_decay(init_lr, gloabl_steps, decay_steps, decay_rate=0.9, staircase=True)
    #lr = tf.train.exponential_decay(0.1, gloabl_steps,gloabl_steps,0.9,staircase=True)
    tf.summary.scalar('learning_rate', lr)
 
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=gloabl_steps)
    # accuracy
    labels_one_hot = tf.one_hot(labels_true,N_CLASSES,1,0)
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.arg_max(labels_one_hot, 1))  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #correct = tf.nn.in_top_k(model, batch_labels, 1)
    #correct = tf.cast(correct, tf.float16)
    #accuracy = tf.reduce_mean(correct)
    tf.summary.scalar('train_acc', accuracy)
    
    #summary_op = tf.summary.merge_all()
    sess = tf.Session(config=config)
    train_writer = tf.summary.FileWriter(logs_dir, sess.graph)
    
    #sess.run(train_writer)
    
    saver = tf.train.Saver(max_to_keep=2,keep_checkpoint_every_n_hours=2)
    #var_list = tf.trainable_variables() 
    #g_list = tf.global_variables()
    #bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    #bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    #var_list += bn_moving_vars
    #saver = tf.train.Saver(var_list=var_list, max_to_keep=10)
    
    sess.run(tf.global_variables_initializer())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    flag_write_meta_graph = True
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                    break
            images, labels = sess.run([batch_train, batch_labels])
            #print(labels)
            model_out,_, learning_rate, tra_loss, tra_acc = sess.run([model,optimizer, lr, loss, accuracy],feed_dict={x:images, labels_true:labels})
            #print(model_out)
            if (step+1) % 50 == 0:
                print('Epoch %3d/%d, Step %6d/%d, lr %f, train loss = %.2f, train accuracy = %.2f%%' %(step/one_epoch_step, MAX_STEP/one_epoch_step, step, MAX_STEP, learning_rate, tra_loss, tra_acc*100.0))
                test_images, test_labels = sess.run([test_images_batch,test_labels_batch])
                l,acc = sess.run([ loss, accuracy], feed_dict = {x:test_images, labels_true: test_labels})
                print('Epoch %3d/%d, Step %6d/%d, test loss = %.2f, test accuracy = %.2f%%' %(step/one_epoch_step, MAX_STEP/one_epoch_step, step, MAX_STEP, l, acc*100.0))
                #summary_str = sess.run(summary_op)
                #train_writer.add_summary(summary_str, step)
            
            if (step+1) % 1000 == 0 or (step + 1) == MAX_STEP:
#                if step !=0:
#                    flag_write_meta_graph = True
                checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                saver.save(sess,checkpoint_path,global_step=step+1, write_meta_graph=flag_write_meta_graph)
            #spyder块注释，选取块，ctrl +4
            # 反块注释， ctrl +5
# =============================================================================
#                 # 写入序列化的 PB 文件
#                 #convert_variables_to_constants 需要指定output_node_names，list()，可以多个
#                 constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['softmax_class/BiasAdd'])
#                 with tf.gfile.FastGFile(pb_model_dir+'/model_'+str(step+1)+'.pb', mode='wb') as f:
#                      f.write(constant_graph.SerializeToString())
# =============================================================================
                

# =============================================================================
#                 # saved model 格式保存
#                 # 简单的saved model格式文件保存方法
#                 tf.saved_model.simple_save(sess,
#                                             "./saved_models/model_"+str(step),
#                                             inputs={"MyInput": x},
#                                             outputs={"MyOutput": model})
# 
#                 #复杂形式
#                 builder = tf.saved_model.builder.SavedModelBuilder("./saved_models1/model_"+str(step))
# 
#               
#                 signature = predict_signature_def(inputs={'myInput': x},
#                                                   outputs={'myOutput': model})
#                 builder.add_meta_graph_and_variables(sess=sess,
#                                                      tags=[tf.saved_model.tag_constants.SERVING],
#                                                      signature_def_map={'predict': signature})
#                 builder.save()
# =============================================================================
            
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()

if __name__=='__main__':
    main()
