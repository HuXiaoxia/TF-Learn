# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:10:10 2019

@author: Huxx
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

from VGG16_TF import VGG

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import gfile

model_path='./models_saved'
N_CLASSES=5


img = Image.open('test.jpeg')
img = img.resize((224,224),Image.ANTIALIAS) 
img = np.asarray(img) / 255.0
img = img.reshape([-1,224,224,3])

#--------------------------------ckpt格式----------------------------------------------#
# =============================================================================
# with tf.Session() as sess:
#     print(tf.train.latest_checkpoint(model_path))
#     #####   ckpt（4个文件）模型加载的两种方式    #########
# #    saver  = tf.train.Saver()
# #    x = tf.placeholder(dtype = tf.float32, shape = [None,224,224,3], name = 'input_image')
# #    model = VGG(input_tensor=x,num_class=N_CLASSES,keep_prob=0.5).vgg16()  # 用tf.Variable实现的vgg网络
# #    sess.run(tf.global_variables_initializer())
#     
#     saver = tf.train.import_meta_graph(model_path+'/'+'model.ckpt-6000.meta') # .meta文件名
#     saver.restore(sess, tf.train.latest_checkpoint(model_path))
#     print('Done for loading ckpt model')
# #    var_list=tf.global_variables()
# #    output_node_names=[var_list[i].name for i in range(len(var_list))]
# #    print(output_node_names)
#     
# #    a = [ tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
# #    print(a)
#     #name_variable_to_restore = ''
#     
#     graph = tf.get_default_graph()
#     image = graph.get_tensor_by_name("input_image:0")
#     #Now, access the op that you want to run. 
#     op_to_restore = graph.get_tensor_by_name("softmax_class/BiasAdd:0")
#     op =  tf.nn.softmax(op_to_restore)
#     
#     c,d  = sess.run([op,op_to_restore],feed_dict={image:img})
#     print (c.shape)
#     print(d)
#     print(c)
# =============================================================================
    
# =============================================================================
#     #打印模型中参数值
#     # 第一种方式
#     checkpoint_path = tf.train.latest_checkpoint(model_path)
#     reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
#     var_to_shape_map = reader.get_variable_to_shape_map()
#     for key in var_to_shape_map:
#         print("tensor_name: ", key)
# #        print(reader.get_tensor(key)) # Remove this is you want to print only variable names
# #        print(reader.get_tensor(key).shape)
#     #w = graph.get_operation_by_name("word_embedding/W").outputs[0]
#     #print (sess.run(w))
#     # 
#     #第二种方式
#     w = graph.get_operation_by_name("softmax_class/kernel").outputs[0]
#     w = sess.run(w)
#     print(w)
#     print(w.shape)
#     
#     #第三种方式
#     from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#  
#     print_tensors_in_checkpoint_file(model_path+'/'+'model.ckpt-9000', #ckpt文件名字
#                  None, # 如果为None,则默认为ckpt里的所有变量
#                  False, # bool 是否打印所有的tensor，这里打印出的是tensor的值，一般不推荐这里设置为False
#                  True) # bool 是否打印所有的tensor的name
# #上面的打印ckpt的内部使用的是pywrap_tensorflow.NewCheckpointReader所以，掌握NewCheckpointReader才是王道
# =============================================================================
    
    
    
    
    
    
#--------------------------------pb格式----------------------------------------------#
    
# =============================================================================
# with tf.gfile.FastGFile("pb_models/model_1000.pb", "rb") as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     result, x = tf.import_graph_def(graph_def,return_elements=["softmax_class/BiasAdd:0", "input_image:0"])
# 
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     result = sess.run(result, feed_dict={x: img})
#     print(result)
#     
#     
# #重新加载模型文件，并使用Tensorboard进行可视化处理
# from tensorflow.python.platform import gfile
# model = 'pb_models/model_1000.pb'
# graph = tf.get_default_graph()
# graph_def = graph.as_graph_def()
# graph_def.ParseFromString(gfile.FastGFile(model, 'rb').read())
# #tf.import_graph_def(graph_def, name='graph')
# summaryWriter = tf.summary.FileWriter('log/', graph)
# =============================================================================


#-------------------------------saved model格式----------------------------------------------#

# =============================================================================
# with tf.Session(graph=tf.Graph()) as sess:
#   tf.saved_model.loader.load(sess, ["serve"], "./saved_models1/model_0")
#   graph = tf.get_default_graph()
#   summaryWriter = tf.summary.FileWriter('log/', graph)
#   x = sess.graph.get_tensor_by_name('input_image:0')
#   y = sess.graph.get_tensor_by_name('softmax_class/BiasAdd:0')
#   
#   c = sess.run(y,feed_dict={x:img})
#   print(c)
# =============================================================================

model_path="vgg_flower.pb"

with tf.gfile.FastGFile(model_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    result, x = tf.import_graph_def(graph_def,return_elements=["fc_out_fc3:0", "input_image:0"])
 
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    result = sess.run(result, feed_dict={x: img})
    print(result)
