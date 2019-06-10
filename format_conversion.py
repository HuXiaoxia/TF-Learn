# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:27:03 2019

@author: HuXiaoxia
"""

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
# ckpt转pb
def freeze_graph(input_checkpoint, output_graph,output_node_names):
    '''

    :param input_checkpoint: xxx.ckpt(千万不要加后面的xxx.ckpt.data这种，到ckpt就行了!)
    :param output_graph: PB模型保存路径，***.pb
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names =  output_node_names # 模型输入节点，根据情况自定义, 可通过tensorboard可视化查看
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph() # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
    
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) # 恢复图并得到数据
        # resnet或者有batch normalzition会出现：
        # ValueError: Input 0 of node vgg_16/conv1/conv1_1/BatchNorm/cond_1/AssignMovingAvg/Switch was passed float from vgg_16/conv1/conv1_1/BatchNorm/moving_mean:0 \
        # incompatible with expected float_ref.类似问题
        # 则需要在restore模型后加入：
        for node in input_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
                
            elif node.op == 'Assign':
                node.op = 'Identity'
                if 'use_locking' in node.attr: del node.attr['use_locking']
                if 'validate_shape' in node.attr: del node.attr['validate_shape']
                if len(node.input) == 2:
                    # input0: ref: Should be from a Variable node. May be uninitialized.
                    # input1: value: The value to be assigned to the variable.
                    node.input[0] = node.input[1]
                    del node.input[1]
        
        
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,# 等于:sess.graph_def
            output_node_names=output_node_names.split(","))# 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点
   

# pb---->saved model 没成功，需要再探究
def convert_pb_saved_model(model_path,out_put_path):
    model_path=model_path
    with tf.gfile.FastGFile(model_path, "rb") as f:
        graph = tf.get_default_graph()
        graph_def = graph.as_graph_def()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='graph')
        summaryWriter = tf.summary.FileWriter('log/', graph)
        
        result, x = tf.import_graph_def(graph_def,return_elements=["fc_out_fc3:0", "input_image:0"])
#        graph_def = tf.GraphDef()
#        graph_def.ParseFromString(f.read())
        
#        x =tf.get_default_graph().get_tensor_by_name("input_image:0")
#        result= tf.get_default_graph().get_tensor_by_name("fc_out_fc3:0")
        
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            builder = tf.saved_model.builder.SavedModelBuilder(out_put_path)
            signature = predict_signature_def(inputs={'myInput': x},
                                               outputs={'myOutput': result})
            builder.add_meta_graph_and_variables(sess=sess,
                                                  tags=[tf.saved_model.tag_constants.SERVING],
                                                  signature_def_map={'predict': signature})
            builder.save()     

if __name__=='__main__':
    model_path = 'vgg_flower.pb'
    out_put="./saved_models1/model_flower"
    convert_pb_saved_model(model_path,out_put)
    #freeze_graph('./models_saved/model.ckpt-9860','vgg_flower.pb','fc_out_fc3')
