# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:24:52 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
from CRNN import CRNN
import cfg
import os 
os.environ['CUDA_VISIBLE_DEVICES']='1'
import time
from utils_crnn import _read_tfrecord,_sparse_matrix_to_list

def main():
    input_image = tf.placeholder(dtype = tf.float32, shape = [cfg.Batch_size,32,None,3], name = 'input_image')
    input_labels = tf.sparse_placeholder(tf.int32, name='input_labels')
    input_sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[cfg.Batch_size], name='input_sequence_lengths')
    
    
    tfrecord_path = os.path.join(cfg.data_dir, 'train.tfrecord')
    images, labels, sequence_lengths, imagenames = _read_tfrecord(tfrecord_path=tfrecord_path)

    # decode the training data from tfrecords
    batch_images, batch_labels, batch_sequence_lengths, batch_imagenames = tf.train.batch(
            tensors=[images, labels, sequence_lengths, imagenames], batch_size=cfg.Batch_size, dynamic_pad=True,
            capacity=1000 + 2*cfg.Batch_size, num_threads=cfg.num_threads)
    
    crnn = CRNN(input_image,cfg.NUM_CLASS)
    with tf.variable_scope('CRNN_CTC', reuse=False):
        pred,net_out = crnn.build_conv()
        
    
    ctc_loss = tf.reduce_mean(
            tf.nn.ctc_loss(labels=input_labels, inputs=net_out, sequence_length=input_sequence_lengths,
                           ignore_longer_outputs_than_inputs=True))

    ctc_decoded, ct_log_prob = tf.nn.ctc_beam_search_decoder(net_out, input_sequence_lengths, merge_repeated=False)

    sequence_distance = tf.reduce_mean(tf.edit_distance(tf.cast(ctc_decoded[0], tf.int32), input_labels))

    global_step = tf.train.create_global_step()

    learning_rate = tf.train.exponential_decay(cfg.learning_rate, global_step, cfg.decay_steps, cfg.decay_rate, staircase=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=ctc_loss, global_step=global_step)
    
    init_op = tf.global_variables_initializer()

    # set tf summary
    tf.summary.scalar(name='CTC_Loss', tensor=ctc_loss)
    tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    tf.summary.scalar(name='Seqence_Distance', tensor=sequence_distance)
    merge_summary_op = tf.summary.merge_all()

    # set checkpoint saver
    saver = tf.train.Saver(max_to_keep=2)
    if not os.path.exists(cfg.model_dir):
        os.makedirs(cfg.model_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'crnn_ctc_ocr_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = os.path.join(cfg.model_dir, model_name)  

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        summary_writer = tf.summary.FileWriter(cfg.log_dir)
        summary_writer.add_graph(sess.graph)

        # init all variables
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(cfg.Max_strps):
            imgs, lbls, seq_lens = sess.run([batch_images, batch_labels, batch_sequence_lengths])

            _, cl, lr, sd, preds, summary = sess.run(
                [optimizer, ctc_loss, learning_rate, sequence_distance, ctc_decoded, merge_summary_op],
                feed_dict = {input_image:imgs, input_labels:lbls, input_sequence_lengths:seq_lens})
            #print(preds[0])
            #print(pred)
            #print('step:{} ctc_loss:{}'.format(step,cl))
            if (step + 1) % cfg.step_per_save == 0: 
                summary_writer.add_summary(summary=summary, global_step=step)
                saver.save(sess=sess, save_path=model_save_path, global_step=step)

            if (step + 1) % cfg.step_per_eval == 0:
                # calculate the precision
                preds = _sparse_matrix_to_list(preds[0])
                gt_labels = _sparse_matrix_to_list(lbls)

                accuracy = []

                for index, gt_label in enumerate(gt_labels):
                    pred = preds[index]
                    total_count = len(gt_label)
                    correct_count = 0
                    try:
                        for i, tmp in enumerate(gt_label):
                            if tmp == pred[i]:
                                correct_count += 1
                    except IndexError:
                        continue
                    finally:
                        try:
                            accuracy.append(correct_count / total_count)
                        except ZeroDivisionError:
                            if len(pred) == 0:
                                accuracy.append(1)
                            else:
                                accuracy.append(0)
                accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)

                print('step:{:d} learning_rate={:9f} ctc_loss={:9f} sequence_distance={:9f} train_accuracy={:9f}'.format(
                    step + 1, lr, cl, sd, accuracy))
            
        # close tensorboard writer
        summary_writer.close()

        # stop file queue
        coord.request_stop()
        coord.join(threads=threads)

    
        
if __name__=='__main__':
    main()    
        
