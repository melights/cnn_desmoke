from __future__ import division
from __future__ import print_function
# only keep warnings and errors
import os
import sys
import numpy as np
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from desmoke_net import *

original_image_path='/home/long/data/daVinci/train/image_0/'
smoke_image_path='/home/long/data/render_smokes2/image/3/'
filenames_file='train_7000.txt'
batch_size = 16
capacity =2048
training_nums=7553
height=128
width=256
num_epochs=100

def data_loader(smoke_path,orig_path, filenames):
    reader = tf.TextLineReader()
    input_queue = tf.train.string_input_producer([filenames], shuffle=False)
    _, line = reader.read(input_queue)
    smoke_images = tf.image.decode_png(tf.read_file(tf.string_join([smoke_path,line])), channels=3)
    smoke_images  = tf.image.convert_image_dtype(smoke_images,  tf.float32)
    smoke_images  = tf.image.resize_images(smoke_images,  [height, width], tf.image.ResizeMethod.AREA)
    original_images = tf.image.decode_png(tf.read_file(tf.string_join([orig_path,line])), channels=3)
    original_images  = tf.image.convert_image_dtype(original_images,  tf.float32)
    original_images  = tf.image.resize_images(original_images,  [height, width], tf.image.ResizeMethod.AREA)

    smoke_batch,original_batch=tf.train.batch([smoke_images, original_images], batch_size=batch_size, capacity=capacity)
    return smoke_batch,original_batch

def main():
        global_step = tf.Variable(0, trainable=False)
        # LEARNING RATE
        steps_per_epoch = np.ceil(training_nums / batch_size).astype(np.int32)
        num_total_steps=steps_per_epoch*num_epochs

        learning_rate = tf.train.exponential_decay(
        0.0001,                # Base learning rate.
        global_step,  # Current index into the dataset.
        steps_per_epoch,          # Decay step.
        0.9,                # Decay rate.
        staircase=True)

        #OPTIMIZER
        opt_step = tf.train.AdamOptimizer(learning_rate)

        # DATA
        smoke_image,original_image = data_loader(smoke_image_path,original_image_path, filenames_file)

        # MODEL
        model = DesmokeNet(smoke_image,original_image)
        total_loss = model.total_loss
        train_op = opt_step.minimize(total_loss,global_step=global_step)

        # SUMMARY
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('total_loss', total_loss)
        summary_op = tf.summary.merge_all()

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)
        # SAVER
        summary_writer = tf.summary.FileWriter('./model', sess.graph)
        train_saver = tf.train.Saver()

        # COUNT PARAMS 
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("number of trainable parameters: {}".format(total_num_parameters))

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        #TRAINING
        start_step = global_step.eval(session=sess)
        start_time = time.time()
        for step in range(start_step, num_total_steps):
            before_op_time = time.time()
            _, loss_value = sess.run([train_op, total_loss])
            duration = time.time() - before_op_time
            #print("Step: {} | examples/s: {:4.2f} ".format(step,batch_size/duration))
            if step and step % 100 == 0:
                examples_per_sec = batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar
                print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value, time_sofar, training_time_left))
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
            if step and step % 10000 == 0:
                train_saver.save(sess, './model/model', global_step=step)

        train_saver.save(sess, './model/model', global_step=num_total_steps)
if __name__ == '__main__':
    main()