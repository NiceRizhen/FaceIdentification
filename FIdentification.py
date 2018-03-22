# -*- coding: utf-8 -*-

import cv2
import tensorflow as tf
import data_read
import network
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略警告

Image_w = 32
Image_h = 32


def run_training():
    data, label = data_read.allDataGet()
    data, label = data_read.get_batch(data, label, Image_w, Image_h,1, 1)

    print('Data got')
    xp = tf.placeholder(tf.float32, [1, Image_w, Image_h, 3])
    yp = tf.placeholder(tf.int32, [1])

    pred = network.CNN_layer(xp)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=yp)
    loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    print('model bulit')

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        print('start training')
        for step in range(30):
            print('step: ' + str(step))
            train_image, train_label = sess.run([data, label])
            [_ , lo] = sess.run([train_step,loss], feed_dict={xp : train_image, yp : train_label})
            print(lo)

        saver.save(sess, 'D:\model\Face.ckpt')

if __name__ == '__main__' :
    run_training()