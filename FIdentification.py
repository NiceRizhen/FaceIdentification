# -*- coding: utf-8 -*-

import tensorflow as tf
import data_read
import network

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略警告

Image_w = 64
Image_h = 64
output = 2

def run_training():
    data, label = data_read.allDataGet()
    data, label = data_read.get_batch(data, label, Image_w, Image_h, 15, 15)

    print('Data got')
    xp = tf.placeholder(tf.float32, [None, Image_w, Image_h, 3])
    yp = tf.placeholder(tf.int32, [None])

    pred = network.CNN_layer(xp, output)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=yp)
    loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.AdamOptimizer(0.5).minimize(loss)

    print('model bulit')

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('start training')
        for step in range(30):
            print('step: ' + str(step))
            train_image, train_label = sess.run([data, label])
            [_ , lo] = sess.run([train_step,loss], feed_dict={xp : train_image, yp : train_label})
            print('loss:',lo)

        coord.request_stop()
        coord.join(threads)
        saver.save(sess, 'D:\model\Face.ckpt')

        tm = data_read.image_get_face('D:\p4.png')

        pd = sess.run(pred,feed_dict={xp : tm})
        print(pd)

if __name__ == '__main__' :
    run_training()