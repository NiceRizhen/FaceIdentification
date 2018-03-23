# -*- coding: utf-8 -*-

import tensorflow as tf
import data_read

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')


def CNN_layer(input_image, output):
    Weight = {
        'con1' : tf.Variable(tf.truncated_normal([5, 5, 3, 16], stddev=0.1, dtype=tf.float32)),
        'con2' : tf.Variable(tf.truncated_normal([5,5,16,32], stddev=0.05, dtype=tf.float32)),
        'con3' : tf.Variable(tf.truncated_normal([5,5,32, 64],stddev=0.1, dtype=tf.float32)),
        'fc4' : tf.Variable(tf.truncated_normal([8*8*64,64],stddev=0.1, dtype=tf.float32)),
        'out5' : tf.Variable(tf.truncated_normal([64, output], stddev=0.1, dtype=tf.float32))
    }

    biases = {
        'bia1' : tf.Variable(tf.random_normal([16], stddev=0.1, dtype=tf.float32)),
        'bia2': tf.Variable(tf.random_normal([32], stddev=0.1, dtype=tf.float32)),
        'bia3' : tf.Variable(tf.random_normal([64], stddev=0.1, dtype=tf.float32)),
        'fc4': tf.Variable(tf.random_normal([64], stddev=0.1, dtype=tf.float32)),
        'out5': tf.Variable(tf.random_normal([output], stddev=0.1, dtype=tf.float32))
    }

    con1 = tf.nn.conv2d(input_image, Weight['con1'], strides = [1, 1, 1, 1], padding = 'SAME')   #64*64*16
    con1_relu = tf.nn.relu(con1 + biases['bia1'])
    pool1 = max_pool(con1_relu) #32*32*16

    con2 = tf.nn.conv2d(pool1, Weight['con2'], strides=[1,1,1,1], padding='SAME')   #32*32*32
    con2_relu = tf.nn.relu(con2 + biases['bia2'])
    pool2 = max_pool(con2_relu)  #16*16*32

    con3 = tf.nn.conv2d(pool2, Weight['con3'], strides=[1,1,1,1], padding='SAME')   #16*16*64
    con3_relu = tf.nn.relu(con3 + biases['bia3'])
    pool3 = max_pool(con3_relu)  #8*8*64

    pool3_flat = tf.reshape(pool3, [-1, 8*8*64])
    fc4 = tf.nn.relu(tf.matmul(pool3_flat, Weight['fc4']) + biases['fc4'])

    pred = tf.nn.sigmoid(tf.matmul(fc4, Weight['out5']) + biases['out5'])

    return pred