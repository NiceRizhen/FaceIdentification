# -*- coding: utf-8 -*-

import cv2
import tensorflow as tf
import data_read

Image_w = 64
Image_h = 64
batch_size = 4
Capacity = 30


data, label = data_read.allDataGet()
data, label = data_read.get_batch(data, label, Image_w, Image_h, batch_size, Capacity)

xs = tf.placeholder(tf.float32, [batch_size, Image_w, Image_h, 3])
ys = tf.placeholder(tf.int32, [batch_size])

